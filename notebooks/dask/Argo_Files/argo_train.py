from hera.workflows import script, Workflow, Steps, Parameter, Resources, Container
from hera.shared import global_config
import os

# config
#global_config.host = os.getenv("ARGO_HOST", "")
ns = os.getenv("ARGO_NAMESPACE")
sa = os.getenv("ARGO_ACCOUNT_NAME")
if not ns or not sa:
    raise RuntimeError(f"Missing ARGO_NAMESPACE/ARGO_ACCOUNT_NAME (ns={ns!r}, sa={sa!r})")

global_config.namespace = ns
global_config.service_account_name = sa

DASK_IMAGE = "444488357543.dkr.ecr.ap-southeast-2.amazonaws.com/easi-dask-cuda-torch:2025.09.0"


# 1. set up dask cluster
# 2. install dependancies on cluster
# 3. install easi-tools on both argo pod and cluster
# 3b. Figure out best way to include model type?
# 4. launch trainign job
# 5. Return S3 location?

@script(
    image=DASK_IMAGE,
    resources=Resources(cpu_request=1, memory_request="2Gi"),
)
def create_gpu_cluster(
    cluster_name: str,
    namespace: str,
    service_account: str,
    dask_image: str,
    num_workers: int = 2,
    timeout: int = 900,
    # >1/worker not working yet
    gpus_per_worker: int = 1,
):
    from dask_kubernetes.operator import KubeCluster, make_cluster_spec
    from dask.distributed import Client
    
    print("Creating dask cluster...")
    
    # TODO: Add whatever other inputs here
    dask_spec = make_cluster_spec(
        name=cluster_name,
        image=dask_image,
        n_workers=num_workers,
        worker_command="dask-cuda-worker",
    )
    
    # Important: Gives dask access to S3 bucket
    dask_spec["spec"]["scheduler"]["spec"]["serviceAccountName"] = service_account
    dask_spec["spec"]["worker"]["spec"]["serviceAccountName"] = service_account

    # dask_spec["spec"]["worker"]["spec"]["containers"][0]["resources"] = {
    #     "limits": {"nvidia.com/gpu": gpus_per_worker}
    # }
    # Add GPU limits to the *worker* container only
    worker0 = dask_spec["spec"]["worker"]["spec"]["containers"][0]
    worker0.setdefault("resources", {}).setdefault("limits", {})
    worker0["resources"]["limits"]["nvidia.com/gpu"] = gpus_per_worker

    cluster = KubeCluster(
        custom_cluster_spec=dask_spec,
        namespace=namespace,
        shutdown_on_close=False,
    )
    client = Client(cluster)

    try:
        client.wait_for_workers(n_workers=num_workers, timeout=timeout)
        print("\nCluster ready", client.dashboard_link)
    finally:
        client.close()



@script(
    image=DASK_IMAGE,
    resources=Resources(cpu_request=1, memory_request="2Gi"),
)
def submit_to_dask(
    cluster_name: str,
    namespace: str,
    repo_prefix: str,
    s3_bucket: str,
    output_dir: str,
    log_dir: str,
    model_path: str,
    model_name: str,
    num_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    num_workers: int = 2,
    step_size: int = 5,
    gamma: int = 0.1,
    easi_tools_git_ref: str = "main",
):
    from easi_tools.dask_training import (
        GeoBatchSpec,
        BatchAdapter,
        get_num_samples,
        make_dali_iterator,
        train_one_epoch,
        evaluate,
        SegmentationTask,
        wrap_model_with_fsdp,
        train_one_epoch_log,
        WorldCoverLabelMapper,
        load_latest_training_log,
        test_model,
        inspect_timing
        )
    from urllib.parse import urlparse
    import boto3, os
    
    def _s3join(prefix: str, name: str) -> str:
        return prefix.rstrip("/") + "/" + name
    
    def train(
        repo_prefix,
        s3_bucket,
        output_dir,
        log_dir,
        num_epochs,
        batch_size,
        learning_rate,
        model_name,
        snapshot_id=None,
        checkpoint_file=None,
        enable_timing_logs=False,
        val_split: float = 0.2,
        seed: int = 42,
        step_size: int = 5,
        gamma: int = 0.1,
    ):
        import torch
        import numpy as np
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
            FullOptimStateDictConfig,
        )
        from sklearn.model_selection import train_test_split
        import io, boto3, json
        from datetime import datetime
        from s3torchconnector import S3Checkpoint
        from torch.optim.lr_scheduler import StepLR
        import importlib

        start_epoch = 0
        num_classes = 11

        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # ---- S3 keys ----
        best_key = _s3join(output_dir, "model_best.pth")
        latest_key = _s3join(output_dir, "model_latest.pth")
        latest_meta_key = _s3join(output_dir, "model_latest.json")

        # ---- Data split (same as yours) ----
        label_mapper = WorldCoverLabelMapper(num_classes=num_classes)
        n = get_num_samples(s3_bucket, repo_prefix, snapshot_id=snapshot_id)
        indices = np.arange(n)

        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=seed, shuffle=True
        )

        train_idx_rank = train_idx[rank::world_size]
        val_idx_rank = val_idx[rank::world_size]

        spec = GeoBatchSpec(x_key="features", y_key="labels")
        label_lut = label_mapper.make_lut(255)
        adapter = BatchAdapter(label_lut=label_lut)

        train_dl = make_dali_iterator(
            base_indices=train_idx_rank,
            batch_size=batch_size,
            device_id=local_rank,
            bucket=s3_bucket,
            repo_prefix=repo_prefix,
            snapshot_id=snapshot_id,
            shuffle=True,
            spec=spec,
            adapter=adapter,
        )
        val_dl = make_dali_iterator(
            base_indices=val_idx_rank,
            batch_size=batch_size,
            device_id=local_rank,
            bucket=s3_bucket,
            repo_prefix=repo_prefix,
            snapshot_id=snapshot_id,
            shuffle=False,
            spec=spec,
            adapter=adapter,
        )

        # ---- Model / FSDP ----
        # CHange model as needed
        mod = importlib.import_module("user_model_def") 
        ModelCls = getattr(mod, model_name)  # "PrithviSegmentation"
        # EDIT here for changing class stuff
        base_model = ModelCls(num_classes=num_classes).to(device)
        model, optimizer = wrap_model_with_fsdp(base_model, lr=learning_rate)

        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        # EDIT here for differnt task
        task = SegmentationTask(num_classes=num_classes, class_weights=None)

        val_loss_min = float("inf")
        checkpoint_connection = S3Checkpoint(region="ap-southeast-2")

        # Ensure FSDP uses FULL state dict conventions for both save and load
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True),
            FullOptimStateDictConfig(rank0_only=True),
        )  # recommended API for FSDP state dict handling

        # ---- Optional checkpoint restore (FSDP-correct) ----
        if checkpoint_file:
            # Read on all ranks (simple, avoids rank0-only edge cases)
            with checkpoint_connection.reader(f"s3://{s3_bucket}/{checkpoint_file}") as reader:
                checkpoint = torch.load(reader, map_location="cpu")

            # Configure FSDP to load FULL model + FULL optim state on all ranks
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False),
                FullOptimStateDictConfig(rank0_only=False),
            )  # API used to control model+optimizer checkpoint formats [page:4]

            # Load model weights into the FSDP wrapper
            model.load_state_dict(checkpoint["model"])

            start_epoch = int(checkpoint["epoch"]) + 1
            val_loss_min = float(checkpoint.get("val_loss_min", val_loss_min))

        # ---- Logging (same behavior) ----
        all_logs = []

        for epoch in range(start_epoch, num_epochs):
            if enable_timing_logs:
                train_stats = train_one_epoch_log(
                    model, optimizer, scheduler, task, train_dl, device, log_every=5
                )
            else:
                train_stats = train_one_epoch(model, optimizer, scheduler, task, train_dl, device)

            val_stats = evaluate(model, task, val_dl, device)

            log = {
                "worker_rank": rank,
                "world_size": world_size,
                "local_rank": local_rank,
                "epoch": epoch,
                "start_time": datetime.now().isoformat(),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_stats["loss"]),
                "train_accuracy": float(train_stats["accuracy"]),
                "val_loss": float(val_stats["loss"]),
                "val_accuracy": float(val_stats["accuracy"]),
                "end_time_val": datetime.now().isoformat(),
                "repo_prefix": repo_prefix,
                "snapshot_id": snapshot_id,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
            if enable_timing_logs:
                log.update({
                    "t_fetch": float(train_stats.get("t_fetch", 0.0)),
                    "t_to_device": float(train_stats.get("t_to_device", 0.0)),
                    "t_step": float(train_stats.get("t_step", 0.0)),
                    "timing_samples": int(train_stats.get("timing_samples", 0)),
                    "num_batches": int(train_stats.get("num_batches", 0)),
                })

            # keep epoch boundaries aligned
            dist.barrier()

            # compute "best" decision consistently
            is_best = float(log["val_loss"]) < float(val_loss_min)
            if is_best:
                val_loss_min = float(log["val_loss"])

            # ---- Build full state dict (collective; call on all ranks) ----
            full_state = model.state_dict()  # under FULL_STATE_DICT type set earlier

            # ---- Build optimizer state dict correctly for FSDP (collective) ----
            full_optim_state = FSDP.optim_state_dict(model, optimizer)

            if rank == 0:
                all_logs.append(log)

                checkpoint = {
                    "model": full_state,
                    "optimizer": full_optim_state,
                    "epoch": epoch,
                    "val_loss_min": float(val_loss_min),
                }

                # Always write latest
                with checkpoint_connection.writer(f"s3://{s3_bucket}/{latest_key}") as writer:
                    torch.save(checkpoint, writer)

                boto3.client("s3").put_object(
                    Body=json.dumps({"latest": latest_key, "epoch": epoch}).encode("utf-8"),
                    Bucket=s3_bucket,
                    Key=latest_meta_key,
                )

                # Write best if improved
                if is_best:
                    with checkpoint_connection.writer(f"s3://{s3_bucket}/{best_key}") as writer:
                        torch.save(checkpoint, writer)

        # ---- Write single JSONL log file at end (same behavior) ----
        if rank == 0 and all_logs:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_key = _s3join(log_dir, f"training_logs_{run_id}.jsonl")

            buf = io.StringIO()
            for row in all_logs:
                buf.write(json.dumps(row) + "\n")

            boto3.client("s3").put_object(
                Body=buf.getvalue().encode("utf-8"),
                Bucket=s3_bucket,
                Key=jsonl_key,
            )

    import sys, subprocess
    from pathlib import Path
    import shutil

    packages = [
        'dask-pytorch-ddp',
        'zarr>=3.0.8',
        'icechunk',
        'nvidia-dali-cuda120'
        ]

    # 1) Install regular deps on the Argo driver pod
    # TODO: Check if argo pod can survive without installing some of these
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    )

    # 2) Sparse-checkout only easi_tools/ from the repo
    # TODO: Change when fork gets merged or if added to image
    repo_url = "https://github.com/cg379/easi-notebooks.git"
    clone_dir = Path("/tmp/easi-notebooks")

    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--no-checkout", "--branch", easi_tools_git_ref, repo_url, str(clone_dir)]
    )
    subprocess.check_call(["git", "-C", str(clone_dir), "sparse-checkout", "init", "--cone"])
    subprocess.check_call(["git", "-C", str(clone_dir), "sparse-checkout", "set", "easi_tools"])
    subprocess.check_call(["git", "-C", str(clone_dir), "checkout"])

    # 3) Make easi_tools importable
    sys.path.insert(0, str(clone_dir))

    from dask_kubernetes.operator import KubeCluster
    from dask.distributed import Client
    from dask_pytorch_ddp import dispatch

    cluster = KubeCluster.from_name(name=cluster_name, namespace=namespace)
    client = Client(cluster)

    # just incase
    client.wait_for_workers(n_workers=num_workers, timeout=900)

    print("\nCluster Dashboard:", client.dashboard_link)

    def _pip_install_libs(packages):
        import sys, subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages]
        )
        import icechunk, zarr
        return {"icechunk": icechunk.__version__, "zarr": zarr.__version__}

    # Runtime install on workers (temporary until baked into image)
    worker_versions = client.run(_pip_install_libs, packages)
    sched_versions = client.run_on_scheduler(_pip_install_libs, packages)


    try:
        from urllib.parse import urlparse
        import boto3, os
        
        def download_s3_to_local(s3_uri: str, local_path: str):
            u = urlparse(s3_uri)
            assert u.scheme == "s3"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            boto3.client("s3").download_file(u.netloc, u.path.lstrip("/"), local_path)
            return local_path
        
        local_model_py = download_s3_to_local(model_path, "/tmp/user_model_def.py")
        client.upload_file(local_model_py)
        # Stop an error that sometimes happens with large clusters
        sched_identity = client.run_on_scheduler(lambda dask_scheduler: dask_scheduler.identity())
        _real_scheduler_info = client.scheduler_info
        client.scheduler_info = lambda: sched_identity
        # Send it off
        futs = dispatch.run(
            client,
            train,
            repo_prefix=repo_prefix,
            s3_bucket=s3_bucket,
            output_dir=output_dir,
            log_dir=log_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
            step_size=step_size,
            gamma=gamma,)

        client.gather(futs)

    finally:
        client.scheduler_info = _real_scheduler_info
        if client:
            print("Closing client...")
            client.close()


with Workflow(
    generate_name="training-job-",
    entrypoint="main",
    arguments=[
        Parameter(name="cluster_name", value="dask-{{workflow.uid}}"),
        Parameter(name="namespace", value="{{workflow.namespace}}"),
        Parameter(name="repo_prefix", value="your/prefix"),
        Parameter(name="s3_bucket", value="your-bucket"),
        Parameter(name="output_dir", value="models/experiment-1"),
        Parameter(name="log_dir", value="logs/experiment-1"),
        Parameter(name="num_workers", value=2),
        Parameter(name="easi_tools_git_ref", value="main"),
        Parameter(name="dask_image", value=DASK_IMAGE),
        Parameter(name="dask_service_account_name", value=sa),
        Parameter(name="model_path", value=""),
        Parameter(name="model_name", value="PrithviSegmentation"),
        Parameter(name="num_epochs", value=2),
        Parameter(name="batch_size", value=8),
        Parameter(name="learning_rate", value=0.001),
        Parameter(name="step_size", value=5),
        Parameter(name="gamma", value=0.1),
    ],
    namespace=global_config.namespace,
    service_account_name=global_config.service_account_name,
) as w:

    delete_cluster = Container(
        name="delete-daskcluster",
        image="bitnami/kubectl:latest",
        command=["sh", "-c"],
        args=[r'''kubectl delete daskclusters.kubernetes.dask.org "{{workflow.parameters.cluster_name}}" \
-n "{{workflow.parameters.namespace}}" --ignore-not-found=true || true'''],
    )

    # Backup cleanup if something fails mid-way
    w.on_exit = delete_cluster

    with Steps(name="main"):
        create_gpu_cluster(
            name="create-daskcluster",
            arguments={
                "cluster_name": "{{workflow.parameters.cluster_name}}",
                "namespace": "{{workflow.parameters.namespace}}",
                "service_account": "{{workflow.parameters.dask_service_account_name}}",
                "dask_image": "{{workflow.parameters.dask_image}}",
                "num_workers": "{{workflow.parameters.num_workers}}",
            },
        )

        submit_to_dask(
            name="dataset-writer",
            arguments={
                "cluster_name": "{{workflow.parameters.cluster_name}}",
                "namespace": "{{workflow.parameters.namespace}}",
                "repo_prefix": "{{workflow.parameters.repo_prefix}}",
                "s3_bucket": "{{workflow.parameters.s3_bucket}}",
                "output_dir": "{{workflow.parameters.output_dir}}",
                "log_dir": "{{workflow.parameters.log_dir}}",
                "easi_tools_git_ref": "{{workflow.parameters.easi_tools_git_ref}}",
                "num_epochs": "{{workflow.parameters.num_epochs}}",
                "batch_size": "{{workflow.parameters.batch_size}}",
                "learning_rate": "{{workflow.parameters.learning_rate}}",
                "num_workers": "{{workflow.parameters.num_workers}}",
                "model_path": "{{workflow.parameters.model_path}}",
                "model_name": "{{workflow.parameters.model_name}}",
                "step_size": "{{workflow.parameters.step_size}}",
                "gamma": "{{workflow.parameters.gamma}}",
            },
        )

        # Normal-path cleanup (still keep on_exit as backup)
        delete_cluster(name="delete-daskcluster-step")

print(w.to_yaml())
