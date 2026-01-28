from hera.workflows import script, Workflow, Steps, Parameter, Resources, Container
from hera.shared import global_config
import os

# config
ns = os.getenv("ARGO_NAMESPACE")
sa = os.getenv("ARGO_ACCOUNT_NAME")
if not ns or not sa:
    raise RuntimeError(f"Missing ARGO_NAMESPACE/ARGO_ACCOUNT_NAME (ns={ns!r}, sa={sa!r})")

global_config.namespace = ns
global_config.service_account_name = sa

DASK_IMAGE = "444488357543.dkr.ecr.ap-southeast-2.amazonaws.com/easi-dask-cuda-torch:2025.09.0"

@script(
    image=DASK_IMAGE,
    resources=Resources(cpu_request=1, memory_request="2Gi"),
)
def create_cluster(
    cluster_name: str,
    namespace: str,
    service_account: str,
    dask_image: str,
    num_workers: int = 2,
    timeout: int = 900,
    # TODO: Add whatever else is needed
):
    from dask_kubernetes.operator import KubeCluster, make_cluster_spec
    from dask.distributed import Client
    
    print("Creating dask cluster...")
    
    # TODO: Add whatever other inputs here
    dask_spec = make_cluster_spec(
        name=cluster_name,
        image=dask_image,
        n_workers=num_workers,
    )
    
    # Important: Gives dask access to S3 bucket
    dask_spec["spec"]["scheduler"]["spec"]["serviceAccountName"] = service_account
    dask_spec["spec"]["worker"]["spec"]["serviceAccountName"] = service_account

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
# When libraries egt installed onto images, brake steps into seperate pods
# i.e. Cluster creation pod -> writer pod -> cluster close pod
def process_with_dask(
    config_path: str,
    cluster_name: str,
    namespace: str,
    dataset_name: str,
    num_workers: int = 2,
    tile_size: int = 224,
    bucket: str = "",
    base_prefix: str = "",
    branch: str = "main",
    resume: str = "true",
    easi_tools_git_ref: str = "main",
):
    import sys, subprocess
    from pathlib import Path
    import shutil

    # 1) Install regular deps on the Argo driver pod
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "zarr>=3.0.8", "icechunk"]
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

    # 4) Connect to the existing Dask cluster
    from dask_kubernetes.operator import KubeCluster
    from dask.distributed import Client

    cluster = KubeCluster.from_name(name=cluster_name, namespace=namespace)
    client = Client(cluster)

    # just incase
    client.wait_for_workers(n_workers=num_workers, timeout=900)

    print("\nCluster Dashboard:", client.dashboard_link)

    def _pip_install_icechunk():
        import sys, subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "zarr>=3.0.8", "icechunk"]
        )
        import icechunk, zarr
        return {"icechunk": icechunk.__version__, "zarr": zarr.__version__}

    # Runtime install on workers (temporary until baked into image)
    worker_versions = client.run(_pip_install_icechunk)
    sched_versions = client.run_on_scheduler(_pip_install_icechunk)

    # Now imports are safe
    from easi_tools.dask_helpers import (
        load_config, spec_from_config, make_catalog,
        GridRegionSampler, STACIceChunkBuilder
    )

    cfg = load_config(config_path)
    spec = spec_from_config(cfg)
    catalog = make_catalog(cfg)
    
    try:
        # Sometimes read cmd line inputs as strings
        resume_bool = resume.strip().lower() in ("true", "1", "yes", "y")
        sampler = GridRegionSampler(tile_size=tile_size)

        builder = STACIceChunkBuilder(
            catalog=catalog,
            bucket=bucket,
            base_prefix=base_prefix,
            dataset_name=dataset_name,
            spec=spec,
            sampler=sampler,
        )

        snapshot_id = builder.build(branch=branch, resume=resume_bool)
        print("Snapshot ID:", snapshot_id)
        print("Repo prefix:", builder.repo_prefix)
        
    finally:
        if client:
            print("Closing client...")
            client.close()
        # if cluster:
        #     print("Closing cluster...")
        #     cluster.close()



with Workflow(
    generate_name="eo-dataset-creation-",
    entrypoint="main",
    arguments=[
        Parameter(name="cluster_name", value="dask-{{workflow.uid}}"),
        Parameter(name="namespace", value="{{workflow.namespace}}"),
        Parameter(name="config_path", value="s3://your-bucket/config.json"),
        Parameter(name="dataset_name", value="training_dataset_v4"),
        Parameter(name="bucket", value="your-bucket"),
        Parameter(name="base_prefix", value="your/prefix"),
        Parameter(name="num_workers", value=2),
        Parameter(name="tile_size", value=224),
        Parameter(name="easi_tools_git_ref", value="main"),
        Parameter(name="dask_image", value=DASK_IMAGE),
        Parameter(name="dask_service_account_name", value=sa),
        Parameter(name="resume", value="true"),
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
        create_cluster(
            name="create-daskcluster",
            arguments={
                "cluster_name": "{{workflow.parameters.cluster_name}}",
                "namespace": "{{workflow.parameters.namespace}}",
                "service_account": "{{workflow.parameters.dask_service_account_name}}",
                "dask_image": "{{workflow.parameters.dask_image}}",
                "num_workers": "{{workflow.parameters.num_workers}}",
            },
        )

        process_with_dask(
            name="dataset-writer",
            arguments={
                "config_path": "{{workflow.parameters.config_path}}",
                "cluster_name": "{{workflow.parameters.cluster_name}}",
                "namespace": "{{workflow.parameters.namespace}}",
                "dataset_name": "{{workflow.parameters.dataset_name}}",
                "num_workers": "{{workflow.parameters.num_workers}}",
                "tile_size": "{{workflow.parameters.tile_size}}",
                "bucket": "{{workflow.parameters.bucket}}",
                "base_prefix": "{{workflow.parameters.base_prefix}}",
                "easi_tools_git_ref": "{{workflow.parameters.easi_tools_git_ref}}",
                "resume": "{{workflow.parameters.resume}}",
            },
        )

        # Normal-path cleanup (still keep on_exit as backup)
        delete_cluster(name="delete-daskcluster-step")

print(w.to_yaml())
