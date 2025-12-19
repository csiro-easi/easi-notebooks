"""
Generate the Argo YAML with:
  python hera-dask_test.py > dask_s3_smoketest.yaml

Submit with (example):
  argo submit dask_s3_smoketest.yaml -p bucket=my-bucket
"""

import os

from hera.shared import global_config
from hera.workflows import (
    Workflow,
    Steps,
    Parameter,
    Resource,
    Container,
    Resources,
    models as m,
)
from hera.workflows.models import Metadata


# Global Argo config 
global_config.host = os.getenv("ARGO_HOST", "")
global_config.namespace = os.getenv("ARGO_NAMESPACE", "")
global_config.service_account_name = os.getenv("ARGO_ACCOUNT_NAME", "")

# Images supplied via env (portable across environments)
DRIVER_IMAGE = os.getenv("JUPYTER_IMAGE_SPEC", "")
WAIT_IMAGE = os.getenv("WAIT_IMAGE", DRIVER_IMAGE)  # default: reuse driver image for waiting
owner = os.environ['JUPYTERHUB_USER'].replace('@', '__AT__')
team = os.environ['DASK_GATEWAY__CLUSTER__OPTIONS__EASI_USER_ALLOCATION'].strip("'")



# Workflow config
generate_name = "dask-s3-smoketest-"
entrypoint = "main"
on_exit = "cleanup"

# Tune as needed
WAIT_RESOURCES = Resources(cpu_request=0.1, cpu_limit=0.2, memory_request="64Mi", memory_limit="128Mi")
DRIVER_RESOURCES = Resources(cpu_request=0.5, cpu_limit=1, memory_request="512Mi", memory_limit="1Gi")


with Workflow(
    generate_name=generate_name,
    entrypoint=entrypoint,
    on_exit=on_exit,
    pod_metadata=Metadata(labels={"owner": owner, "team": team}),
    arguments=[
        Parameter(name="bucket"),
        Parameter(name="s3_key", value="argo-smoketest/dask_s3_test.txt"),
        Parameter(name="aws_region", value="us-west-2"),
        Parameter(name="dask_cluster_name", value="dask-{{workflow.uid}}"),
        Parameter(name="dask_namespace", value="{{workflow.namespace}}"),
        Parameter(name="worker_replicas", value="2"),
    ],
) as w:
    # Templates
    create_cluster = Resource(
        name="create-dask-cluster",
        action="create",
        set_owner_reference=True,  # helps cleanup if workflow is deleted early
        manifest="""
            apiVersion: kubernetes.dask.org/v1
            kind: DaskCluster
            metadata:
              name: "{{workflow.parameters.dask_cluster_name}}"
              namespace: "{{workflow.parameters.dask_namespace}}"
            spec:
              worker:
                replicas: {{workflow.parameters.worker_replicas}}
            """,
                )

    # Python-based wait, easier then dask wait
    wait_for_scheduler = Container(
        name="wait-for-scheduler",
        image=WAIT_IMAGE,
        command=["sh", "-lc"],
        args=[
            r"""
            set -euo pipefail
            
            python - <<'PY'
            import os, socket, time
            
            host = os.environ["DASK_SCHEDULER_HOST"]
            port = int(os.environ.get("DASK_SCHEDULER_PORT", "8786"))
            
            deadline = time.time() + 180  # 3 minutes
            while time.time() < deadline:
                try:
                    with socket.create_connection((host, port), timeout=2):
                        print(f"Scheduler reachable at {host}:{port}")
                        raise SystemExit(0)
                except OSError:
                    time.sleep(2)
            
            raise SystemExit(f"Scheduler not reachable at {host}:{port}")
            PY
            """
        ],
        env=[
            m.EnvVar(
                name="DASK_SCHEDULER_HOST",
                value="{{workflow.parameters.dask_cluster_name}}-scheduler.{{workflow.parameters.dask_namespace}}",
            ),
            m.EnvVar(name="DASK_SCHEDULER_PORT", value="8786"),
        ],
        resources=WAIT_RESOURCES,
    )

    run_test = Container(
        name="run-dask-s3-test",
        image=DRIVER_IMAGE,
        command=["bash", "-lc"],
        args=[
            r"""
            set -euo pipefail
            
            python - <<'PY'
            import os
            import boto3
            from dask.distributed import Client
            
            bucket = os.environ["BUCKET"]
            key = os.environ["S3_KEY"]
            addr = os.environ["DASK_SCHEDULER_ADDRESS"]
            
            client = Client(addr)
            
            future = client.submit(lambda x: x * x, 7)
            result = future.result()
            
            s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
            s3.put_object(Bucket=bucket, Key=key, Body=str(result).encode("utf-8"))
            
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read().decode("utf-8")
            print("Read back from s3:", data)
            
            client.close()
            PY
            """
        ],
        env=[
            m.EnvVar(name="AWS_REGION", value="{{workflow.parameters.aws_region}}"),
            m.EnvVar(name="BUCKET", value="{{workflow.parameters.bucket}}"),
            m.EnvVar(name="S3_KEY", value="{{workflow.parameters.s3_key}}"),
            m.EnvVar(
                name="DASK_SCHEDULER_ADDRESS",
                value="tcp://{{workflow.parameters.dask_cluster_name}}-scheduler.{{workflow.parameters.dask_namespace}}:8786",
            ),
        ],
        resources=DRIVER_RESOURCES,
    )

    delete_cluster = Resource(
        name="delete-dask-cluster",
        action="delete",
        manifest="""
            apiVersion: kubernetes.dask.org/v1
            kind: DaskCluster
            metadata:
              name: "{{workflow.parameters.dask_cluster_name}}"
              namespace: "{{workflow.parameters.dask_namespace}}"
            """,
    )

    # Entrypoint steps
    with Steps(name="main"):
        create_cluster(name="create")
        wait_for_scheduler(name="wait")
        run_test(name="test")

    # onExit handler
    with Steps(name="cleanup"):
        delete_cluster(name="delete")


print(w.to_yaml())
