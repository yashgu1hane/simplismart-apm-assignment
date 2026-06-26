import os
from dotenv import load_dotenv
load_dotenv()

from simplismart import DeploymentCreate, Simplismart

print("ORG_ID:", os.getenv("ORG_ID"))
print("MODEL_REPO_ID:", os.getenv("MODEL_REPO_ID"))

client = Simplismart()

deployment = client.create_deployment(
        DeploymentCreate(
            model_repo=os.getenv("MODEL_REPO_ID", "model-repo-uuid"),
            org=os.getenv("ORG_ID"),
            gpu_id="nvidia-h100",
            name="llama 3.1 8b",
            min_pod_replicas=1,
            max_pod_replicas=1,
            autoscale_config={"targets": [{"metric": "gpu", "target": 80}]},
            env_variables={},
            healthcheck={"path": "/", "port": 8000},
            ports={"http": {"port": 8000}},
            metrics_path=["/v1/chat/completions"],
            fast_scaleup=False,
            scale_to_zero_enabled=True,
            deployment_tag="v1.0",
            auth_enabled=True,
            extra_details={},
        
    )
) 


