import wandb
import os

# Configuration
PROJECT_NAME = "eurosat-classification"
ENTITY = None # Optional: set to your username or team name if needed

def upload_model(file_path, model_name, model_type):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Uploading {model_name}...")
    run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="upload-model", name=f"upload-{model_name}")
    
    artifact = wandb.Artifact(name=model_name, type="model", description=f"Pretrained {model_type} model for EuroSAT")
    artifact.add_file(file_path)
    
    run.log_artifact(artifact)
    run.finish()
    print(f"Successfully uploaded {model_name}")

if __name__ == "__main__":
    # Upload ResNet50
    upload_model("best_resnet50_eurosat.pth", "resnet50-eurosat", "resnet50")
    
    # Upload Swin Transformer
    upload_model("best_swin_eurosat.pth", "swin-eurosat", "swin-tiny")
