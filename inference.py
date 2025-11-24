import torch
import timm
import argparse
from PIL import Image
from torchvision import transforms
import os

def get_args():
    parser = argparse.ArgumentParser(description='Inference on EuroSAT images')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_type', type=str, default='swin', choices=['swin', 'resnet'], help='Model type (swin or resnet)')
    parser.add_argument('--model_path', type=str, default='best_swin_eurosat.pth', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu, cuda, mps)')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Class names
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                   'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    num_classes = len(class_names)

    # Model setup
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found locally.")
        print("Attempting to download from WandB...")
        try:
            import wandb
            run = wandb.init(project="eurosat-classification", job_type="inference")
            
            artifact_name = f"swin-eurosat:latest" if args.model_type == 'swin' else f"resnet50-eurosat:latest"
            artifact = run.use_artifact(artifact_name)
            artifact_dir = artifact.download()
            
            # The file name in the artifact might be different, but we uploaded it with the same name
            # Let's assume the downloaded file has the same name as the one we uploaded
            downloaded_file = os.path.join(artifact_dir, args.model_path)
            
            if os.path.exists(downloaded_file):
                print(f"Model downloaded to {downloaded_file}")
                args.model_path = downloaded_file # Update path to the downloaded file
            else:
                 # If the file name in artifact is different, try to find a .pth file
                files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
                if files:
                    args.model_path = os.path.join(artifact_dir, files[0])
                    print(f"Model found in artifact: {args.model_path}")
                else:
                    print("Error: Could not find .pth file in downloaded artifact.")
                    return
            
        except Exception as e:
            print(f"Error downloading model from WandB: {e}")
            print("Please ensure you have wandb installed and logged in, or provide a valid local model path.")
            return

    print(f"Loading {args.model_type} model from {args.model_path}...")
    if args.model_type == 'swin':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes, img_size=128)
    elif args.model_type == 'resnet':
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes) # img_size not needed for resnet creation
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        # Fix for torch.compile adding '_orig_mod.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model = model.to(device)
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prediction
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    try:
        img = Image.open(args.image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            max_prob, pred_idx = torch.max(probs, 1)
            
            pred_class = class_names[pred_idx.item()]
            confidence = max_prob.item()
            
        print(f"\nPrediction: {pred_class}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == '__main__':
    main()
