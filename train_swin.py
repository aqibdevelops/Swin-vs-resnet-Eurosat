import torch
import torch.nn as nn
import torch.optim as optim
import timm
import argparse
import time
from tqdm import tqdm
from utils import get_dataloaders, calculate_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, scheduler=None, dry_run=False):
    since = time.time()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in tqdm(dataloader, desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if dry_run:
                    break
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_swin_eurosat.pth')
        
        if scheduler:
            scheduler.step()
        
        if dry_run:
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Swin Transformer on EuroSAT')
    parser.add_argument('--data_dir', type=str, default='data/eurosat/2750', help='Path to dataset')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs (head only)')
    parser.add_argument('--finetune_epochs', type=int, default=40, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch for debugging')
    
    args = parser.parse_args()
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
        
    # Data loaders
    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Model setup
    # Using swin_tiny_patch4_window7_224 and resizing input to 128x128
    # We need to specify img_size=128 so the model adapts the positional embeddings
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes, img_size=128)
    model = model.to(device)
    
    # Phase 1: Warmup (Head only)
    print(f"Starting Phase 1: Warmup (Head only) for {args.warmup_epochs} epochs...")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
        
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.warmup_epochs, dry_run=args.dry_run)
    
    # Phase 2: Full Fine-tuning
    print(f"Starting Phase 2: Full Fine-tuning for {args.finetune_epochs} epochs...")
    for p in model.parameters():
        p.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.finetune_epochs, scheduler=scheduler, dry_run=args.dry_run)

    # Evaluation
    print("Running final evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if args.dry_run:
                break
                
    from utils import get_metrics
    report, cm = get_metrics(all_labels, all_preds, class_names)
    print("\nClassification Report:\n")
    print(report)
    print("\nConfusion Matrix:\n")
    print(cm)

if __name__ == '__main__':
    main()
