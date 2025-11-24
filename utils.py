import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    Creates training and validation data loaders for the EuroSAT dataset.
    
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for training.
        val_split (float): Fraction of data to use for validation.
        num_workers (int): Number of worker threads for data loading.
        
    Returns:
        train_loader, val_loader, class_names
    """
    # EuroSAT images are 64x64, so we keep them as is (or resize if needed)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, class_names

def calculate_accuracy(outputs, labels):
    """
    Calculates the accuracy of the model predictions.
    """
    _, preds = torch.max(outputs, 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds), dtype=torch.float32)

from sklearn.metrics import classification_report, confusion_matrix

def get_metrics(y_true, y_pred, class_names):
    """
    Calculates classification metrics.
    
    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        class_names (list): List of class names.
        
    Returns:
        report (str): Classification report.
        cm (array): Confusion matrix.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
