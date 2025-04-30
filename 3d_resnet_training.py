import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pytorchvideo.transforms import ShortSideScale
from torch.utils.data import DataLoader, random_split
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from tqdm import tqdm

from dependencies.dataloader import ImglistToTensor, VideoFrameDataset


def build_model(device):
    print("Loading SlowR50 model from PyTorchVideo...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    NUM_CLASSES = 2
    num_features = model.blocks[-1].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_features, NUM_CLASSES)

    for param in model.blocks[5].proj.parameters():
        param.requires_grad = True

    print("\nTrainable parameters:")
    trainable_params_found = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}")
            trainable_params_found = True


    # Set to GPU or CPU
    device = "cpu"
    # model = model.eval()
    model = model.to(device)
    model

    return model

def create_dataset(root, annotation_file, preprocess):
    print(f"Creating VideoFrameDataset from {root}...")
    return VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        num_segments=32,
        frames_per_segment=1,
        imagefile_template='{:05d}.jpg',
        transform=preprocess,
        test_mode=False
    )

def create_data_loaders(dataset, batch_size, train_ratio=0.8):
    print("Splitting dataset into training and validation sets...")
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"Dataset size: {dataset_size}, Train size: {train_size}, Validation size: {val_size}")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Creating DataLoaders with batch size {batch_size}...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate batch statistics
        batch_loss = loss.item()
        running_loss += batch_loss * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Calculate current accuracy
        current_acc = 100 * correct / total
        
        # Update progress bar with batch metrics
        progress_bar.set_description(f'Batch {i+1}/{len(loader)} | Loss: {batch_loss:.4f} | Acc: {current_acc:.2f}%')
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate batch statistics
            batch_loss = loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            progress_bar.set_description(f'Val Loss: {batch_loss:.4f} | Val Acc: {current_acc:.2f}%')
    
    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total
    
    return val_loss, val_acc

def main():
    print("Starting 3D ResNet training pipeline...")
    
    # Configuration
    print("Initializing configuration...")
    device = "cpu"
    print(f"Using device: {device}")
    
    # Dataset parameters
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    side_size = 256
    crop_size = 256
    num_frames = 32
    sampling_rate = 8
    frames_per_second = 30
    batch_size = 16
    num_epochs = 10
    
    # Paths
    root = 'data/images/train_500/rgb'
    annotation_file = os.path.join(root, '../annotations.txt')
    print(f"Data root: {root}")
    print(f"Annotation file: {annotation_file}")
    
    # Create preprocessing transform
    print("Setting up preprocessing transforms...")
    preprocess = transforms.Compose([
        ImglistToTensor(),
        NormalizeVideo(mean=mean, std=std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size=(crop_size, crop_size))
    ])
    
    # Build model
    model = build_model(device)
    
    # Create dataset and loaders
    dataset = create_dataset(root, annotation_file, preprocess)
    train_loader, val_loader = create_data_loaders(dataset, batch_size)
    
    # Define loss function and optimizer
    print("Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print(f"Beginning training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        print("Training phase...")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        print("Validation phase...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} Results:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('New best model saved!')
        
        print("-" * 30)
    
    print("Training complete!")

if __name__ == "__main__":
    main()