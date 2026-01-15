import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm 

# Fake distributed environment variables for Swin config
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

sys.path.append(os.path.abspath('..'))

# Import from Swin project
from config import get_config
from models import build_model

# Check for Apple Metal (MPS) acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal acceleration (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

class Args:
    cfg = "../configs/swin/chest_xray_finetune.yaml" 
    opts = ["MODEL.NUM_CLASSES", "2"] 
    batch_size = 32
    data_path = "" # Not needed here, defined below
    zip = False
    cache_mode = "part"
    pretrained = ""
    resume = ""
    accumulation_steps = 1
    use_checkpoint = False
    amp_opt_level = ""
    output = "output"
    tag = "default"
    eval = True         # Important: eval mode
    throughput = False
    enable_amp = False
    fused_window_process = False
    fused_layernorm = False
    optim = "adamw"
    local_rank = 0

def main():
    # Setup Config and Model
    args = Args()
    config = get_config(args)
    
    # Build the model
    print(f"Creating model {config.MODEL.NAME}...")
    model = build_model(config)
    
    # Load "Best" weights
    # Make sure the path is correct!
    checkpoint_path = "../output/swin_tiny_patch4_window7_224/v2_clean/ckpt_best.pth" 
    
    if not os.path.exists(checkpoint_path):
        # Fallback if you used the default path
        checkpoint_path = "../output/ckpt_best.pth" 
        
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle potential key discrepancies (e.g. 'module.')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loading result: {msg}")

    # Move model to device (MPS or CPU)
    model.to(device)
    model.eval()

    # --------------------------------------------------------------------------
    # 2. Prepare Dataset and Dataloader
    # --------------------------------------------------------------------------
    # Path to test folder containing NORMAL and PNEUMONIA subfolders
    test_dir = "../dataset/chest_xray/test" 
    
    print(f"Loading test dataset from: {test_dir}")

    # Same transforms used during training/validation
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Classes found: {test_dataset.classes}")
    # ImageFolder sorts alphabetically: 0=NORMAL, 1=PNEUMONIA

    # --------------------------------------------------------------------------
    # 3. Evaluation Loop
    # --------------------------------------------------------------------------
    all_preds = []
    all_targets = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        # Use tqdm for progress bar if available, otherwise use loader directly
        try:
            loader_iter = tqdm(test_loader)
        except ImportError:
            loader_iter = test_loader

        for images, targets in loader_iter:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # --------------------------------------------------------------------------
    # 4. Final Report
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL RESULTS ON TEST SET")
    print("="*50)
    
    # Classification Report (Precision, Recall, F1-Score)
    # target_names must match the alphabetical order of folders
    print(classification_report(all_targets, all_preds, target_names=test_dataset.classes, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nMatrix Details:")
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Normal): {tn}")
    print(f"False Positives (Pneumonia): {fp}")
    print(f"False Negatives (Normal - Dangerous!): {fn}")
    print(f"True Positives (Pneumonia): {tp}")
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"\nTotal Accuracy: {accuracy*100:.2f}%")

    print("\nSaving results for visualization...")    
    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)
    
    # Removed the leading slash '/' to avoid permission errors on Mac
    np.save('results_preds.npy', preds_np)
    np.save('results_targets.npy', targets_np)
    
    print("Done! Files 'results_preds.npy' and 'results_targets.npy' saved.")

if __name__ == "__main__":
    main()