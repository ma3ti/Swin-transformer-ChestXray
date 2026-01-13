import torch
import sys
import os

# Fix path to ensure we can import 'models'
sys.path.append(os.getcwd())

try:
    from models.swin_transformer import SwinTransformer
except ImportError:
    print("Error: Could not import SwinTransformer.")
    print("Make sure you are running this script from the 'Swin-Transformer' root folder.")
    sys.exit(1)

# 1. Setup Device (MPS for Mac)
# We use a try-except block because some older PyTorch versions on Mac might be fussy
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except AttributeError:
    device = torch.device("cpu")

print(f"Running on: {device}")  # <--- This was the line with the error

# 2. Define the Model (Swin-Tiny)
model = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7
)

# 3. Move model to GPU
model.to(device)
model.eval()

# 4. Create a dummy image (Batch size 1, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 5. Run Inference
try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("Success! The Swin Transformer is running on your Mac.")
except Exception as e:
    print(f"An error occurred during inference: {e}")