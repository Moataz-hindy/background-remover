import gradio as gr
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BackgroundRemoval
from utils import postprocess_mask

# Handle spaces decorator for Hugging Face ZeroGPU or local fallback
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is None:
                return lambda f: f
            return func

# Determine device and load model globally at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_model(weights_path):
    model = BackgroundRemoval(use_residual=True)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ZeroGPU best practice: load model and place on device at module level
final_model = load_custom_model("models/res_model.pth").to(device)

transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

@spaces.GPU(duration=60)
def remove_background(input_image):
    if input_image is None:
        return None
        
    # Ensure input is 3-channel RGB
    if input_image.ndim == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_image.shape[2] == 4:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
    elif input_image.shape[2] == 1:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        
    orig_h, orig_w = input_image.shape[:2]
    
    # Cap maximum dimension to 1024 for responsive web processing
    MAX_DIM = 1024
    if max(orig_h, orig_w) > MAX_DIM:
        scale = MAX_DIM / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        input_image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        orig_h, orig_w = new_h, new_w
        
    # Prepare tensor for the neural network
    transformed = transform(image=input_image)
    tensor_img = transformed['image'].float().unsqueeze(0) / 255.0
    tensor_img = tensor_img.to(device)
    tensor_img_flipped = torch.flip(tensor_img, dims=[3])
    
    # Model inference with Test-Time Augmentation (TTA)
    with torch.inference_mode():
        logits1 = final_model(tensor_img)
        probs1 = torch.sigmoid(logits1)
        
        logits2 = final_model(tensor_img_flipped)
        probs2 = torch.flip(torch.sigmoid(logits2), dims=[3])
        
        avg_probs = (probs1 + probs2) / 2.0
        
    raw_mask = (avg_probs > 0.4).float().cpu()
    
    # Apply post-processing cleanup on the 256x256 mask
    clean_mask = postprocess_mask(
        raw_mask, 
        apply_morphology=True, 
        apply_contour_filling=True, 
        apply_largest_component=False, 
        apply_smoothing=True
    ) 
    
    # Resize mask back to image resolution
    high_res_mask = cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert mask to 0-255 uint8 alpha channel
    alpha_channel = (high_res_mask * 255).astype(np.uint8)
    
    # Merge RGB + Alpha to create a transparent PNG cutout
    r, g, b = cv2.split(input_image)
    cutout = cv2.merge((r, g, b, alpha_channel))
    
    return cutout

demo = gr.Interface(
    fn=remove_background,  
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Upload Image"),
    outputs=gr.Image(type="numpy", image_mode="RGBA", format="png", label="Background Removed"),
    title="U-Net Background Remover",
    description="Powered by a custom Residual U-Net built from scratch. Inference utilizes Test-Time Augmentation (TTA) and OpenCV Post-Processing."
)

if __name__ == "__main__":
    demo.launch()