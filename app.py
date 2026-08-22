import spaces  
import gradio as gr
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BackgroundRemoval
from utils import postprocess_mask

# -------------------------------------------------------------
# 1. LOAD BOTH MODELS ON CPU
# -------------------------------------------------------------
def load_custom_model(weights_path):
    model = BackgroundRemoval()
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# CHANGE THESE FILENAMES TO MATCH YOUR UPLOADED .PTH FILES
model_v1 = load_custom_model("models/res_model.pth")       # Experiment 6
model_v2 = load_custom_model("models/final_model.pth")  # Experiment 8


# Define the transform
transform = A.Compose([
    A.Resize(height=384, width=384),
    ToTensorV2(),
])

# -------------------------------------------------------------
# 2. ISOLATE THE GPU WORK (Now with Ensembling!)
# -------------------------------------------------------------
@spaces.GPU
def run_model_on_gpu(tensor_img, model_choice):
    tensor_img = tensor_img.to("cuda")
    tensor_img_flipped = torch.flip(tensor_img, dims=[3])
    
    # Helper function to do TTA for a specific model
    def get_model_prediction(model):
        model.to("cuda")
        with torch.inference_mode():
            logits1 = model(tensor_img)
            probs1 = torch.sigmoid(logits1)
            
            logits2 = model(tensor_img_flipped)
            probs2 = torch.flip(torch.sigmoid(logits2), dims=[3]) 
            
            avg_probs = (probs1 + probs2) / 2.0
        model.to("cpu") # Free GPU memory immediately
        return avg_probs

    # Choose what to run based on the Dropdown
    if model_choice == "Model A: The Portrait Specialist":
        final_probs = get_model_prediction(model_v1)
        
    elif model_choice == "Model B: The Generalist (Harsh Lighting)":
        final_probs = get_model_prediction(model_v2)
        
    else: # THE ENSEMBLE (Run both and average them!)
        probs_A = get_model_prediction(model_v1)
        probs_B = get_model_prediction(model_v2)
        final_probs = (probs_A + probs_B) / 2.0
        
    # Apply the threshold to the final probabilities
    raw_mask = (final_probs > 0.4).float().cpu() 
    
    return raw_mask

# -------------------------------------------------------------
# 3. CPU MAIN FUNCTION: Handles the massive 4K arrays instantly
# -------------------------------------------------------------
def remove_background(input_image, model_choice):
    if input_image is None:
        return None
        
    orig_h, orig_w = input_image.shape[:2]
    
    # 1. Fast resize down to 384x384
    transformed = transform(image=input_image)
    tensor_img = transformed['image'].float().unsqueeze(0) / 255.0
    
    # 2. Send the tiny tensor and model choice to the GPU
    raw_mask = run_model_on_gpu(tensor_img, model_choice)
    
    # 3. Clean the tiny mask on the CPU
    clean_mask = postprocess_mask(raw_mask) 
    
    # 4. Scale the clean mask back up to original 4K/1080p resolution
    high_res_mask = cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    high_res_mask_3d = np.expand_dims(high_res_mask, axis=-1)
    
    # 5. Apply the mask to the image
    cutout = (input_image * high_res_mask_3d).astype(np.uint8)
    
    return cutout

# -------------------------------------------------------------
# 4. CREATE THE GRADIO UI
# -------------------------------------------------------------
demo = gr.Interface(
    fn=remove_background,  
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Radio(
            choices=[
                "Model A: The Portrait Specialist", 
                "Model B: The Generalist (Harsh Lighting)",
                "Model C: The Ensemble (Averages A + B)" 
            ],
            value="Model C: The Ensemble (Averages A + B)", # Default selection
            label="Select Model Version"
        )
    ],
    outputs=gr.Image(type="numpy", label="Background Removed"),
    title="U-Net Background Remover",
    description="A Custom ResUNet built from scratch. Compare the Specialist, the Generalist, and the Ensemble!"
)

demo.launch()