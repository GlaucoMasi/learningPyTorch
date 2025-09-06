import os
import torch
import gradio as gr

from typing import Tuple, Dict
from model import create_effnet2_model
from timeit import default_timer as timer
from safetensors.torch import load_file, safe_open
from huggingface_hub import hf_hub_download, snapshot_download

class_names = ["pizza", "steak", "sushi"]

effnetb2, effnetb2_transforms = create_effnet2_model(num_classes=101)

MODEL_REPO = "glaucomasi/gastrovision_mini_effnetb2_20percent"
FILENAME   = "09_pretrained_effnetb2_20_percent.safetensors"
weights_path = "/home/glauco/Desktop/projects/learningPyTorch/trained_models/effnetb2_food101.safetensors"

# weights_path = hf_hub_download(
#     repo_id=MODEL_REPO,
#     filename=FILENAME,
#     local_dir=".",                 # optional: place a real copy into your app folder
#     local_dir_use_symlinks=False,  # real file instead of symlink (handy for debug)
# )



print("== Debug: file existence ==")
print("Exists?", os.path.exists(weights_path))
if os.path.exists(weights_path):
    print("File size (bytes):", os.path.getsize(weights_path))
    # Peek first bytes
    with open(weights_path, "rb") as f:
        head = f.read(200)
    print("First 200 bytes:", head[:200])
else:
    print("File not found at:", weights_path)

# Try safetensors open
try:
    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
        print("== Debug: safetensors keys sample ==")
        print(keys[:10])
except Exception as e:
    print("== Debug: safetensors load error ==")
    print(repr(e))

sd = load_file(weights_path)
effnetb2.load_state_dict(sd, strict=True)
effnetb2.eval()



def predict(img) -> Tuple[Dict, float]:
    start_time = timer()

    img = effnetb2_transforms(img).unsqueeze(dim=0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Gradio's required format: a dictionary with the probability for each class
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer()-start_time, 5)

    return pred_labels_and_probs, pred_time



title = "Food101 Vision Model"
description = "An EfficientNetB2 feature extractor computer vision model achieving '65%' accuracy on Food101 test dataset"
article = "Created by Glauco Masi"

example_list = [["examples/"+example] for example in os.listdir("examples")]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

demo.launch()
