import os
import torch
import gradio as gr

from typing import Tuple, Dict
from model import create_effnet2_model
from safetensors.torch import load_file
from timeit import default_timer as timer

class_names = ["pizza", "steak", "sushi"]

effnetb2, effnetb2_transforms = create_effnet2_model(num_classes=3)
sd = load_file("09_pretrained_effnetb2_20_percent.safetensors")
missing, unexpected = effnetb2.load_state_dict(sd, strict=True)



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



title = "🍕🥩🍣 Vision Model"
description = "An EfficientNetB2 feature extractor computer vision model able to differentiate photos of pizza, steak and sushi"
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
