import torch
import random
import gradio as gr

from PIL import Image
from typing import Tuple, Dict
from timeit import default_timer as timer

from modules import model_builder, utils, data_setup

data_20_percent_path = utils.download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
    destination="pizza_steak_sushi_20_percent"
)

test_dir = data_20_percent_path/"test"
train_dir = data_20_percent_path/"train"

effnetb2, effnetb2_transforms = model_builder.create_effnetb2(out_features=3, device=torch.device("cpu"))
effnetb2.load_state_dict(torch.load(f="/home/glauco/Desktop/projects/learningPyTorch/trained_models/09_pretrained_effnetb2_20_percent.pth"))
train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=effnetb2_transforms,
    batch_size=32
)

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

test_data_paths = list(test_dir.glob("*/*.jpg"))

# random_image_path = random.sample(test_data_paths, k=1)[0]
# image = Image.open(random_image_path)
# pred_dict, pred_time = predict(image)
# print(f"Prediction path: {random_image_path}")
# print(f"Prediction dictionary: {pred_dict}")
# print(f"Prediction time: {pred_time}")

# Gradio's interface takes in a list of examples as an optional parameter
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]
# print(example_list)

title = "🍕🥩🍣 Vision Model"
description = "An EfficientNetB2 feature extractor computer vision model able to differentiate photos of pizza, steak and sushi"
article = "Created by Glauco Masi"

demo = gr.Interface(
    fn=predict,                                                             # Prediction function
    inputs=gr.Image(type="pil"),                                            # Function's input
    outputs=[                                                               # Function's outputs
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

demo.launch(
    debug=False,
    share=True
)