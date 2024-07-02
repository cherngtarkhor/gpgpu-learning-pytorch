#https://blog.roboflow.com/how-to-use-resnet-50/
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import os
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "forklift.jpg")
image = Image.open(image_path)
saved_path = os.path.join(script_dir, "saved.png")


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

#################### code changes ################
model = model.to("xpu")
inputs = inputs.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
