#https://blog.roboflow.com/how-to-use-resnet-50/
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import os
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

############# Profiler ###############
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet50_2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
############# Profiler ###############

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


############# Profiler ###############
prof.start()

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        for _ in range(5):
            prof.step()
            logits = model(**inputs).logits

prof.stop()
############# Profiler ###############

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])