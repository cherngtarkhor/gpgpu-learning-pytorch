import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import os
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

############# Profiler ###############
prof = torch.profiler.profile(
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet50_1'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
############# Profiler ###############

# Load a pre-trained ResNet50 model
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()  # Set the model to evaluation mode

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load an image from a URL or a local file
#img_url = "https://example.com/path/to/your/image.jpg"  # Replace with your image URL
#response = requests.get(img_url)
#img = Image.open(BytesIO(response.content))

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "husky.jpg")
#saved_path = os.path.join(script_dir, "saved.png")
img = Image.open(image_path)

# Preprocess the image
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

#################### code changes ################
model = model.to("xpu")
batch_t = batch_t.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

############# Profiler ###############
prof.start()

# Perform the forward pass and get the predictions
with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        for _ in range(5):
            prof.step()
            output = model(batch_t)

prof.stop()
############# Profiler ###############

# Load ImageNet class names
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Get the top 1 predicted class
_, idx = torch.max(output, 1)
predicted_label = labels[idx.item()]
print(f"Predicted label: {predicted_label}")


# Get the top 5 predicted classes
_, indices = torch.topk(output, 5)
percentages = torch.nn.functional.softmax(output, dim=1)[0] * 100

# Print the top 5 predicted labels with their confidence scores
for idx in indices[0]:
    label = labels[idx.item()]
    confidence = percentages[idx].item()
    print(f"{label}: {confidence:.2f}%")

# Save the processed image (optional)
processed_img = transforms.ToPILImage()(img_t)
processed_img.save("processed_image.jpg")
