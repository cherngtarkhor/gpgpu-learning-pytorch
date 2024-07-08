#https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
#need pip install validators
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############


############# Profiler ###############
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet50_3'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
############# Profiler ###############


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#################### code changes ################
device = torch.device("xpu")
#################### code changes ################
print(f'Using {device} for inference')

#Load the model pretrained on ImageNet dataset.
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

resnet50.eval().to(device)

#Prepare sample input data.
uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)

#################### code changes ################
resnet50 = ipex.optimize(resnet50, dtype=torch.float16)
#################### code changes ################

############# Profiler ###############
prof.start()

#Run inference. Use pick_n_best(predictions=output, n=topN) helper function to pick N most probably hypothesis according to the model.
with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        for _ in range(5):
            prof.step()
            output = torch.nn.functional.softmax(resnet50(batch), dim=1)

prof.stop()
############# Profiler ###############

results = utils.pick_n_best(predictions=output, n=5)


script_dir = os.path.dirname(os.path.abspath(__file__))
#Display the result.
for index, (uri, result) in enumerate(zip(uris, results)):
#for uri, result in zip(uris, results):
    img = Image.open(requests.get(uri, stream=True).raw)
    img.thumbnail((256,256), Image.LANCZOS)
    plt.imshow(img)
    plt.show()
    image_name = "image_" + str(index) + ".jpg"
    saved_path = os.path.join(script_dir, image_name)
    img.save(saved_path)
    print(result)
