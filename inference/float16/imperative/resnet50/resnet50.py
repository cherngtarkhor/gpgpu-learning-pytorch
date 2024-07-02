import torch
import torchvision.models as models

############# code changes ###############
import intel_extension_for_pytorch as ipex

############# code changes ###############

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()

#save into jit
#jit_model = torch.jit.trace((3,224,224),model)
#jit_model = torch.jit.freeze(jit_model)
#jit_model = torch.jit.save(jit_model, 'jit_model.pt')

data = torch.rand(1, 3, 224, 224)
#change to an image, use opencv or pil
print("in:", data)

#################### code changes ################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)


#################### code changes ################

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        output = model(data)
        print("out:", output)

print("Execution finished")
