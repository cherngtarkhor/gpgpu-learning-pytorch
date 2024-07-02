import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")

#model.eval()
model.eval().to("xpu")

### torchscript save a jit model ###
# scripted_model = torch.jit.script(model)
# frozen_model = torch.jit.freeze(scripted_model) # optional
# #torch.jit.save(frozen_model, "resnet50_jit_model_cpu_sfs_1.pt")
# torch.jit.save(frozen_model, "resnet50_jit_model_xpu_sfs_1.pt")
### torchscript save a jit model ###

data = torch.rand(1, 3, 224, 224)
print("in:", data)

data = data.to("xpu")

#### torchscript save a jit model with input ###
jit_trace_model = torch.jit.trace(model, data)
jit_frozen_model = torch.jit.freeze(jit_trace_model)
#torch.jit.save(jit_frozen_model, 'resnet50_jit_model_cpu_tfs_1.pt')
torch.jit.save(jit_frozen_model, 'resnet50_jit_model_xpu_tfs_1.pt')
#### torchscript save a jit model with input ###

print("Execution finished")
