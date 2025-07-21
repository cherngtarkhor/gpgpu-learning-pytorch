# https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.30%2Bxpu/docs/tutorials/features/int8_overview_xpu.md
# status: error
import torch
# import intel_extension_for_pytorch

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

# Define model
# model = M().to("xpu")
model = M()
model.eval()
modelImpe = torch.quantization.QuantWrapper(model)

# Define QConfig
qconfig = torch.quantization.QConfig(activation=torch.quantization.observer.MinMaxObserver .with_args(qscheme=torch.per_tensor_symmetric),
    weight=torch.quantization.default_weight_observer)  # weight could also be perchannel

modelImpe.qconfig = qconfig

# Prepare model for inserting observer
torch.quantization.prepare(modelImpe, inplace=True)

# Calibration to obtain statistics for Observer
# for data in calib_dataset:
#     modelImpe(data)

# Convert model to create a quantized module
torch.quantization.convert(modelImpe, inplace=True)

# Inference
inference_data = torch.randn(4, 4, 4, 4)
modelImpe(inference_data)
