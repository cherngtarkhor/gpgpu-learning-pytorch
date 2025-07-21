import torch
import torch.nn as nn

# Define a custom 4-bit quantization function
class Quantize4bit(nn.Module):
    def forward(self, x):
        scale = (x.max() - x.min()) / 15  # Scale for 4-bit (16 levels)
        zero_point = x.min()
        quantized = torch.round((x - zero_point) / scale).clamp(0, 15)
        dequantized = quantized * scale + zero_point
        return dequantized

# Apply to a model
model = nn.Linear(10, 10)
quant_layer = Quantize4bit()

for name, param in model.named_parameters():
    param.data = quant_layer(param.data)
