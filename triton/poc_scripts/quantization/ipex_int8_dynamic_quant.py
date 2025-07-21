import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pretrained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
text = "This is a test sentence."
inputs = tokenizer(text, return_tensors="pt")

# Enable IPEX optimization
model = model.to("cpu")  # Ensure the model is on CPU or Intel GPU
model.eval()
model = ipex.optimize(model)

# Apply INT8 quantization
qconfig = ipex.quantization.default_dynamic_qconfig
prepared_model = ipex.quantization.prepare(model, qconfig=qconfig)
quantized_model = ipex.quantization.convert(prepared_model)

# Run inference
with torch.no_grad():
    outputs = quantized_model(**inputs)

print("Logits:", outputs.logits)
