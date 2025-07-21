import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Define the 4-bit quantization function
def quantize_4bit(tensor):
    """
    Quantize a tensor to 4-bit precision.
    Args:
        tensor (torch.Tensor): The tensor to quantize.
    Returns:
        torch.Tensor: Quantized tensor.
    """
    scale = (tensor.max() - tensor.min()) / 15  # 4-bit range
    zero_point = tensor.min()
    quantized = torch.round((tensor - zero_point) / scale).clamp(0, 15)
    dequantized = quantized * scale + zero_point
    return dequantized

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Example; replace with your preferred model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Quantize the model weights
print("Quantizing model weights to 4-bit precision...")
for name, param in model.named_parameters():
    if param.requires_grad:  # Only quantize trainable parameters
        param.data = quantize_4bit(param.data)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define input text and tokenize
input_text = "Once upon a time, in a faraway land,"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Run inference
print("Running inference...")
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=True,
        top_k=10,
        top_p=0.95,
        temperature=0.7,
    )

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:")
print(generated_text)
