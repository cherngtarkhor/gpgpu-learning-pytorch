import intel_extension_for_pytorch as ipex
import torch
torch._dynamo.config.suppress_errors = True
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import time

# load_in_8bit=False,
# load_in_4bit=False,
# llm_int8_threshold=6.0,
# llm_int8_skip_modules=None,
# llm_int8_enable_fp32_cpu_offload=False,
# llm_int8_has_fp16_weight=False,
# bnb_4bit_compute_dtype=None,
# bnb_4bit_quant_type="fp4",
# bnb_4bit_use_double_quant=False,
# bnb_4bit_quant_storage=None,
# **kwargs,
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quant=False,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "facebook/opt-350m"

load_start_time = time.time()
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="xpu")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, quantization_config=bnb_config, device_map="xpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Describe the tallest tower in world."

# model = ipex.llm.optimize(model, inplace=True, quantization_config=bnb_config, device="xpu")

model = model.to("xpu")

load_end_time = time.time()

# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, Data type: {param.dtype}")


# inputs = tokenizer(text, return_tensors="pt").to("xpu")
# generation_start_time = time.time()
# outputs = model.generate(**inputs, max_new_tokens=64)
# generation_end_time = time.time()
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


inputs = tokenizer.encode(text, return_tensors="pt").to('xpu')
kwargs = { 'input_ids': inputs, 'max_new_tokens': 64, 'use_cache': True }

for i in range(2):
    generation_start_time = time.time()
    outputs = model.generate(**kwargs)
    torch.xpu.synchronize()
    generation_end_time = time.time()
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # benchmark metric
    num_input_tokens = inputs.shape[1]
    num_output_tokens = outputs.size(1)
    num_generated_tokens = num_output_tokens - num_input_tokens
    loading_time = load_end_time - load_start_time
    generation_time = generation_end_time - generation_start_time
    throughput = num_generated_tokens / generation_time
    max_memory = torch.xpu.max_memory_allocated()

    print(f"### Input Tokens Count: {num_input_tokens}")
    print(f"### Output Tokens Count: {num_output_tokens}")
    print(f"### Generated Tokens Count: {num_generated_tokens}")
    print(f"### Loading Time: {loading_time:.6f} secs")
    print(f"### Inference Time: {generation_time:.6f} secs")
    print(f"### Throughput: {throughput:.6f} tokens/sec")
    print(f"### Max memory allocated: {max_memory / (1024 ** 3):02} GB")
