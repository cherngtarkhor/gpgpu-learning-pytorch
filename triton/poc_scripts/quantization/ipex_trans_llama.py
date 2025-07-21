# https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm/int4_weight_only_quantization.html#weight-only-quantization-runtime
# https://github.com/intel/neural-compressor/blob/master/examples/3.x_api/pytorch/nlp/huggingface_models/language-modeling/quantization/transformers/weight_only/text-generation/run_generation_gpu_woq.py

import intel_extension_for_pytorch as ipex
# from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from neural_compressor.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
# torch._dynamo.config.suppress_errors = True
from neural_compressor.transformers import (
    RtnConfig,
    AwqConfig,
    TeqConfig,
    GPTQConfig,
    AutoRoundConfig,
)
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--load", action="store_true")
parser.add_argument("--inference", action="store_true")
parser.add_argument("--optimize", action="store_true")
args = parser.parse_args()


device_map = "xpu"
# model_name ="meta-llama/Meta-Llama-3-8B"
# model_name ="meta-llama/Llama-2-7b-hf"
model_name ="meta-llama/Llama-3.2-1B-Instruct"

current_script_dir = os.path.dirname(os.path.abspath(__file__))
saved_dir = os.path.abspath(os.path.join(current_script_dir, 'qmodel_llama3'))

load_start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# prompt = "Once upon a time, there existed a little girl,"
prompt = "Describe the tallest tower in world."
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device_map)

woq_quantization_config = RtnConfig(
    # bits=4,
    # sym=True,
    group_size=128,
    compute_dtype="fp16",
    scale_dtype="fp16",
    weight_dtype="int4_fullrange",
    # use_layer_wise=args.use_layer_wise,
    # quant_lm_head=args.quant_lm_head,
)

if args.quantize:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.float16, quantization_config=woq_quantization_config, trust_remote_code=True)
    # woq_quantization_config = WeightOnlyQuantConfig(compute_dtype="fp16", weight_dtype="int4_fullrange", scale_dtype="fp16", group_size=64)
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device_map, load_in_4bit=True)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True)

if args.save:
    model.save_pretrained(saved_dir)

if args.load:
    model = AutoModelForCausalLM.from_pretrained(saved_dir, device_map=device_map, torch_dtype=torch.float16, quantization_config=woq_quantization_config, trust_remote_code=True)

# float32 after optimize
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, Data type: {param.dtype}")

if args.optimize:
    print("Optimize with IPEX...")
    if args.quantize or args.load:
        # model = model = ipex.optimize_transformers(model.eval(), inplace=True, dtype=torch.float16, quantization_config=quantization_config, device=device_map) # OK
        model = ipex.llm.optimize(model, inplace=True, dtype=torch.float16, quantization_config=woq_quantization_config, device=device_map) # OK
        # model = ipex.optimize(model, inplace=True, dtype=torch.float16) # Error    
    else:
        model = ipex.llm.optimize(model, inplace=True, dtype=torch.float16, device=device_map)

# float16 with quantize, mix float16/32 with load
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, Data type: {param.dtype}")

load_end_time = time.time()

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