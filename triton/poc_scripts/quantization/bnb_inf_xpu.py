# https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/blob/master/examples/gpu/llm/bitsandbytes/bnb_inf_xpu.py

# python bnb_inf_xpu.py --model_name meta-llama/Llama-3.2-1B-Instruct --quant_type int8 --max_new_tokens 64 --device xpu

import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf", required=False, type=str, help="model_name")
parser.add_argument("--quant_type", default="int8", type=str, help="quant type", choices=["int8", "nf4", "fp4"])
parser.add_argument("--max_new_tokens", default=64, type=int, help="min_gen_len")
parser.add_argument("--device", default="cpu", type=str, help="device type", choices=["cpu", "xpu"])
args = parser.parse_args()

def get_current_device():
    return Accelerator().process_index

device_map={'':get_current_device()} if args.device == 'xpu' else None

MAX_NEW_TOKENS = args.max_new_tokens
model_id = args.model_name
# torch_dtype = torch.bfloat16
torch_dtype = torch.float16

# text = 'I am happy because'
text = "Describe the tallest tower in world."

load_start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)

print('Loading model {}...'.format(model_id))
if args.quant_type == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_quant_type=args.quant_type,
                                            bnb_4bit_use_double_quant=False,
                                            bnb_4bit_compute_dtype=torch_dtype)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch_dtype, device_map=args.device)

load_end_time = time.time()

with torch.no_grad():
    # warmup
    model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    # model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    print("warm-up complite")
    generation_start_time = time.time()
    generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    generation_end_time = time.time()
    latency = generation_end_time - generation_start_time
    print(input_ids.shape)
    print(generated_ids.shape)
    result = "| latency: " + str(round(latency * 1000, 3)) + " ms |"
    print('+' + '-' * (len(result) - 2) + '+')
    print(result)
    print('+' + '-' * (len(result) - 2) + '+')

    # benchmark metric
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = generated_ids.size(1)
    num_generated_tokens = num_output_tokens - num_input_tokens
    loading_time = load_end_time - load_start_time
    throughput = num_generated_tokens / latency
    max_memory = torch.xpu.max_memory_allocated()

    print(f"### Input Tokens Count: {num_input_tokens}")
    print(f"### Output Tokens Count: {num_output_tokens}")
    print(f"### Generated Tokens Count: {num_generated_tokens}")
    print(f"### Loading Time: {loading_time:.6f} secs")
    print(f"### Inference Time: {latency:.6f} secs")
    print(f"### Throughput: {throughput:.6f} tokens/sec")
    print(f"### Max memory allocated: {max_memory / (1024 ** 3):02} GB")

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"output: {output}")
