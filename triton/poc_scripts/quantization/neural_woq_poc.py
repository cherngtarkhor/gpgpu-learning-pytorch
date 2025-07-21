# https://github.com/intel/neural-compressor/blob/master/examples/helloworld/torch_woq/quant_mistral.py
# https://github.com/intel/neural-compressor/blob/master/examples/notebook/pytorch/Quick_Started_Notebook_of_INC_for_Pytorch.ipynb

# from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
import torch
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--load", action="store_true")
parser.add_argument("--inference", action="store_true")
args = parser.parse_args()

def get_prompt(user_input: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    prompt_texts = [f'<|begin_of_text|>']

    if system_prompt != '':
        prompt_texts.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>')

    for history_input, history_response in chat_history:
        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>')
        prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>')

    prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    return ''.join(prompt_texts)


device_map = "xpu"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# prompt = "Once upon a time, there existed a little girl,"
# inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device_map)
saved_dir = "/home/ctkhor/unsloth/unsloth/saved_model"

# messages = "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"
messages = "Describe the tallest tower in the world."
# messages = "What is Unsloth?"
# messages = "What is AI?"    

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

DEFAULT_SYSTEM_PROMPT = """\
"""
prompt = get_prompt(messages, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
inputs = tokenizer.encode(prompt, return_tensors="pt").to('xpu')


if args.quantize:
    # https://intel.github.io/neural-compressor/latest/docs/source/quantization_weight_only.html#examples
    # float_model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1")
    float_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype = torch.float16, device_map=device_map)
    # float_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype = torch.float16)

    # https://github.com/intel/neural-compressor/blob/91184490c3b2c3e777f54fdffc27344170f6eb2d/neural_compressor/config.py#L1187
    woq_conf = PostTrainingQuantConfig(approach="weight_only",
        op_type_dict={
            ".*": {  # re.match
                "weight": {
                    "bits": 4,  # 1-8 bit
                    "group_size": -1,  # -1 (per-channel)
                    "scheme": "sym",
                    "algorithm": "RTN",
                    # "dtype": "int4"#"fp4"
                },
            },
        },
        op_name_dict = {
            'lm_head': {"weight": {'dtype': 'fp32'}, },
            'embed_out': {"weight": {'dtype': 'fp32'}, },  # for dolly_v2
        },
        recipes={
            'rtn_args':{'enable_full_range': False, 'enable_mse_search': False},
            # 'gptq_args':{'percdamp': 0.01, 'actorder':True, 'block_size': 128, 'nsamples': 128, 'use_full_length': False},
            # 'awq_args':{'enable_auto_scale': True, 'enable_mse_search': True, 'n_blocks': 5},
        }
    )

    # https://github.com/intel/neural-compressor/blob/91184490c3b2c3e777f54fdffc27344170f6eb2d/neural_compressor/quantization.py#L33
    quantized_model = fit(model=float_model, conf=woq_conf)

    # https://github.com/intel/neural-compressor/blob/91184490c3b2c3e777f54fdffc27344170f6eb2d/neural_compressor/quantization.py#L33
    model = quantized_model.export_compressed_model(
                                                # compression_dtype=torch.int8,
                                                # compression_dim=0,
                                                # use_optimum_format=False,
                                                # scale_dtype=torch.float16
                                                )

    if args.save:
        float_model.save_pretrained(saved_dir)

if args.load:
    model = AutoModelForCausalLM.from_pretrained(saved_dir, trust_remote_code=True, torch_dtype = torch.float16)


if args.inference:
    model = model.to(device_map)

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data type: {param.dtype}")

    start_time = time.time()
    output = model.generate(input_ids = inputs, eos_token_id=terminators, max_new_tokens=128, use_cache = True)

    torch.xpu.synchronize()
    end_time = time.time()

    output = output.cpu()

    output_str = tokenizer.decode(output[0], skip_special_tokens=False)
    print(output_str)

    num_input_tokens = inputs.size(1)
    num_output_tokens = output.size(1)
    num_generated_tokens = num_output_tokens - num_input_tokens
    generation_time = end_time - start_time
    throughput = num_generated_tokens / generation_time
    print(f"num_input_tokens : {num_input_tokens}")
    print(f"num_output_tokens : {num_output_tokens}")
    print(f"Generated Tokens: {num_generated_tokens}")
    print(f"Inference time: {generation_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")