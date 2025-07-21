import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoConfig, AutoTokenizer
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    AutoRoundConfig,
)

device = "xpu"
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Once upon a time, a little girl"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

quantization_config = AutoRoundConfig(
    tokenizer=tokenizer,
    bits=4,
    group_size=32,
    max_input_length=2048,
    compute_dtype="fp16",
    scale_dtype="fp16",
    weight_dtype="int4",  # int4 == int4_clip
    calib_iters=2,
    calib_len=32,
    nsamples=2,
    lr=0.0025,
    minmax_lr=0.0025,
)
qmodel = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    quantization_config=quantization_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# optimize the model with ipex, it will improve performance.
qmodel = ipex.optimize_transformers(
    qmodel, inplace=True, dtype=torch.float16, quantization_config=True, device=device
)
output = qmodel.generate(inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
