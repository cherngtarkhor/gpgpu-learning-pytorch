# https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm/int4_weight_only_quantization.html#weight-only-quantization-runtime
import intel_extension_for_pytorch as ipex
from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

device = "xpu"
model_name = "Qwen/Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Once upon a time, there existed a little girl,"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# woq_quantization_config = WeightOnlyQuantConfig(compute_dtype="fp16", weight_dtype="int4_fullrange", scale_dtype="fp16", group_size=64)
# qmodel = AutoModelForCausalLM.from_pretrained(model_name, device_map="xpu", quantization_config=woq_quantization_config, trust_remote_code=True)

# qmodel = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="xpu", trust_remote_code=True)

saved_dir = "/home/ctkhor/unsloth/unsloth/qmodel_qwen"
# qmodel.save_pretrained(saved_dir)
qmodel = AutoModelForCausalLM.from_pretrained(saved_dir, trust_remote_code=True, device_map=device)

# optimize the model with ipex, it will improve performance.
qmodel = ipex.optimize_transformers(qmodel, inplace=True, dtype=torch.float16, quantization_config="woq", device=device)

output = qmodel.generate(inputs)
