#https://pytorch.org/hub/huggingface_pytorch-transformers/
#Using modelForQuestionAnswering for question answering
import torch
import transformers
from transformers import BertModel

############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

############# Profiler ###############
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/bert_3'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
############# Profiler ###############

question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# The format is paragraph first and then question
text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

#################### code changes ################
segments_tensors = segments_tensors.to("xpu")
tokens_tensor = tokens_tensor.to("xpu")
question_answering_model = question_answering_model.to("xpu")
model = ipex.optimize(question_answering_model, dtype=torch.float16)
#################### code changes ################

############# Profiler ###############
prof.start()
# Predict the start and end positions logits
with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        for _ in range(5):
            prof.step()
            out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)
prof.stop()
############# Profiler ###############

# get the highest prediction
answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
assert answer == "puppeteer"
print("answer: ", answer)

# Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training)
start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
#################### code changes ################
start_positions = start_positions.to("xpu")
end_positions = end_positions.to("xpu")
#################### code changes ################
multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)

print("Execution finished")
