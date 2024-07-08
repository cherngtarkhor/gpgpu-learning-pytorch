#Similarity: input a sentence and output the most simlar sentence
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

############# Profiler ###############
prof1 = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/bert_1_1'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

prof2 = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/bert_1_2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
############# Profiler ###############


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

model.eval()

# Input sentence
input_sentence = "Hello, how are you?"
#input_sentence = "Thank you, good bye!"
#input_sentence = "Let's meet again some other time!"
#input_sentence = "Good morning"
#input_sentence = "It's raining now!"

# Tokenize the input sentence
inputs = tokenizer(input_sentence, return_tensors='pt')

#################### code changes ################
inputs = inputs.to("xpu")
model = model.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

############# Profiler ###############
# Get the outputs from the model
prof1.start()
with torch.no_grad():
    ########################### code changes ########################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ########################### code changes ########################
        for _ in range(5):
            prof1.step()
            outputs = model(**inputs)
prof1.stop()
############# Profiler ###############

# Get the embeddings for the input sentence (use the [CLS] token)
input_embeddings = outputs.last_hidden_state[:, 0, :]

# Define some predefined sentences to compare with
predefined_sentences = [
    "Hi, what's up?",
    "Goodbye, see you later.",
    "Hello, how are you doing?",
    "The weather is nice today.",
    "Have a nice day!"
]

############# Profiler ###############
prof2.start()
# Tokenize and get embeddings for the predefined sentences
predefined_embeddings = []
for sentence in predefined_sentences:
    inputs = tokenizer(sentence, return_tensors='pt')
    #################### code changes ################
    inputs = inputs.to("xpu")
    #################### code changes ################
    with torch.no_grad():
        ########################### code changes ########################
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
        ########################### code changes ########################
            prof2.step()
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        predefined_embeddings.append(embeddings)
prof2.stop()
############# Profiler ###############

# Compute cosine similarities
similarities = []
for embeddings in predefined_embeddings:
    similarity = F.cosine_similarity(input_embeddings, embeddings)
    similarities.append(similarity.item())

# Find the most similar sentence
most_similar_index = similarities.index(max(similarities))
most_similar_sentence = predefined_sentences[most_similar_index]

# Output the most similar sentence
print(f"Input sentence: {input_sentence}")
print(f"Most similar sentence: {most_similar_sentence}")
