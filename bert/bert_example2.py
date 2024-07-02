#https://www.kaggle.com/code/eriknovak/pytorch-bert-sentence-similarity
##Similarity: Analyse similarity of sentences and visualize in plot
import torch
import transformers
from transformers import BertModel
from transformers import BertTokenizer
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np

############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############


def visualize(distances, figsize=(10, 5), titles=None):
    # get the number of columns
    ncols = len(distances)
    # create the subplot placeholders
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize)
    
    for i in range(ncols):
        
        # get the axis in which we will draw the matrix
        axes = ax[i] if ncols > 1 else ax
        
        # get the i-th distance
        distance = distances[i]
        
        # create the heatmap
        axes.imshow(distance)
        
        # show the ticks
        axes.set_xticks(np.arange(distance.shape[0]))
        axes.set_yticks(np.arange(distance.shape[1]))
        
        # set the tick labels
        axes.set_xticklabels(np.arange(distance.shape[0]))
        axes.set_yticklabels(np.arange(distance.shape[1]))
        
        # set the values in the heatmap
        for j in range(distance.shape[0]):
            for k in range(distance.shape[1]):
                text = axes.text(k, j, str(round(distance[j, k], 3)),
                               ha="center", va="center", color="w")
        
        # set the title of the subplot
        title = titles[i] if titles and len(titles) > i else "Text Distance"
        axes.set_title(title, fontsize="x-large")
        
    fig.tight_layout()
    plt.show()



model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

texts  = [
    'Obama speaks to the media in Illinois',
    'The president greets the press in Chicago',
    'Oranges are my favorite fruit',
]

encodings = tokenizer(
    texts, # the texts to be tokenized
    padding=True, # pad the texts to the maximum length (so that all outputs have the same length)
    return_tensors='pt' # return the tensors (not lists)
)

# vocab_size = model.config.vocab_size
# batch_size = 1
# seq_length = 512
# data = torch.randint(vocab_size, size=[batch_size, seq_length])
# #print in string, instead of int
# #print("In: ", data)

#################### code changes ################
encodings = encodings.to("xpu")
model = model.to("xpu")
# data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

print("KEY", encodings.keys())
print("input_ids", encodings['input_ids'])
print("attention_mask", encodings['attention_mask'])
print("token_type_ids", encodings['token_type_ids'])
for tokens in encodings['input_ids']:
    print(tokenizer.convert_ids_to_tokens(tokens))

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
#        output = model(data)
        embeds = model(**encodings)
        #print("Out: ", output)
        embeds = embeds[0]
        #print("embeds: ", embeds)
        print("embeds.shape", embeds.shape)

        #Way 1: Use the [CLS] Embeddings¶
        CLSs = embeds[:, 0, :]
        # normalize the CLS token embeddings
        normalized = f.normalize(CLSs, p=2, dim=1)
        # calculate the cosine similarity
        cls_dist = normalized.matmul(normalized.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
#        cls_dist = cls_dist.numpy()

#        visualize([cls_dist], titles=["CLS"])
        
        #Way 2: Computing the Mean of All Output Vectors
        MEANS = embeds.mean(dim=1)
        # normalize the MEANS token embeddings
        normalized = f.normalize(MEANS, p=2, dim=1)
        # calculate the cosine similarity
        mean_dist = normalized.matmul(normalized.T)
        mean_dist = mean_dist.new_ones(mean_dist.shape) - mean_dist
#        mean_dist = mean_dist.numpy()
        
#        visualize([mean_dist], titles=["MEAN"])
        
        #Way 3: Compute the max-over-time of the Output Vectors
        MAXS, _ = embeds.max(dim=1)
        # normalize the MEANS token embeddings
        normalized = f.normalize(MAXS, p=2, dim=1)
        # calculate the cosine similarity
        max_dist = normalized.matmul(normalized.T)
        max_dist = max_dist.new_ones(max_dist.shape) - max_dist
#        max_dist = max_dist.numpy()
        
#        visualize([max_dist], titles=["MAX"])


print("Execution finished")

