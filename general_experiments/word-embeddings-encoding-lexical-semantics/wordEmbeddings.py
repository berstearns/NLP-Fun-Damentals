import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing_text import CONTEXT_SIZE, CONTEXT_TYPE, tokenizedSentence, dataset, vocab, word_to_idx
from models import NGramLanguageModel

torch.manual_seed(1)


'''
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
'''





print(tokenizedSentence)
#### N-Gram Language Modelling example
vocab_size = len(vocab)
EMBEDDING_DIM = 10
#ngram_embeddings = nn.Embedding(2, EMBEDDING_DIM)  # 2 words in vocab, 10 dimensional embeddings


model = NGramLanguageModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE) 


losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
#model.forward(
N_EPOCHS = 10
for epoch_idx in range(N_EPOCHS):
    epoch_loss = 0
    for ctxWords, nextWord in dataset: 
        target_tensor = torch.tensor([word_to_idx[nextWord]], dtype=torch.long)
        model.zero_grad()
        ctxWordsIdxs_tensor = torch.tensor([word_to_idx[w] for w in ctxWords])
        log_probs = model.forward(ctxWordsIdxs_tensor)
        loss = loss_function(log_probs, target_tensor) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss)

print(losses)

#########3 N-Gram bidirectional language modelling example

def predict_nextWord(model, ctxWords):
    with torch.no_grad():
        logprobs = model(ctxWords)
        print(logprobs)

for idx, (ctxWords, nextWord) in enumerate(dataset):
    print(idx)
