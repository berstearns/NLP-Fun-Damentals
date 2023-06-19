import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing_text import CONTEXT_SIZE, CONTEXT_TYPE, tokenizedSentence, dataset, vocab, word_to_idx, idx_to_word
from models import NGramLanguageModel

def train(dataset, model, word_to_idx, N_EPOCHS=100):
    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
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
    for idx, (ctxWords, nextWord) in enumerate(dataset):
        print(f"{idx} PREDICTION: {idx_to_word[predict_nextWord(model, ctxWords)]} EXPECTED: {nextWord} ")
        input()
    return model

def predict_nextWord(model, ctxWords):
    ctxWordsIdxs_tensor = torch.tensor([word_to_idx[w] for w in ctxWords])
    with torch.no_grad():
        logprobs = model(ctxWordsIdxs_tensor)
        return torch.argmax(logprobs, dim=-1).item()



if __name__ == "__main__":
    torch.manual_seed(1)
    print(tokenizedSentence)
    #### N-Gram Language Modelling example
    vocab_size = len(vocab)
    EMBEDDING_DIM = 10
    #ngram_embeddings = nn.Embedding(2, EMBEDDING_DIM)  # 2 words in vocab, 10 dimensional embeddings

    initial_model = NGramLanguageModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE) 
    trained_model = train(dataset, initial_model, word_to_idx, N_EPOCHS=100) 

    for idx, (ctxWords, nextWord) in enumerate(dataset):
        print(f"{idx} PREDICTION: {idx_to_word[predict_nextWord(trained_model, ctxWords)]} EXPECTED: {nextWord} ")
        input()
