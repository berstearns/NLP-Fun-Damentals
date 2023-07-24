import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing_text import CONTEXT_SIZE, CONTEXT_TYPE, tokenizedSentence, dataset, vocab, word_to_idx, idx_to_word
from models import BatchNGramLanguageModel, NGramLanguageModel

def getBatch(dataset, BATCH_MAXSIZE=5):
    batch = []
    for row in dataset:  
        batch.append(row)
        if len(batch) == BATCH_MAXSIZE:
            yield batch  
            batch = []
    yield batch

def batch_train(dataset, model, word_to_idx, N_EPOCHS=100):
    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch_idx in range(N_EPOCHS):
        epoch_loss = 0
        for batch in getBatch(dataset,BATCH_MAXSIZE=5): 
            contexts, targets = zip(*batch)
            batch_target_tensor = torch.tensor([
                                    word_to_idx[nextWord]
                                    for nextWord in targets
                                    ], dtype=torch.long)
            model.zero_grad()
            ctxWordsIdxs_tensor = torch.tensor([
                                            [word_to_idx[w] for w in ctxWords]
                                            for ctxWords in contexts
                                            ])
            log_probs = model.forward(ctxWordsIdxs_tensor)
            loss = loss_function(log_probs, batch_target_tensor) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
    print(losses)
    return model

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
    return model

def predict_nextWord(model, contexts):
    ctxWordsIdxs_tensor = torch.tensor([
                            [word_to_idx[w] for w in ctxWords]
                            for ctxWords in contexts
                            ])
    with torch.no_grad():
        logprobs = model(ctxWordsIdxs_tensor)
        return torch.argmax(logprobs, dim=1).tolist()



if __name__ == "__main__":
    torch.manual_seed(1)
    print(tokenizedSentence)
    #### N-Gram Language Modelling example
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 10
    #ngram_embeddings = nn.Embedding(2, EMBEDDING_DIM)  # 2 words in vocab, 10 dimensional embeddings

    initial_model = BatchNGramLanguageModel(VOCAB_SIZE,
                                            EMBEDDING_DIM,
                                            CONTEXT_SIZE) 
    trained_model = batch_train(dataset, initial_model, word_to_idx, N_EPOCHS=10000) 

    count = 0
    acc = 0
    for batch in getBatch(dataset,BATCH_MAXSIZE=5): 
        contexts, targets = zip(*batch)
        for nextWord, pred in  zip(targets, predict_nextWord(trained_model, contexts)):
            count+=1
            print(f"{count} PREDICTION: {idx_to_word[pred]} EXPECTED: {nextWord} ")
            if idx_to_word[pred] == nextWord:
                acc += 1
    print(acc/count)
