from torch import nn
class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, ctxIdxs_tensor):
        embeds = self.embeddings(ctxIdxs_tensor).view((1,-1)) 
        p1 = self.relu(self.linear1(embeds))
        p2 = self.relu(self.linear2(p1))
        log_probs = self.logsoftmax(p2)
        return log_pros
