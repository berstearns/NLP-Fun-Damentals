
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()


tokenizedSentence = test_sentence 

def dataset_from_tokenizedSentence(tokenizedSentence,
                                   CONTEXT_TYPE="unidirectional",
                                   CONTEXT_SIZE=2):
    dataset  =  []
    endpoint = len(test_sentence) if CONTEXT_TYPE == "unidirectional" else len(test_sentence) - CONTEXT_SIZE 
    for i in range(CONTEXT_SIZE, endpoint): # zip(*[arr[i:] for i 
        leftCtx_startIdx = i - CONTEXT_SIZE    
        leftCtx_endIdx = i 
        leftCtxTokens = tokenizedSentence[leftCtx_startIdx : leftCtx_endIdx] 
        if CONTEXT_TYPE == "bidirectional":
            rightCtx_startIdx = i + 1   
            rightCtx_endIdx = i + CONTEXT_SIZE + 1  
            rightCtxTokens = tokenizedSentence[rightCtx_startIdx : rightCtx_endIdx] 
            ctxTokens = leftCtxTokens + rightCtxTokens 
        else:
            ctxTokens = leftCtxTokens
        dataset.append( ( ctxTokens,
                          tokenizedSentence[i] ))
    return dataset
        
def create_vocab(tokenizedSentence):
    vocab = set(tokenizedSentence)
    word_to_idx = { word : idx for idx, word in enumerate(vocab) }
    idx_to_word = { idx : word for idx, word in enumerate(vocab) }
    return vocab, word_to_idx, idx_to_word 


CONTEXT_SIZE = 4 
CONTEXT_TYPE = "unidirectional" 
vocab, word_to_idx, idx_to_word = create_vocab(tokenizedSentence)
dataset = dataset_from_tokenizedSentence(tokenizedSentence, CONTEXT_TYPE, CONTEXT_SIZE)
