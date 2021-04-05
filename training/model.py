# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from imports import *
from batches import batch2TrainData
from train import train
from trainIters import trainIters
from attention_layer import Attn
from decoder import LuongAttnDecoderRNN
from encoder import EncoderRNN
import os
from greedy_decoder import GreedySearchDecoder
from evaluate import evaluateInput
from full_training import full_training

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

##############################################################################
#%%
## Lowercase, trim, and remove non-letter characters
def normalizeString(string):
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string

######################### READ DATA ##########################################
# %%

test = "EMNLP_dataset/d_t.txt"
real = "EMNLP_dataset/dialogues_text.txt"

DATA_PATH = real

qa_pairs = []
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))
data = 'pairs.txt'

save_dir = os.path.join(os.getcwd(),'results')

#%%
def getData():
    with open(DATA_PATH, "r") as f:
        ## Load the data
        for line in f:
            sentences = line.split("__eou__")
            sentences.pop()                                     # remove the end-line
            ## Separate conversation into pairs of sentences
            for idx in range(len(sentences)-1):
                inputLine = normalizeString(sentences[idx].strip())              # .strip() removes start/end whitespace
                targetLine = normalizeString(sentences[idx+1].strip())

                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])

    ## Write pairs into csv file
    with open(data, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter = delimiter, lineterminator='\n')
        for pair in qa_pairs:
            writer.writerow(pair)

######################### TRIM DATA ##########################################
# %%
MAX_LENGTH = 15

def filterPairs(pairs):
    new_pairs = []
    for pair in pairs:
        if len(pair[0].split(" ")) < MAX_LENGTH and len(pair[0].split(" ")) < MAX_LENGTH:
            new_pairs.append(pair)
    return new_pairs

def trimPairsByWords(voc, pair):
    for sentence in pair:
        for word in sentence.split(" "):
            try:
                voc.word2index(word)
            except:
                return False
    return True

def prepareData():
    lines = open(data, "r").read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    pairs = filterPairs(pairs)
    voc = Vocab()
    for _ in range(4):
        voc.word2index("PAD", train=True)
        voc.word2index("SOS", train=True)
        voc.word2index("EOS", train=True)
    for pair in pairs:
        for sentence in pair:
            for word in sentence.split(" "):
                voc.word2index(word, train=True)
    voc = voc.prune_by_count(3)
    new_pairs = []
    for pair in pairs:
        if trimPairsByWords(voc,pair):
            new_pairs.append(pair)
    pairs = new_pairs
    return voc, pairs

# %%
getData()
voc, pairs = prepareData()
print('total dialogues '+ str(len(pairs)))
print('total words '+ str(voc.__len__()))



#################### PREPARE DATA ############################################
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches


# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)

################### LOAD MODEL ###############################################
# %%
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter))

encoder_optimizer_sd = []
decoder_optimizer_sd = []

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.__len__(), hidden_size)    # Word embedding 
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.__len__(), decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

################### TRAINING ###############################################
# %%
# full_training(model_name, voc, pairs, encoder, decoder, 
#               embedding, encoder_n_layers, decoder_n_layers, 
#               save_dir, batch_size, 
#               loadFilename, encoder_optimizer_sd, decoder_optimizer_sd)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, device)

evaluateInput(encoder, decoder, searcher, voc, device)