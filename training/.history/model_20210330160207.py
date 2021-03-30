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

test = "/home/gery/Documents/Deep learning/I17-1099.Datasets/EMNLP_dataset/d_t.txt"
real = "/home/gery/Documents/Deep learning/I17-1099.Datasets/EMNLP_dataset/dialogues_text.txt"

DATA_PATH = real

qa_pairs = []
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))
data = 'pairs.txt'

save_dir = C:\Users\deea_\Desktop\depp\deeplearning_project1\Project'

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
voc, pairs = prepareData()
print('total dialogues '+ str(len(pairs)))
print('total words '+ str(voc.__len__()))



#################### PREPARE DATA ############################################
# %%
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

################### LOAD MODEL ###############################################
# %%
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.__len__(), hidden_size)
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
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, loadFilename)