# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from imports import *
from batches import batch2TrainData

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
# %%
