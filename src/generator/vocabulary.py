from vocab import Vocab

######################### TRIM DATA ##########################################

# Remove sentences with more words than MAX_LENGTH words
def filter_pairs(pairs, max_length):
    new_pairs = []
    for pair in pairs:
        if len(pair[0].split(" ")) < max_length and len(pair[0].split(" ")) < max_length:
            new_pairs.append(pair)
    return new_pairs

# Remove sentences of which words are not in the vocabulary
def trim_pairs_by_words(voc, pair):
    for sentence in pair:
        for word in sentence.split(" "):
            try:
                voc.word2index(word)
            except:
                return False
    return True

def trim_sentence_by_words(voc, sentence):
    for word in sentence.split(" "):
        try:
            voc.word2index(word)
        except:
            return False
    return True

def prepare_data(pairs_path, max_length):
    lines = open(pairs_path, "r").read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]  # retrieve pairs
    pairs = filter_pairs(pairs, max_length)             # Remove sentences with more words than MAX_LENGTH
    
    voc = Vocab()
    min_occurrences = 0                     # Set the minimum occurrences of a word to be in the vocabulary

    for _ in range(min_occurrences + 1):    # Add PAD, SOS, EOS tags multiple times so the are not
        voc.word2index("PAD", train=True)  # removed when trimming the vocabulary
        voc.word2index("SOS", train=True)
        voc.word2index("EOS", train=True)
    
    for pair in pairs:                     # Add all words in the vocabulary
        for sentence in pair:
            for word in sentence.split(" "):
                voc.word2index(word, train=True)
    voc = voc.prune_by_count(min_occurrences)    # Remove words with few occurrences

    # Only keep the sentences of which words are in the vocabulary
    new_pairs = []
    for pair in pairs:
        if trim_pairs_by_words(voc,pair):
            new_pairs.append(pair)

    # Return the vocabulary and the new pairs of sentences
    return voc, new_pairs            

def trim_lines(lines_path, voc):
    lines = open(lines_path, "r").read().strip().split('\n')

    new_lines = []
    for line in lines:
        if trim_sentence_by_words(voc, line):
            new_lines.append(line)

    return new_lines