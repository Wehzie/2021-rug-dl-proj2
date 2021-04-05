from imports import *
from batches import indexesFromSentence

MAX_LENGTH = 15


def normalizeString(string):
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string

def evaluate(encoder, decoder, searcher, voc, sentence, device, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # lengths = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word(token.item()) for token in tokens]
    # print(decoded_words)
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, device):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, device)
            # Format and print response sentence
            response = []
            for word in output_words:
                if word != 'EOS':
                    response.append(word)
                else:
                    break
            print('Bot:', ' '.join(response))

        except KeyError:
            print("Error: Encountered unknown word.")