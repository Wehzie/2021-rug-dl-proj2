import torch
import random
import numpy as np

from batches import indexes_from_sentence
from read_data import normalize_string

def evaluate(encoder, decoder, searcher, voc, sentence, device, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # lengths = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
    # Decode sentence with searcher
    tokens, _ = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word(token.item()) for token in tokens]
    # print(decoded_words)
    return decoded_words


def evaluate_input(encoder, decoder, searcher, voc, device, max_length):
    input_sentence = ""
    while True:
        try:
            # Get input sentence
            input_sentence = input("> ")
            # Check if it is quit case
            if input_sentence == "q" or input_sentence == "quit":
                break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, input_sentence, device, max_length
            )
            # Format and print response sentence
            response = []
            for word in output_words:
                if word != "EOS":
                    response.append(word)
                else:
                    break
            print("Bot:", " ".join(response))

        except KeyError:
            print("Error: Encountered unknown word.")


def sentence_maker(encoder, decoder, searcher, voc, input_sentence, device, max_length):
    output_words = evaluate(
        encoder, decoder, searcher, voc, input_sentence, device, max_length
    )
    response = []
    for word in output_words:
        if word != "EOS":
            response.append(word)
        else:
            break
    return response


def list_to_string(s):
    string = ""
    for word in s:
        string += word
        string += " "
    return string[:-1]


def create_conversations(
    encoder,
    decoder,
    searcher,
    voc,
    device,
    max_length,
    save_file,
    sentences_lengths,
    test_lines,
):
    min_sentence_length = min(sentences_lengths)
    max_sentence_length = max(sentences_lengths)

    sentences_distribution, _ = np.histogram(
        sentences_lengths, bins=(max_sentence_length - 1)
    )
    chatbot_conv = len(test_lines)  # number of conversations the chatbot will generate
    chatbot_conv_lengths = random.choices(
        np.arange(min_sentence_length, max_sentence_length + 1),
        weights=sentences_distribution,
        k=chatbot_conv,
    )
    step = 1
    with open(save_file, "a") as file:
        for conv_length in chatbot_conv_lengths:
            print("Creating conversation:"+str(step))
            step += 1
            conversation_starter = test_lines.pop(0)
            file.write(conversation_starter + " __eou__ ")
            response = sentence_maker(
                encoder,
                decoder,
                searcher,
                voc,
                conversation_starter,
                device,
                max_length,
            )
            for _ in range(conv_length):
                response = list_to_string(response)
                file.write("" + response + " __eou__ ")
                input_sentence = response
                response = sentence_maker(
                    encoder, decoder, searcher, voc, input_sentence, device, max_length
                )
            file.write("\n")
