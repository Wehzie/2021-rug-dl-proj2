# pylint: disable=locally-disabled, multiple-statements, line-too-long, invalid-name, wildcard-import, unused-wildcard-import

import os
import random
import pickle

import torch
import torch.nn as nn
import bcolz

from model_attention import Attention
from model_encoder import Encoder
from model_decoder import Decoder
from model_greedy_decoder import GreedySearchDecoder

from read_data import get_data, get_test_data
from train_generator import full_training
from glove_embeddings import get_embeddings
from batches import batch_to_train_data
from vocabulary import trim_lines
from evaluate import create_conversations

def run_generator():
    ##############################################################################

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    DATA_PATH = "data_dailydialog/dialogues_training.txt"
    DATA_TEST_PATH = "data_app/test/dialogues_test.txt"

    PAIRS_PATH = os.path.join("data_app/generator", "pairs_trimmed.txt")
    LINES_PATH = os.path.join("data_app/generator", "lines_trimmed.txt")
    save_dir = os.path.join("data_app/generator", "results")
    glove_path = os.path.join("data_app/generator", "glove")

    ################### Input ####################################################
    option = input(
        "Please select what you would like to do:\n 1. Train and chat  2. Load model and chat \n"
    )
    embedding = '2'
    if option == '1':
        embedding = input(
            "Please select what kind of embedding would you like to use:\n 1. GloVe  2. Embedding layer \n"
        )
    ################### READ, NORMALIZE, CREATE PAIRS ############################

    MAX_LENGTH = 15  # maximum words in a sentence
    sentences_lengths = get_data(DATA_PATH, PAIRS_PATH)  # read_data.py

    # if the trimmed lines are already saved skip getTestData
    if os.path.exists(LINES_PATH) == False or os.stat(LINES_PATH).st_size == 0:
        get_test_data(DATA_TEST_PATH, LINES_PATH, MAX_LENGTH)  # read_data.py

    #################### CREATE VOCABULARY AND NEW PAIRS #########################

    voc, pairs = prepare_data(PAIRS_PATH, MAX_LENGTH)  # vocabulary.py

    print("total dialogues " + str(len(pairs)))
    print("total words " + str(voc.__len__()))

    lines = trim_lines(LINES_PATH, voc)
    #################### PREPARE DATA IN BATCHES #################################

    small_batch_size = 5
    batches = batch_to_train_data(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    ################### LOAD MODEL ###############################################

    # Configure models
    model_name = "cb_model"
    attn_model = "dot"
    # attn_model = "general"
    # attn_model = 'concat'
    hidden_size = 100
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 50

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 8000

    if option == '2':
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
        encoder_sd = checkpoint["en"]
        decoder_sd = checkpoint["de"]
        encoder_optimizer_sd = checkpoint["en_opt"]
        decoder_optimizer_sd = checkpoint["de_opt"]
        embedding_sd = checkpoint["embedding"]
        voc.__dict__ = checkpoint["voc_dict"]


    print("Building encoder and decoder ...")
    ########################## Initialize word embeddings###########################
    if embedding == '1':
        get_embeddings()
        vectors = bcolz.open(f"{glove_path}/6B.100.dat")[:]
        words = pickle.load(open(f"{glove_path}/6B.100_words.pkl", "rb"))
        word2idx = pickle.load(open(f"{glove_path}/6B.100_idx.pkl", "rb"))

        glove = {w: vectors[word2idx[w]] for w in words}

        matrix_len = voc.__len__()
        weights_matrix = np.zeros((matrix_len, 100))
        words_found = 0
        for i, word in enumerate(voc.to_dict()["index2word"]):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(100,))

        num_embeddings, embedding_dim = weights_matrix.shape

        embedding = nn.Embedding(num_embeddings, embedding_dim)
        embedding.load_state_dict({"weight": torch.tensor(weights_matrix)})

    else:
        # simple one-hot-word embedding
        hidden_size = 500
        embedding = nn.Embedding(voc.__len__(), hidden_size)

    if loadFilename:
        embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = Decoder(
        attn_model, embedding, hidden_size, voc.__len__(), decoder_n_layers, dropout
    )
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print("Models built and ready to go!")

    ################### TRAINING ###############################################
    if option == '1':
        full_training(
            model_name,
            voc,
            pairs,
            encoder,
            decoder,
            embedding,
            encoder_n_layers,
            decoder_n_layers,
            save_dir,
            batch_size,
            loadFilename,
            encoder_optimizer_sd,
            decoder_optimizer_sd,
            MAX_LENGTH,
            device,
        )

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, device)

    # If you want to chat with the model, uncomment the following method

    # evaluate_input(encoder, decoder, searcher, voc, device, MAX_LENGTH)


    ####################################################3# CHATBOT CONVERSATIONS
    save_file = os.path.join(
        os.path.join("data_app/generator", "fake_conversations"), "conversation_"+attn_model+".txt"
    )

    create_conversations(
        encoder,
        decoder,
        searcher,
        voc,
        device,
        MAX_LENGTH,
        save_file,
        sentences_lengths,
        lines,
    )

    print("Conversations will be saved in :" + save_file)

run_generator()
