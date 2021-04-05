# Read me for Chatbot training/chatting

To run the training and chat:
    1. Uncomment lines: 181-184 in model.py
    2. Comment lines: 142-144 in model.py

To load a model and chat:
    1. Comment lines: 181-184 in model.py
    2. Uncomment lines: 142-144 in model.py

To access the the output/input in the chat:
    1. Go to evaluate.py
    2. The function evaluateInput handles both input and output
    3. input_sentence gets the input from the keyboard but can be modified
    4. output_words contains all the decoded words(the response)
