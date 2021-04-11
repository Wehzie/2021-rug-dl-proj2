# 2021 RUG Deep Learning Project 2

We develop two models.
First, conversation producing model, where two instances of this model aim to reproduce human interaction
Second, a judge model that evaluates how human a conversation is.
Both models operate on text data.

## Install Requirements

Navigate to project root directory.

    pip install -r requirements.txt

Installing PyTorch may be less trivial; consult the documentation since installation depends on your hardware.

## Running

### Discriminator

To run the training
  1. Ensure the variable 'train_mode' in line 25 in main.py is set to 'True'
  2. Subsequently run main.py
  
To test the generator data on the discriminator
  1. Ensure the variable 'train_mode' in line 25 in main.py is set to 'False'
  2. Subsequently run main.py

### Generator

To run the model simply run the file model.py using python3.

If you want to use GloVe embeddings you first have to download them from here:
<https://www.kaggle.com/anindya2906/glove6b>

Get file glove.6B.100d.txt.

Add the .txt file to the Generator/glove folder and then select the option when
prompted by model.py.

In case you want to actually chat with the model uncomment line 177 from model.py.
Press 'q' or type 'quit' to quit the chat.

## Informal References

### Tutorials

We used the following tutorials to build our app.

- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

### Dataset

https://www.aclweb.org/anthology/I17-1099/