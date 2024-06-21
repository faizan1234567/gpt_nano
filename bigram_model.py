"""
Train a bigram language model to predict the next word in the sequence using only 
the previous word. 

A bigram language model is a type of probabilistic language model that 
predicts the next word in a sequence based on the previous word. It considers pairs 
of consecutive words (bigrams) to estimate the likelihood of a word following 
another word. This model relies on the Markov assumption, which in this case, implies 
that the probability of a word only depends on the preceding word.

Author: Muhammad Faizan
python bigram_model.py -h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same as Andrej Karpathy used to reproduce the random numbers
torch.manual_seed(1337) 
from load_data import getDataset



if __name__ == "__main__":
    pass