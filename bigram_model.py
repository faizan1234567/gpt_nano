"""
=====================================================================================
Train a bigram language model to predict the next word in the sequence using only 
the previous word. 

A bigram language model is a type of probabilistic language model that 
predicts the next word in a sequence based on the previous word. It considers pairs 
of consecutive words (bigrams) to estimate the likelihood of a word following 
another word. This model relies on the Markov assumption, which in this case, implies 
that the probability of a word only depends on the preceding word.

Author: Muhammad Faizan
python bigram_model.py -h
=====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(1337) 

from load_data import getDataset

import argparse
import yaml
import os
from pathlib import Path
from configs import from_dict
import logging
from tqdm import tqdm


# Bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # Idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Get avg loss
@torch.no_grad()
def estimate_loss(eval_iters, device, model=None, dataset=None):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split, 0.9)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.to(device)
    model.train()
    return out