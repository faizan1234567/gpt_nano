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
python bigram_model.py --cfg <path>
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


if __name__ == "__main__":
    # Read from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/bigram.yaml', help='config file path')
    args = parser.parse_args()

    # Init config
    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)  # convert dict to object

    # Init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    # Define model
    vocab_size = config.dataset.vocab_size
    model = BigramLanguageModel(vocab_size)
    
    # Prepare dataset
    dataset = getDataset(text_file=config.dataset.fname, block_size=config.general.block_size, 
                         batch_size=config.training.batch_size)
    
    
    if not config.training.train:
        logger.info("Without training") 
        xb, yb = dataset.get_batch("train", config.dataset.train_split)                    
        logits, loss = model(xb, yb)
        logger.info(f"Loss without training: {loss.item()} ")

    else:
        # Train
        optimizer = torch.optim.AdamW(model.parameters(), config.training.lr)

        # Typical pytorch training loop
        logger.info("Training")
        for iter in tqdm(range(config.training.iterations), mininterval=0.2):

            # Sample a batch of data
            xb, yb =  dataset.get_batch("train", config.dataset.train_split) 

            # Evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        logger.info(f"Loss after training: {loss.item()}")

    # Generate the text
    print("\nThe AI poet:")
    print(dataset.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=config.inference.max_new_tokens)[0].tolist()))
