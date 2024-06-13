"""
read, encode, and load the dataset

python load_data.py
"""
from pathlib import Path
import os
import sys
from typing import Union

import torch

class getDataset():
    def __init__(self, text_file: Union[str,Path], block_size: int = 8, 
                 batch_size: int = 4):
        self.text_file = text_file
        self.block_size = block_size
        self.batch_size = batch_size
        
        self.text = self.get_text()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # creating a simple lookup table for encoding and decoding the text
        stoi = {ch : i for i, ch in enumerate(self.chars) }
        itos = {i: ch for i, ch in enumerate(self.chars) }
        self.encode = lambda s: [stoi[c] for c in s] 
        self.decode = lambda l: ''.join([itos[i] for i in l])

        # create dataset
        torch.manual_seed(1337) # reproducibility with Andrej Karpathy's implementation
        self.data = torch.tensor(self.encode(self.text), dtype= torch.long)

    def get_text(self):
        """
        read text file adn return text
        """
        try:
            # don't add 'utf-8' encoding in here
            with open(self.text_file, 'r') as file:
                text = file.read()
            return text
        except FileNotFoundError:
            print(f"File not Found: {self.text_file}")
            sys.exit(1)
    
    def split_data(self, split_ratio: float = 0.9):
        """
        split the dataset into train and validation with
        split_ratio, default = 0.9 (90%)

        Parameters
        ----------
        split_ratio: float (split ratio)

        Return
        ------
        train: torch.Tensor (train set)
        val: torch.Tensor  (validation set)
        """
        n_split = int(split_ratio * len(self.data)) 
        train_data = self.data[:n_split]
        val_data = self.data[n_split:]
        return (train_data, val_data)
    
    def get_batch(self, split: str = "train", train_val_split: float = 0.9):
        """
        get a batch of xs and ys of train or validation dataset

        Parameters
        ----------
        split: str (name of the dataset)
        train_val_split: split the actual dataset into train and validation

        Return
        ------
        x: torch.Tensor (batch of input)
        y: torch.Tensor (batch of labels)
        """
        # generate a small batch of data of inputs x and targets y
        train_data, val_data = self.split_data(train_val_split)
        data = train_data if split == 'train' else val_data # data selector
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in ix])
        return (x, y)
        

# test code
if __name__ == "__main__":
    dataset = getDataset(text_file='dataset/input.txt', block_size=8, 
                         batch_size=4)
    
    input, gt = dataset.get_batch("train", 0.9)
    
    print('inputs:')
    print(input.shape)
    print(input)
    print('targets:')
    print(gt.shape)
    print(gt)

        
    
        