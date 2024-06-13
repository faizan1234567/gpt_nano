"""
read, encode, and load the dataset
"""
from pathlib import Path
import os
import sys
from typing import Union

class getDataset():
    def __init__(self, text_file: Union[str,Path], train_test_split: float = 0.9):
        self.text_file = text_file
        self.train_test_split = train_test_split

        self.chars = sorted(list(set(self.get_text())))
        self.vocab_size = len(self.chars)

        # creating a simple lookup table for encoding and decoding the text
        stoi = {ch : i for i, ch in enumerate(self.chars) }
        itos = {i: ch for i, ch in enumerate(self.chars) }
        self.encode = lambda s: [stoi[c] for c in s] 
        self.decode = lambda l: ''.join([itos[i] for i in l])

    def get_text(self):
        """
        read text file adn return text
        """
        try:
            with open(self.text_file, 'r', 'utf-8') as file:
                text = file.read()
            return text
        except FileNotFoundError:
            sys.exit(1)
            return 
    

        
    
        