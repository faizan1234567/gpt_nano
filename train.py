"""
Train the GPTLanguage model
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
from load_data import getDataset
from model import GPTLanguageModel
import yaml
from pathlib import Path
import logging
from bigram_model import estimate_loss, BigramLanguageModel
from configs import from_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/bigram.yaml', help='config file path')
    parser.add_argument("--model", default="GPT", type = str, help="model to use for training either GPT or bigram" )
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

    # if cuda available train with it
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model
    vocab_size = config.dataset.vocab_size
    if args.model == "GPT":
        model = GPTLanguageModel(vocab_size=vocab_size, block_size=config.general.block_size,
                                 n_layer=config.model.n_layers, num_heads= config.model.num_heads,
                                 n_emb=config.model.n_emb, dropout=config.model.dropout)
    else:
        model = BigramLanguageModel(vocab_size).to(device)
    
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
        logger.info("Training\n")

        for iter in range(config.training.iterations):
             
            if iter % config.training.eval_interval == 0:
                losses = estimate_loss(config.training.eval_iters, device, model=model, dataset=dataset)
                print(f"step {iter: 05d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


            # Sample a batch of data
            xb, yb =  dataset.get_batch("train", config.dataset.train_split) 
            xb, yb  = xb.to(device), yb.to(device)
            # Evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print()
        logger.info(f"Loss after training: {loss.item()}")

    # Generate the text
    print("\nThe AI poet:")
    print(dataset.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=config.inference.max_new_tokens)[0].tolist()))