"""
Train the GPTLanguage model
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from load_data import getDataset
from model import GPTLanguageModel
import yaml
import os
from pathlib import Path
import logging
from bigram_model import estimate_loss, BigramLanguageModel
from configs import from_dict
from omegaconf import OmegaConf
import tqdm

def main(args):
    # Init config
    cli_config = OmegaConf.from_cli(args)

    # load the configuration settings
    if 'config_file' not in cli_config:
        config_file = 'configs/bigram.yaml'
    else:
        config_file = cli_config['config_file']

    # read config file
    config_from_file = OmegaConf.load(config_file)

    # overide config with command line args
    config = OmegaConf.merge(config_from_file, cli_config)
    print('Configuration:', config)

    # Init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    # if cuda available enable it
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != "cpu":
        #BUG fix 
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] ='0'

    # Define model
    vocab_size = config.dataset.vocab_size
    if args.model == "GPT":
        model = GPTLanguageModel(vocab_size=vocab_size, block_size=config.general.block_size,
                                 n_layer=config.model.n_layers, num_heads= config.model.num_heads,
                                 n_emb=config.model.n_emb, dropout=config.model.dropout)
    else:
        model = BigramLanguageModel(vocab_size)
    
    torch.manual_seed(1337)
    # Prepare dataset
    dataset = getDataset(text_file=config.dataset.fname, block_size=config.general.block_size, 
                         batch_size=config.training.batch_size)
    
    model.to(device)
    if not config.training.train:
        # just print loss without training (to check if model works)
        logger.info("Without training") 
        xb, yb = dataset.get_batch("train", config.dataset.train_split)                    
        logits, loss = model(xb, yb)
        logger.info(f"Loss without training: {loss.item()} ")

    else:
        # Train
        optimizer = torch.optim.AdamW(model.parameters(), config.training.lr)

        logger.info('Number of paramaters (M):',
        sum(p.numel() for p in model.parameters()) / 1e6)

        # if compile the model
        if config.training.compile:
            model = torch.compile(model)

        # Typical pytorch training loop
        logger.info(f"Training on {device}\n")
        best_loss = float('inf')
        running_loss = 0
        prog_bar = tqdm.trange(config.training.iterations)

        for iter in prog_bar:
            optimizer.zero_grad(set_to_none=True)
            
            # evaluate model on a specific iteration
            if iter % config.training.eval_interval == 0:
                losses = estimate_loss(config.training.eval_iters, model=model, dataset=dataset)
                train_loss, val_loss = losses["train"], losses["val"]
                prog_bar.set_description(
                f'train loss:{(train_loss)/(iter+1):.4f}, {val_loss=:.4f}')
                if best_loss >= val_loss:
                    best_loss = val_loss
                      # Save the model state, iteration, and other metadata
                    checkpoint = {
                        'iteration': iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                torch.save(checkpoint, os.path.join(args.save_ckpt, 'best_model_checkpoint.pth'))
                # print(f"step {iter: 05d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


            # Sample a batch of data
            xb, yb =  dataset.get_batch("train", config.dataset.train_split) 
            # Evaluate the loss
            _, loss = model(xb, yb)
            running_loss += loss
            loss.backward()
            optimizer.step()
        print()
        logger.info(f"Loss after training: {running_loss.item()/config.training.iterations}")

    # Generate the text
    print("\nThe AI poet:")
    print(dataset.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=config.inference.max_new_tokens)[0].tolist()))


# if the file directly run from the terminal
if __name__ == "__main__":
    # parse the args to the function
    main(sys.argv[1:])