# GPT-Nano: A light weight GPT model
Building a small text generation model from scrath. Inspired by the [YouTube video lecture]() by Andrej Karpathy.

The purpose of this project is to understand the inner workings of large GPT models. Large GPT models are trained on extensive corpora of textual data sourced from books, the internet, and other repositories, utilizing powerful GPUs. Following GPT protocols, a transformer-based decoder model will be trained on a smaller dataset to predict the next character in a sequence, in contrast to modern GPT models that use subword-level tokenization.

Once trained, the model can generate text by starting with an initial random character. For example, if trained on Shakespeare's text, the model will produce text resembling Shakespeare's style.

However, because we are modeling characters and working with limited data and sequence lengths, we should not expect the model to generate semantically coherent text.

## Installation
```bash
 git clone https://github.com/faizan1234567/gpt_nano.git
 cd gpt_nano
```

Create  a virtual enviroment using python venv
```bash
python3 -m venv gpt_nano
source gpt_nano/bin/activate
```
alternatively, you can use anaconda package manager
```bash
conda create -n gpt_nano python=3.8.10 -y
conda activate gpt_nano
```

Now install all the required dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Installation done.!


## Usage

All the settings are located under `configs/`. There are two model variants in `configs `: `bigram.yaml` and `gpt.yaml`.  The last one is heavy, you might need a GPU Machine to run this. 

To train the model, simply run `python train.py -h`. By default, the `bigram.yaml` config file will be loaded. To load GPT model, run 

```
python train.py --cfg configs/gpt.yaml --model "GPT" --save_ckpt <path>
```

## Improvments
- Saving best checkpoints as the loss decreases 
- Seperating configuration settings and creating seperate files for each bigram and GPT  models
- Writing training loop for each of the model

## ToDo
- [ ] Adding a seperate text generation script and streaming the model output with Gradio
- [ ] Training on Urdu dataset, suggestions welcome ;) 
- [ ] Adding Multi GPU training support for large dataset for bigger network as we scale up
- [ ] chaning configs values with command line args
- [x] Add installation / virtual environment instructions
- [x] Other tokenization techniques


Suggestions and PRs welcome.

# Acknowledgements
- This repository is based on the [video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [GitHub repository](https://github.com/karpathy/ng-video-lecture) by Karpathy.
- The transformer architecture was introduced in the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762).
- The Shakespeare dataset [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Command line code implementation and configs parameter values from [this cool repoistory](https://github.com/Usman-Rafique/GPT-Nano/tree/main). 

