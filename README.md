# GPT-Nano: A light weight GPT model for text generation
Building a small text generation model from scratch. Inspired by the [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6533s&ab_channel=AndrejKarpathy) by Andrej Karpathy.

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


## Usage

All the settings are located under `configs/`. There are two model variants in `configs `: `bigram.yaml` and `gpt.yaml`.  The last one is heavy, you might need a GPU machine to run this. It will about 4GB of GPU RAM with default settings. 

To train the model, simply run `python train.py`. By default, the `bigram.yaml` config file will be loaded. To load GPT model, run 

```
python train.py config_file=configs/gpt.yaml
```
You can change other parameter on command line, like the `batch_size`, `block_size`, `learning rate`, number of `iterations`, and other parameters. 

```
python train.py config_file=configs/gpt.yaml  training.batch_size=64 training.iterations=10000 general.block_size=512 model.num_heads=6
```

## Sample Output
Training 4 layer GPT decoder model on William Shakespeare's text. The sample file is located under `output` directory. The model learns the structure and starts to geneartes accurate and grammatically correct words considering small architecture, dataset, and training for just 5000 iterations. 

```bash
YORTHUMBERLAND:
Why, would yet stay be enought
That he, the return by thy honour wrords bloody.

ROMEO:
There is froth, no meat ta'en, that all sad
And winkless the impress'd that if thou lovest.
Heart me, and hadst then droverenses with a pite.

JULIET:
I would thorow' the comes to deep to bed.
Ah, for I! nay fearest good my swife in of the,
This thoughts form oflly: if he refurnut no guess:
As heree in hope other by all of grainty with contems
For I be fear a despas; blessing thy warrant to daughter:
'Tmer thou not brow, if she beggets 'he be to lives;
Exp not selfs and drop himsh's boar;
And he that I have done them as lives a doth feel.
I
DUKE VINCENTIO:
I that thou know thou canst worth of thee.

DUKE VINCENTIO:
But too, on thou lovick that is but spitent
As breast knees his bend wit ripecial what life.
```

## Improvments
- Saving best checkpoints as the loss decreases 
- Seperating configuration settings and creating seperate files for each bigram and GPT  models
- Writing training loop for each of the model
- data loading

## ToDo
- [ ] Adding a seperate text generation script and streaming the model output with Gradio
- [ ] Training on Urdu dataset, suggestions welcome ;) 
- [ ] Adding Multi GPU training support for large dataset for bigger network as we scale up
- [x] chaning configs values with command line args
- [x] Add installation / virtual environment instructions
- [x] Other tokenization techniques


# Acknowledgements
- This repository is based on the fantastic [video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Karpathy. 
- The transformer architecture was introduced in the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762).
- The Shakespeare dataset [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Command line code implementation and configs parameter values from [this cool repoistory](https://github.com/Usman-Rafique/GPT-Nano/tree/main)

