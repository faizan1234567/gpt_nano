"""
implementation of building block of transformer decoder architecutre
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from load_data import getDataset


# Single head
class Head(nn.Module):
    """one single head of self attention"""
    def __init__(self, n_emb:int, head_dim: int, bias: bool = False, 
                 dropout: float = 0.2):
        super().__init__()
        self.n_emb = n_emb
        self.head_dim = head_dim
        self.bias = bias
        self.dropout = dropout

        self.query = nn.Linear(self.n_emb, self.head_dim, bias=bias)
        self.key = nn.Linear(self.n_emb, self.head_dim, bias=bias)
        self.value = nn.Linear(self.n_emb, self.head_dim, bias=bias)
        self.register_buffer("tril", torch.tril(torch.ones(self.n_emb, self.n_emb)))

        # dropout to randomly turn off some activations 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        # B, T, head_size
        query = self.query(x) 
        key = self.query(x)
        # (B, T, head_size) (B, head_size, T) --> (B, T, T)
        attn_wei = query @ key.transpose(-2, -1) * self.head_dim ** -0.5 
        attn_wei = attn_wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        attn_wei = F.softmax(attn_wei, dim=-1)
        # Apply dropout
        attn_wei = self.dropout(attn_wei)
        v = self.value(x)
        # Weighted aggregation of the values
        # (B, T, T) (B, T, head_size) --> (B, T, head_size)
        out = attn_wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multi head in parrellel"""
    def __init__(self, num_heads: int = 4, n_emb:int =32, head_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_dim
        self.dropout = dropout

        self.heads = [Head(n_emb=n_emb, head_dim= head_dim) for _ in range(num_heads)]
        self.proj = nn.Linear(head_dim * num_heads, n_emb)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    """feed forward layer followed by non linearity"""
    def __init__(self, n_emb: int, dropout: float = 0.2):
        super().__init__()
        self.n_emb = n_emb
        self.dropout = dropout
        self.net = nn.Sequential(nn.Linear(n_emb, 4*n_emb), 
                                 nn.ReLU(), 
                                 nn.Linear(4*n_emb, n_emb), 
                                 nn.Dropout(dropout))
    
    def forward(self, x):
        x = self.net(x)
        return x


class Block(nn.Module):
    """Transformer block: self attention, feed forward net and all the other
    components"""
    def __init__(self, n_emb: int, n_heads: int, dropout: float = 0.3):
        super().__init__()
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.dropout = dropout

        head_size = n_emb // n_heads
        # Multi head self attention
        self.sa = MultiHeadAttention(num_heads= n_heads, n_emb= n_emb, head_dim= head_size, 
                                     dropout= self.dropout)
        # Feed forward layer
        self.mlp = MLP(n_emb= n_emb, dropout= self.dropout)
        # Layer Norms
        self.l1 = nn.LayerNorm(n_emb)
        self.l2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.l1(x))
        x = x + self.mlp(self.l2(x))
        return x        


class GPTLanguageModel(nn.Module):
    """GPT character level language model"""
    def __init__(self, vocab_size: int = 65, block_size: int = 8, n_layer: int = 4, num_heads: int = 4, n_emb: int =32, 
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.num_heads = num_heads
        self.n_emb = n_emb
        self.dropout = dropout
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Each token reads off the logits for the next token from the lookup table
        self.tokens_embedding_table = nn.Embedding(self.vocab_size, self.n_emb)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb=self.n_emb, n_heads=self.num_heads, dropout= self.dropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_emb)
        self.lm_head = nn.Linear(self.n_emb, self.vocab_size)

        # Initialize model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target = None):
        B, T = idx.shape
        
        # (B, T, C)
        token_emb = self.tokens_embedding_table(idx)
        # (T, C)
        positional_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        # (B, T, C)
        x = token_emb + positional_emb
        print(x.device)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # (B, T, vocab_size)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = target.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # Generate text
    def generate(self, idx, max_new_tokens):
        # Idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index to last blok size token
            idx_cond = idx[:, -self.block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    
