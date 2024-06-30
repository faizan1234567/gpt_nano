"""
implementation of building block of transformer decoder architecutre
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


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
        # Layer Norm
        self.l1 = nn.LayerNorm(n_emb)
        self.l2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.l1(x))
        x = x + self.mlp(self.l2(x))
        return x        




# test the code
if __name__ == "__main__":
    emb_dim = 32
    head_dim = 16
    dropout = 0.3
    # attention_head = MulftiHeadAttention(num_heads=4, n_emb=emb_dim, head_dim=head_dim, dropout=dropout)
    # net = MLP(emb_dim, 0.2)
    net = Block(n_emb=32, n_heads= 4, dropout= 0.3)
    x = torch.rand(4, 8, 32)
    output = net(x)
    print(output.shape) 

    
