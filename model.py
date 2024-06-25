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

        # dropout to randomly shut off some activations
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
        out = attn_wei @ v
        return out

if __name__ == "__main__":
    emb_dim = 32
    head_dim = 16
    dropout = 0.3
    attention_head = Head(emb_dim, head_dim=head_dim, dropout=dropout)
    x = torch.rand(4, 8, 32)
    output = attention_head(x)
    print(output.shape) # B, T, head_dim

    
