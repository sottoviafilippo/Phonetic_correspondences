import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np



# roughly trying to follow the structure of the "Attention is all you need" paper https://arxiv.org/abs/1706.03762

# share embedding for both sides - for now both languages share latin alphabet. to be changed later for more general cases

# no unicode or normalize calls on the strings to preserve the accents

# TO DO: define positional encoding as a module with a buffer for better efficiency


def PositionalEncoding(input_data : torch.Tensor, base_den: int = 500) -> torch.Tensor:
    # in the original paper what I call base_den is 10000 but here I reduce it because I have a much smaller number of tokens (around 20)
    # define as a function for simplicity since it is fixed 

    # format of input_data: (batch, sequence_length, d_model)

    sequence_length = (input_data.shape)[-2]
    d_model = (input_data.shape)[-1]

    positions = torch.arange(sequence_length).unsqueeze(1).float() # unsqueeze gives dimension (seguence_length, 1)

    # compute 1/ff ** (2i / d_model) (called denominators below)
    # use exp log for numerical stability
    # torch.arange(0, d_model, 2) gives [0, 2, 4, ...]
    denominators = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(base_den) / d_model))  # dimension (d_model/2,) 

    pos_enc = torch.zeros(sequence_length, d_model)
    pos_enc[:,0::2] = torch.sin(positions * denominators) # even dimensions get sin (i/(...))
    pos_enc[:,1::2] = torch.cos(positions * denominators) # odd dimensions get cos (((i - 1))/(...))
    # note that usually the even/odd dimensions pairs are given as (2i, 2i + 1) with i going up to d_model/2
    # so one just gives 2i also for cos - this caused me some confusion

    return input_data + pos_enc # automatically broadcasts over the batch dimension



class FeedForward(nn.Module):
    pass


class MultiHeadAttentionLayer(nn.Module):
    pass


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class EncoderLayer(nn.Module):
    pass


class DecoderLayer(nn.Module):
    pass


class OneHeadAttention(nn.Module):
    "dot-product attention. one head, defined first as simpler example compared with multihead. will become redundant later"

    def __init__(self, d_model, dk, dv):
        super().__init__()

        self.d_model = d_model # embedding dimension
        self.dk = dk
        self.dv = dv

        self.WQ = nn.Linear(d_model, dk)
        self.WK = nn.Linear(d_model, dk)
        self.WV = nn.Linear(d_model, dv)


    def forward(self, x):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        matt = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.dk) # transpose(-2,-1): swap the last 2 dims
        matt_softmax = torch.softmax(matt, -1) # softlax is applied within each row, so on the last dim      

        return torch.matmul(matt_softmax, V)  


class MultiHeadAttention(nn.Module):
    pass


class MultiHeadCrossAttention(nn.Module):
    pass


class Transformer(nn.Module):

    def __init__(self, d_model, vocab_size, char_to_idx, max_len = 20, n_heads = 4, n_layers = 2, feedforward_hidden_dim_to_d_model_ratio = 4):
        # for starters start with a light model, just to check its workings
        super().__init__()
    
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.char_to_idx = char_to_idx
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.feedforward_hidden_dim = feedforward_hidden_dim_to_d_model_ratio * d_model # 4 in 1706.03762 paper 
    
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=char_to_idx['<pad>']) 
        #the embedding lives directly in Transformer because Encoder and Decoder share it
    
        # now need to add positional (sinusoidal) encoding (fixed, not learned)
        # PositionalEncoding(sequence_length, self.d_model)
        
        #encoder, decoder ...
        #finally, linear output projection ...


    def train():
        pass

    def train_from_txt():
        #words separated by ;
        pass
    