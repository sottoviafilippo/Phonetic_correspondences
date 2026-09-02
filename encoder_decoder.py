import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np



# roughly trying to follow the structure of the "Attention is all you need" paper https://arxiv.org/abs/1706.03762

# share embedding for both sides - for now both languages share latin alphabet. to be changed later for more general cases

# no unicode or normalize calls on the strings to preserve the accents

# TO DO: define positional encoding as a module with a buffer for better efficiency


def PositionalEncoding(input_data : torch.Tensor, base_den: float = 500) -> torch.Tensor:
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
    # no masking here

    def __init__(self, d_model, dk, dv):
        super().__init__()

        # d_model: embedding dimension
        self.dk = dk

        self.WQ = nn.Linear(d_model, dk)
        self.WK = nn.Linear(d_model, dk)
        self.WV = nn.Linear(d_model, dv)
        self.WO = nn.Linear(dv, d_model) # output projection

    def forward(self, x):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        matt = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.dk) # transpose(-2,-1): swap the last 2 dims
        matt_softmax = torch.softmax(matt, -1) # softlax is applied within each row, so on the last dim      

        return self.WO(torch.matmul(matt_softmax, V))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, dk:int, dv: int, h: int, masking: bool = True):
        super().__init__()

        # d_model embedding dimension
        self.dk = dk
        self.dv = dv
        self.d_model = d_model
        self.h = h # number of heads
        self.masking = masking

        self.WQ = nn.Linear(d_model, h * dk)
        self.WK = nn.Linear(d_model, h * dk)
        self.WV = nn.Linear(d_model, h * dv)
        self.WO = nn.Linear(h * dv, d_model)
        
        def forward(self, x):

            batch_length, sequence_length, _ = x.shape

            # self.WQ(x) has shape (batch_length, sequence_length, h*dk). 
            # I need to bring it to (batch_length, h, sequence_length, dk): 
            # after the batch dimension, h sequence_length x dk matrices one after the other
            # Same K and V (V having dv - a priori not the same as dk)
            # for this, use view and then transpose the right indices
            Q = self.WQ(x).view(batch_length, sequence_length, self.h, self.dk).transpose(-3, -2)
            K = self.WK(x).view(batch_length, sequence_length, self.h, self.dk).transpose(-3, -2)
            V = self.WV(x).view(batch_length, sequence_length, self.h, self.dv).transpose(-3, -2)

            matt = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.dk) 
            # transpose(-2,-1): swap the last 2 dims (corresponds to K^T in the attention formula)

            if self.masking: # cancel contributions from positions yet to be found: basically preserve causality
                mask = torch.tril(torch.ones(sequence_length, sequence_length)) # is 1 on the diag and below, 0 elsewhere
                matt = matt.masked_fill(mask == 0, float("-inf"))

            matt_softmax = torch.softmax(matt, -1) # softmax is applied within each row, so on the last dim    

            V_times_softmax = torch.matmul(matt_softmax, V)

            # now I have to multiply with V, which has dimensions (sequence_length, dv) (thought of as a matrix)  
            # so first I concatenate on index 1 (corresponding to the h heads)
            # to do this rewriting I use view as before, but in the other direction 
            # (need contiguous before to rewrite the dimensions from scratch afterwards)
            
            V_times_softmax = V_times_softmax.view.transpose(-3, -2).contiguous().view(batch_length, sequence_length, self.h * sequence_length)

            # and now I can project out to d_model dimension with W0 and return  
            return self.WO(V_times_softmax)



class MultiHeadCrossAttention(nn.Module):
    pass


class Transformer(nn.Module):

    def __init__(self, d_model, vocab_size, char_to_idx, max_len = 20, n_heads = 4, n_layers = 2, feedforward_hidden_dim_to_d_model_ratio = 4):
        # for starters start with a light model, just to check its workings
        super().__init__()

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # use gpu if possible (mac)
    
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
    