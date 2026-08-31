import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# roughly trying to follow the structure of the "Attention is all you need" paper https://arxiv.org/abs/1706.03762

# share embedding for both sides - for now both languages share latin alphabet. to be changed later for more general cases

# no unicode or normalize calls on the strings to preserve the accents

def PositionalEncoding(sequence_length, d_model):
    # define as a function for simplicity since it is fixed 
    pass


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


class Attention(nn.Module):
    "dot-product attention"
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
        self.feedforward_hidden_dim = feedforward_hidden_dim_to_d_model_ratio * d_model # 4 in 1706.03762 paper (32 * 4 = 128)
    
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
    