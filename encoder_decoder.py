import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np


# roughly trying to follow the structure of the "Attention is all you need" paper https://arxiv.org/abs/1706.03762

# share embedding for both sides - for now both languages share latin alphabet. to be changed later for more general cases

# no unicode or normalize calls on the strings to preserve the accents

class PositionalEncoding(nn.Module):
    # reasons for sinusoidal encoding: - bounded - multiple frequencies - relative 

    def __init__(self):
            super().__init__()



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
    pass