import torch
import torch.nn as nn
#import torch.optim as optim
#from typing import Callable
import numpy as np



# roughly trying to follow the structure of the "Attention is all you need" paper https://arxiv.org/abs/1706.03762

# share embedding for both sides - for now both languages share latin alphabet. to be changed later for more general cases

# no unicode or normalize calls on the strings to preserve the accents

# TO DO LIST

# need padding mask in main transformer class

# define mask outside for better efficiency

# modern papers use pre-layer norm (more stable)



def PositionalEncoding(input_data : torch.Tensor, base_den: float = 500) -> torch.Tensor:
    # in the original paper what I call base_den is 10000 but here I reduce it because I have a much smaller number of tokens (around 20)
    # define as a function for simplicity since it is fixed 

    # format of input_data: (batch, sequence_length, d_model)

    sequence_length = (input_data.shape)[-2]
    d_model = (input_data.shape)[-1]

    # Get the device of the input
    device = input_data.device
    positions = torch.arange(sequence_length, device=device).unsqueeze(1).float()
    
    # compute 1/ff ** (2i / d_model) (called denominators below)
    # use exp log for numerical stability
    # torch.arange(0, d_model, 2) gives [0, 2, 4, ...]
    denominators = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(base_den) / d_model)) 

    pos_enc = torch.zeros(sequence_length, d_model, device=device)
    pos_enc[:,0::2] = torch.sin(positions * denominators) # even dimensions get sin (i/(...))
    pos_enc[:,1::2] = torch.cos(positions * denominators) # odd dimensions get cos (((i - 1))/(...))
    # note that usually the even/odd dimensions pairs are given as (2i, 2i + 1) with i going up to d_model/2
    # so one just gives 2i also for cos - this caused me some confusion

    return input_data + pos_enc # automatically broadcasts over the batch dimension


class PositionalEncodingModule(nn.Module):
    # defined as module with a buffer for better efficiency
    # precomputes the encoding once up to max_len and slices it per forward call,
    # instead of recomputing sin/cos every time (like the PositionalEncoding function)

    def __init__(self, d_model: int, max_len: int = 20, base_den: float = 500):
        # in the original paper what I call base_den is 10000 but here I reduce it because I have a much smaller number of tokens (around 20)
        super().__init__()

        positions = torch.arange(max_len).unsqueeze(1).float()

        # compute 1/ff ** (2i / d_model) (denominators), using exp/log for numerical stability
        # torch.arange(0, d_model, 2) gives [0, 2, 4, ...]
        denominators = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(base_den) / d_model))

        pos_enc = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(positions * denominators)  # even dims get sin
        pos_enc[:, 1::2] = torch.cos(positions * denominators)  # odd dims get cos

        # assert sequence_length <= self.pos_enc.shape[1], "sequence longer than max_len"

        pos_enc = pos_enc.unsqueeze(0)  # shape (1, max_len, d_model), broadcasts over batch

        # register as buffer: moves with .to(device)/.cuda(), saved in state_dict by default,
        # but not treated as a learnable parameter (no gradient, no optimizer update)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # format of input_data: (batch, sequence_length, d_model)
        sequence_length = input_data.shape[-2]

        # check sequence length is not > max_len
        assert sequence_length <= self.pos_enc.shape[1], (
                f"sequence_length ({sequence_length}) exceeds max_len "
                f"({self.pos_enc.shape[1]}) the positional encoding was built with"
            )
        
        return input_data + self.pos_enc[:, :sequence_length, :]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_hidden), 
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.network(x)



class EncoderLayer(nn.Module):
    # embedding and position encoding to be applied before calling the encoder
    def __init__(self, d_model:int, d_hidden: int, dk:int, dv: int, h: int):
        super().__init__()

        self.mhattention = MultiHeadAttention(d_model, dk, dv, h, masking = False)
        self.fforward = FeedForward(d_model, d_hidden)
        # define two norms with independent parameters (gamma and beta, see https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html):
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model) 

    def forward(self, x):
        attention_output = self.mhattention(x)
        x_and_attention = self.norm1(x + attention_output) # add and norm
        return self.norm2(x_and_attention + self.fforward(x_and_attention)) # add and norm


class DecoderLayer(nn.Module):
    # embedding and position encoding to be applied before calling the encoder
    def __init__(self, d_model:int, d_hidden: int, dk:int, dv: int, h: int):
        super().__init__()
    
        self.mhattention = MultiHeadAttention(d_model, dk, dv, h, masking = True) # need masking here
        self.crossattention = MultiHeadCrossAttention(d_model, dk, dv, h)
        self.fforward = FeedForward(d_model, d_hidden)
        # define three norms with independent parameters:
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) 
    
    def forward(self, x, y):
        # x: input (encoded), y: output
        attention_output = self.mhattention(y)
        y_and_attention = self.norm1(y + attention_output) # add and norm
        cross_attention_xy = self.crossattention(y_and_attention, x)
        y_and_cross_attention = self.norm2(y_and_attention + cross_attention_xy) # add and norm
        ff_y_and_cross = self.fforward(y_and_cross_attention)

        return self.norm3(ff_y_and_cross + y_and_cross_attention)


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model:int, d_hidden: int, dk:int, dv: int, h: int):
        # n_layers: number of times the encoder is repeated. in the original paper it was equal to 6
        super().__init__()

        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_hidden, dk, dv, h) for i in range(n_layers)])

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model:int, d_hidden: int, dk:int, dv: int, h: int):
        super().__init__()

        self.decoder = nn.ModuleList([DecoderLayer(d_model, d_hidden, dk, dv, h) for i in range(n_layers)])

    def forward(self, x, y):
        for dec in self.decoder:
            y = dec(x, y)
        return y



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
            # TO DO: define the mask outside for better efficiency
            mask = torch.tril(torch.ones(sequence_length, sequence_length, device=matt.device)) # is 1 on the diag and below, 0 elsewhere
            matt = matt.masked_fill(mask == 0, float("-inf"))

        matt_softmax = torch.softmax(matt, -1) # softmax is applied within each row, so on the last dim    

        softmax_mult_V = torch.matmul(matt_softmax, V)

        # now I have to multiply with V, which has dimensions (sequence_length, dv) (thought of as a matrix)  
        # so first I concatenate on index 1 (corresponding to the h heads)
        # to do this rewriting I use view as before, but in the other direction 
        # (need contiguous before to rewrite the dimensions from scratch afterwards)
            
        softmax_mult_V  = softmax_mult_V.transpose(-3, -2).contiguous().view(batch_length, sequence_length, self.h * self.dv)

        # and now I can project out to d_model dimension with W0 and return  
        return self.WO(softmax_mult_V)



class MultiHeadCrossAttention(nn.Module):
    # to be used in the Decoder
    # for comments see the MultiHeadAttention class

    def __init__(self, d_model:int, dk:int, dv: int, h: int):
        super().__init__()

        # d_model embedding dimension
        self.dk = dk
        self.dv = dv
        self.d_model = d_model
        self.h = h 

        self.WQ = nn.Linear(d_model, h * dk)
        self.WK = nn.Linear(d_model, h * dk)
        self.WV = nn.Linear(d_model, h * dv)
        self.WO = nn.Linear(h * dv, d_model)
        
    def forward(self, queries, keys):
        # no masking for cross-attention: one obviously looks at all queries every time

        batch_length, target_length, _ = queries.shape
        _, input_length, _ = keys.shape

        Q = self.WQ(queries).view(batch_length, target_length, self.h, self.dk).transpose(-3, -2)
        K = self.WK(keys).view(batch_length, input_length, self.h, self.dk).transpose(-3, -2)
        V = self.WV(keys).view(batch_length, input_length, self.h, self.dv).transpose(-3, -2)

        matt = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.dk) 
        matt_softmax = torch.softmax(matt, -1)    
        V_times_softmax = torch.matmul(matt_softmax, V).transpose(-3, -2).contiguous().view(batch_length, target_length, self.h * self.dv)

        return self.WO(V_times_softmax)
    


class Transformer(nn.Module):

    def __init__(self, d_model, vocab_size, char_to_idx, dk, dv, max_len = 20, n_heads = 4, n_layers = 2, feedforward_hidden_dim_to_d_model_ratio = 4):
        # for starters start with a light model, just to check its workings
        # char_to_idx : dictionary from char to int

        super().__init__()

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        ) # use gpu if possible (mac)
    
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.char_to_idx = char_to_idx # char to int dictionary
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.feedforward_hidden_dim = feedforward_hidden_dim_to_d_model_ratio * d_model # 4 in 1706.03762 paper 

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=char_to_idx['<pad>']) 
        self.pos_encoding = PositionalEncodingModule(d_model, max_len=max_len)
        # the embedding lives directly in Transformer because Encoder and Decoder share it

        self.encoder = Encoder(n_layers, d_model, self.feedforward_hidden_dim, dk, dv, h = n_heads)
        self.decoder = Decoder(n_layers, d_model, self.feedforward_hidden_dim, dk, dv, h = n_heads)

        self.exit_linear_projection = nn.Linear(d_model, vocab_size) # in the original paper they use weight tying (basically transpose embed)

        self.to(self.device)


    def forward(self, x, y):
        # x: source, y: target sequence
        
        x_embedded_pos = self.pos_encoding(self.embed(x)*np.sqrt(self.d_model))
        x_encoded = self.encoder(x_embedded_pos)

        y_embedded_pos = self.pos_encoding(self.embed(y)*np.sqrt(self.d_model))
        y_decoded = self.decoder(x_encoded, y_embedded_pos)
        # will need padding mask
          
        return self.exit_linear_projection(y_decoded)


    def generate_sequence(self, x, max_len = 20):
        """generates a sequence from an input one"""

        # first check that the input sequence length is not > max_len
        assert max_len <= self.pos_encoding.pos_enc.shape[1], (
            f"generate_sequence max_len ({max_len}) exceeds the model's positional "
            f"encoding max_len ({self.pos_encoding.pos_enc.shape[1]}); "
            f"rebuild the model with a larger max_len if longer sequences are needed"
        )

        self.eval()

        with torch.no_grad():
            x_encoded = self.encoder(self.pos_encoding(self.embed(x) * np.sqrt(self.d_model))) # encode the input one for all
            y = torch.tensor([[self.char_to_idx['<sos>']]], device=self.device) # initialize y with <sos> token

            for k in range(max_len):
                decoded_output = self.decoder(x_encoded, self.pos_encoding(self.embed(y) * np.sqrt(self.d_model)))
                output_logits = self.exit_linear_projection(decoded_output)
                output_probs = torch.softmax(output_logits, dim=-1)

                # Get the prediction for the last time step. greedy: argmax: would not need to go through softmax. 
                # to be changed later if I want to use softmax to sample randomly
                next_token = torch.argmax(output_probs[:,-1,:], dim=-1).unsqueeze(1)
            
                # Append the predicted token to y
                y = torch.cat([y, next_token], dim=1)

                if next_token.item() == self.char_to_idx['<eos>']:
                    break

        return y # it will by construction return a series of ints, will have to be translated to chars using the dictionary 

    def fit(self):

        # use cross entropy for the loss
        pass


    def fit_from_txt(self):
        #words separated by ;
        pass

    