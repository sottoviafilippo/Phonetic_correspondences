import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# roughly trying to follow the structure of the "Attention is all you need" paper https://arxiv.org/abs/1706.03762
# adapting it to time-series

# attention operaters across time (more advanced options: across variables or both)
# long input: sparse attention would make sense

# TO ADD: windowing function
# key_padding_mask still needed for sequences of different length?



class PositionalEncodingModule(nn.Module):
    # defined as module with a buffer for better efficiency (wrt to a function)
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
    def __init__(self, d_model, d_hidden, dropout: float = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_hidden), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.network(x)


class EncoderLayer(nn.Module):
    # embedding and position encoding to be applied before calling the encoder
    def __init__(self, d_model:int, d_hidden: int, dk:int, dv: int, h: int, dropout: float = 0.1):
        super().__init__()

        self.mhattention = MultiHeadAttention(d_model, dk, dv, h, dropout=dropout)
        self.fforward = FeedForward(d_model, d_hidden, dropout=dropout)
        # define two norms with independent parameters (gamma and beta, see https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html):
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output = self.mhattention(x)
        x_and_attention = self.norm1(x + self.dropout(attention_output)) # add and norm
        return self.norm2(x_and_attention + self.dropout(self.fforward(x_and_attention))) # add and norm


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model:int, d_hidden: int, dk:int, dv: int, h: int, dropout: float = 0.1):
        # n_layers: number of times the encoder is repeated. in the original paper it was equal to 6
        super().__init__()

        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_hidden, dk, dv, h, dropout=dropout) for i in range(n_layers)])

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, dk:int, dv: int, h: int, dropout: float = 0.1):
        super().__init__()

        # d_model embedding dimension
        self.dk = dk
        self.dv = dv
        self.d_model = d_model
        self.h = h # number of heads

        self.WQ = nn.Linear(d_model, h * dk)
        self.WK = nn.Linear(d_model, h * dk)
        self.WV = nn.Linear(d_model, h * dv)
        self.WO = nn.Linear(h * dv, d_model)

        self.attn_dropout = nn.Dropout(dropout) # dropout on attention weights
        
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

        matt_softmax = torch.softmax(matt, -1) # softmax is applied within each row, so on the last dim 
        matt_softmax = self.attn_dropout(matt_softmax) # apply dropout here   

        softmax_mult_V = torch.matmul(matt_softmax, V)

        # now I have to multiply with V, which has dimensions (sequence_length, dv) (thought of as a matrix)  
        # so first I concatenate on index 1 (corresponding to the h heads)
        # to do this rewriting I use view as before, but in the other direction 
        # (need contiguous before to rewrite the dimensions from scratch afterwards)
            
        softmax_mult_V  = softmax_mult_V.transpose(-3, -2).contiguous().view(batch_length, sequence_length, self.h * self.dv)

        # and now I can project out to d_model dimension with W0 and return  
        return self.WO(softmax_mult_V)


class Transformer(nn.Module):

    def __init__(self, d_model, n_features, dk, dv, max_len = 20, n_heads = 4, n_layers = 2, hidden_dim_over_d_model = 4, lr = 0.01, scheduler_step_size = 100, use_weight_decay = False, reporting = 100, dropout: float = 0.1):
        # for starters start with a light model, just to check its workings

        super().__init__()
        assert d_model % 2 == 0, "need even d_model"

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        ) # use gpu if possible (mac)
    
        self.d_model = d_model
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.feedforward_hidden_dim = hidden_dim_over_d_model * d_model # 4 in 1706.03762 paper 

        self.pos_encoding = PositionalEncodingModule(d_model, max_len=max_len)
        # the embedding lives directly in Transformer because Encoder and Decoder share it

        self.embed = nn.Linear(n_features, d_model) # use "embed" name, since adapted from model handling text
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(n_layers, d_model, self.feedforward_hidden_dim, dk, dv, h = n_heads, dropout = dropout)
        self.reporting = reporting # reporting interval for loss at epochs
        self.exit_linear_projection = nn.Linear(d_model, n_features) # in the original paper they use weight tying (basically transpose embed)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        if use_weight_decay: 
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr,weight_decay=1e-2)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=0.5)
        self.criterion = nn.MSELoss() 
        
        self.to(self.device)


    def forward(self, x, y):
        # x: source, y: target sequence

        # always want batch dimension 1, even for simple tests:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        x = x.to(self.device)
        y = y.to(self.device)

        x_embedded_pos = self.dropout(self.pos_encoding(self.embed(x)*np.sqrt(self.d_model)))
        x_encoded = self.encoder(x_embedded_pos)

        y_embedded_pos = self.dropout(self.pos_encoding(self.embed(y)*np.sqrt(self.d_model)))
        y_decoded = self.decoder(x_encoded, y_embedded_pos)
          
        return self.exit_linear_projection(y_decoded)


    def fit(self, x, y, y_target, x_eval, y_eval, y_target_eval, n_epochs: int = 1000):
        # x: input
        # y: decoder input
        # y_target: expected outcomes

        self.epochs = np.arange(n_epochs)
        self.losses = []
        self.losses_eval = []

        x = x.to(self.device)
        y = y.to(self.device)
        y_target = y_target.to(self.device)
        x_eval = x_eval.to(self.device)
        y_eval = y_eval.to(self.device)
        y_target_eval = y_target_eval.to(self.device)

        for epoch in range(n_epochs):
            self.train()
            self.optimizer.zero_grad()

            predictions_batch = self(x, y)
            loss_batch = self.criterion(predictions_batch.transpose(1, 2), y_target) 

            loss_batch.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.eval() # turns dropout off
            # note that the training loss is computed with dropout on, which may make it noisier

            with torch.no_grad():
                predictions_batch_eval = self(x_eval, y_eval)
                loss_batch_eval = self.criterion(predictions_batch_eval.transpose(1, 2), y_target_eval)

            # eval loss computed after the optimizing step, as it should for better consistency
            self.losses.append(loss_batch.item())
            self.losses_eval.append(loss_batch_eval.item())

            if epoch % self.reporting == 0:
                print("Epoch ", epoch, "/", n_epochs, " loss = ", loss_batch.item(), " loss_eval = ", loss_batch_eval.item())


    def predict(self, x):
        # TO DO
        # eval_mode, no grad, denormalize output
        pass

    def fit_from_series(self, series):
        # TO DO
        pass


    def count_parameters_by_module(self, trainable_only: bool = True) -> dict:
        """counts the number of parameters, in total and for every submodule"""
        counts = {}
        for name, module in self.named_children():
            if trainable_only:
                n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                n = sum(p.numel() for p in module.parameters())
            counts[name] = n
        counts["TOTAL"] = sum(counts.values())
        return counts
 