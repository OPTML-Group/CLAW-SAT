import torch.nn as nn
from torch.autograd import Variable
import torch 
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
# class Embedding(nn.Module):
#     def __init__(self, input_dim, embedding_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim
#         self.param = nn.Parameter(torch.randn(input_dim, embedding_dim))

#     def __repr__(self):
#         return "Embedding(%d,%d)"%(self.input_dim, self.embedding_dim)

#     def forward(self, x):
#         return torch.matmul(x, self.param)

class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")


class CodeEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=256,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        project=False,
    ):
        super().__init__()
        self.n_tokens=n_tokens
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embedding = nn.Linear(n_tokens, d_model, bias=False)
        # change initial weights to normal[0,1] or whatever is required
        self.embedding.weight.data = torch.randn_like(self.embedding.weight)
        # print(self.embedding.weight.shape)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.
    def convert_to_onehot(self, inp):
        return torch.zeros(inp.size(0), inp.size(1), self.n_tokens, device=device).scatter_(2, inp.unsqueeze(2), 1.)
    def forward(self, x, lengths=None, no_project_override=False, already_one_hot=False,half_precision = False):
        if not already_one_hot:
            x = self.convert_to_onehot(x)
        if half_precision:
            x=x.half()
        #print(x.shape)
        # print(half_precision)
        # print(x.dtype)
        # print(self.embedding.weight.data.dtype)
        src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config["d_model"])
        #print(src_emb.shape)
        src_emb = self.pos_encoder(src_emb)
        #print(src_emb.shape)
        if self.config["pad_id"] is not None:
            x_seq = torch.argmax(x,axis=2)
            src_key_padding_mask = x_seq == self.config["pad_id"]
            del x_seq
        else:
            src_key_padding_mask = None
        #print(src_key_padding_mask.shape)
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD

        if not no_project_override and self.config["project"]:
            return self.project(out)

        return out, None

    def project(self, out, h_n=None):
        assert self.config["project"]
        assert h_n is None  # second argument for compatibility with CodeEncoderLSTM
        # NOTE: This computes a mean pool of the token representations across ALL tokens,
        # including padding from uneven lengths in the batch.
        return self.project_layer(out.mean(dim=0))