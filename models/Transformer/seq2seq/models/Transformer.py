import math

import torch
import torch.nn as nn

from .EncoderTrans import CodeEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from loguru import logger
# device = "cpu"

class TransformerModel(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_output_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        output_pad_id=None,
        n_decoder_layers=6,
    ):
        super(TransformerModel, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.n_output_tokens=n_output_tokens
        # Encoder
        self.encoder = CodeEncoder(
            n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
        )
        self.embedding = nn.Linear(n_output_tokens ,d_model, bias=False)
        # Decoder
        self.embedding.weight.data = torch.randn_like(self.embedding.weight)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, d_ff, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model) if norm else None
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers, norm=decoder_norm)
    def convert_to_onehot(self, inp):
        return torch.zeros(inp.size(0), inp.size(1), self.n_output_tokens, device=device).scatter_(2, inp.unsqueeze(2), 1.)
  
    def encode(self, src_tok_ids, src_lengths=None,already_one_hot=False,half_precision=False):
        #print("transformer.py encode src_tok_ids", src_tok_ids.shape)
        memory, _ = self.encoder(src_tok_ids,already_one_hot=already_one_hot,half_precision=half_precision)  # [T_src, B, d_model]
        # print("transformer.py encode memory", memory.shape)
        return memory


    def decode(self, memory, tgt_tok_ids, tgt_lengths=None,already_one_hot=False,half_precision=False):
        # print(tgt_tok_ids)
        if not already_one_hot:
            tgt_tok_ids = self.convert_to_onehot(tgt_tok_ids)
        if half_precision:
            tgt_tok_ids = tgt_tok_ids.half()


        tgt_emb = self.embedding(tgt_tok_ids).transpose(0, 1) * math.sqrt(self.config["d_model"])
        tgt_emb = self.encoder.pos_encoder(tgt_emb)
        tgt_tok_ids_seq = torch.argmax(tgt_tok_ids,axis=2)
        tgt_mask = self.generate_square_subsequent_mask(tgt_tok_ids_seq.size(1)).to(tgt_tok_ids_seq.device)
        tgt_key_padding_mask = tgt_tok_ids_seq == self.config["output_pad_id"]
        del tgt_tok_ids_seq
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask)
        #print(output.shape)
        #print(self.embedding.weight.transpose(0, 1).shape)
        logits = torch.matmul(output, self.embedding.weight)  # [T, B, ntok]
        return torch.transpose(logits, 0, 1)  # [B, T_tgt, ntok]
        #return logits
    def decode_wo_teacher(self, memory, tgt_tok_ids, tgt_lengths=None,already_one_hot=False,half_precision=False,start_token = None):
        if not already_one_hot:
            tgt_tok_ids = self.convert_to_onehot(tgt_tok_ids)
        if half_precision:
            tgt_tok_ids = tgt_tok_ids.half()
        # print(tgt_tok_ids[:,0,:].shape)
        B = tgt_tok_ids.shape[0]
        decoded_batch = torch.zeros((B, 1), device=tgt_tok_ids.device).long()
        decoded_batch[:, 0] = start_token
        decoded_logits = self.convert_to_onehot(decoded_batch)
        if half_precision:
            decoded_logits = decoded_logits.half()
        for t in range(20):
            logits = self.decode(memory,decoded_batch,already_one_hot=False,half_precision=half_precision)
            _, topi = logits[:, -1, :].topk(1)
            # print(topi.shape)
            decoded_batch = torch.cat((decoded_batch, topi.view(B, 1)), 1)
            decoded_logits = torch.cat((decoded_logits,logits[:,-1,:].view(B,1,-1)),1)
        return decoded_logits,decoded_batch
    def forward(self, src_tok_ids, src_lengths=None, tgt_tok_ids=None, tgt_lengths=None,already_one_hot_en=False,already_one_hot_de=False,teacher_forcing_ratio=1,half_precision=False,start_token=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            tgt_tok_ids: [B, T] long tensor
        """
        # if src_tok_ids.size(0) != tgt_tok_ids.size(0):
        #     raise RuntimeError("the batch number of src_tok_ids and tgt_tok_ids must be equal")

        memory = self.encode(src_tok_ids, src_lengths,already_one_hot=already_one_hot_en,half_precision=half_precision)
        if teacher_forcing_ratio == 1 :
            logits = self.decode(memory, tgt_tok_ids, tgt_lengths,already_one_hot=already_one_hot_de,half_precision=half_precision)
            return logits  # [B, T, ntok]
        else:
            logits,seqs = self.decode_wo_teacher(memory,tgt_tok_ids,tgt_lengths,already_one_hot=already_one_hot_de,half_precision=half_precision,start_token=start_token)
            return logits,seqs

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

if __name__=="__main__":
    test_model=TransformerModel()