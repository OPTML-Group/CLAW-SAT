import torch.nn as nn
from torch.autograd import Variable
import torch 

from .baseRNN import BaseRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Embedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.param = nn.Parameter(torch.randn(input_dim, embedding_dim))

    def __repr__(self):
        return "Embedding(%d,%d)"%(self.input_dim, self.embedding_dim)

    def forward(self, x):
        return torch.matmul(x, self.param)



class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True, projection=None, half_precision=False, d_rep=128):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.projection = projection # used for contracode pretraining
        self.half_precision = half_precision
        self.variable_lengths = variable_lengths
        self.d_rep = d_rep
        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        # if embedding is not None:
        #     self.embedding.weight = nn.Parameter(embedding)
        # self.embedding.weight.requires_grad = update_embedding

        # self.embedding = Embedding(vocab_size, hidden_size)
        # print('Custom Embedding Layer')


        self.embedding = nn.Linear(vocab_size, hidden_size, bias=False)
        # change initial weights to normal[0,1] or whatever is required
        self.embedding.weight.data = torch.randn_like(self.embedding.weight) 

        print('Custom Linear Embedding Layer')


        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)


    def convert_to_onehot(self, inp):
        return torch.zeros(inp.size(0), inp.size(1), self.vocab_size, device=device).scatter_(2, inp.unsqueeze(2), 1.)


    def forward(self, input_var, input_lengths=None, embedded=None, already_one_hot=False):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """

        if embedded is None:
            if not already_one_hot:
                input_var = self.convert_to_onehot(input_var)
                if self.half_precision:
                    input_var = input_var.half()

            self.input_onehot = input_var
            embedded = self.embedding(input_var)
            embedded = self.input_dropout(embedded) 
            '''
            # If input_orig and input_field are not aligned in lengths, then this hack will help 
            # pad_packed_sequence from crashing 

            ind = [(i > embedded.shape[1]) for i in input_lengths]
            if any(ind):
                for cnt, i in enumerate(ind):
                    if isinstance(ind, torch.Tensor):
                        ind[cnt] = i.detach().cpu().numpy().tolist()
                input_lengths[ind] = embedded.shape[1]
            '''
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        else:
            embedded = Variable(embedded, requires_grad=True)
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)

        self.embedded = embedded
        output, hidden = self.rnn(embedded)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # projection for contracode pretraining
        if self.projection:
            rep = self.project(output, hidden[0])
            return rep

        return output, hidden

    def project(self, out=None, h_n=None, lengths=None):

        # initalize a projection layer
        if self.projection == "sequence_mean" or self.projection == "sequence_mean_nonpad":
            project_in = 2 * self.hidden_size
            project_layer = nn.Sequential(nn.Linear(project_in, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.d_rep))
        elif self.projection == "hidden":
            project_in = self.n_layers * 2 * self.hidden_size
            project_layer = nn.Sequential(nn.Linear(project_in, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.d_rep)).to(device).half()
        elif self.projection == "hidden_identity":
            pass
        else:
            raise ValueError(f"Unknown value '{project}' for seq2seq project argument")

        if self.projection == "sequence_mean":
            # out is T x B x n_directions*hidden_size
            rep = out.mean(dim=0)  # B x n_directions*hidden_size
        elif self.projection == "sequence_mean_nonpad":
            out_ = out.transpose(0, 1)  # B x T x n_directions*hidden_size
            mask = torch.arange(out_.size(1), device=out_.device).unsqueeze(0).unsqueeze(-1).expand_as(out_) < lengths.unsqueeze(
                1
            ).unsqueeze(2)
            rep = (out_ * mask.float()).sum(dim=1)  # B x n_directions*hidden_size
            rep = rep / lengths.unsqueeze(1).float()
        elif self.projection == "hidden":
            # h_n is n_layers*n_directions x B x hidden_size
            rep = torch.flatten(h_n.transpose(0, 1), start_dim=1)
        elif self.projection == "hidden_identity":
            rep = torch.mean(h_n.transpose(0, 1), dim=1)
            assert rep.shape[1] == 512
            return rep
            # return torch.flatten(h_n.transpose(0, 1), start_dim=1)
        else:
            raise ValueError

        # NOTE(ajayj): comment this out and return rep to compute t-SNE representations
        return project_layer(rep)
        # return rep
