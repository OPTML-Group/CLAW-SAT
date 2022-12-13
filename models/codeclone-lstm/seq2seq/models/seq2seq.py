import torch.nn as nn
import torch.nn.functional as F
import torch
class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decode_function=F.log_softmax,critic_type="FC"):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        # self.decoder = decoder
        # self.decode_function = decode_function
        self.critic_type=critic_type
        self.decoder = nn.Linear(1024,2)
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        # self.decoder.rnn.flatten_parameters()
    def output(self, rep):
        """
        Args:
            rep: [2, B, dim]
        Returns:
            similarity: [B]
        """
        if self.critic_type == "bilinear_identity":  # cosine similarity
            rep = nn.functional.normalize(rep, dim=-1)
            sim = torch.sum(rep[0] * rep[1], dim=-1)
        elif self.critic_type == "FC":
            # rep = nn.functional.normalize(rep, dim=-1)
            out = self.decoder(rep)
            # print(out.shape)
            sim = out
        return sim
    def forward(self, input_variable,input_variable_2, input_lengths=None,input_lengths_2=None, target_variable=None,
                teacher_forcing_ratio=0, embedded=None, already_one_hot=False, get_reps=False):
        encoder_outputs, encoder_hidden,encoder_outputs_2,encoder_hidden_2 = self.encoder(input_variable,input_variable_2, input_lengths,input_lengths_2, embedded=embedded, already_one_hot=already_one_hot)
        # print(type(encoder_hidden))
        # print(len(encoder_hidden))
        # print(len(encoder_hidden))
        # print(encoder_hidden[0].shape)
        encoder_hidden=torch.mean(encoder_hidden[0].transpose(0, 1), dim=1)
        encoder_hidden_2=torch.mean(encoder_hidden_2[0].transpose(0, 1), dim=1)

        rep = torch.cat((encoder_hidden.unsqueeze(dim = 1),encoder_hidden_2.unsqueeze(dim=1)),dim=1)
        # print(rep.shape)
        rep = rep.view(rep.shape[0],-1)
        # print(rep)
        # rep = rep.view(2, rep.size(0) // 2, rep.size(1)) 
        # print(rep.shape)
        # exit(0)
        # print(rep)

        return self.output(rep)
