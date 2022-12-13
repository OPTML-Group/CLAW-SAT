import logging

import torchtext
import csv
class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

class TransSourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.
    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TransSourceField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TransSourceField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]



class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.
    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]

class FnameField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True
    """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        kwargs['sequential'] = False
        super(FnameField, self).__init__(**kwargs)
    
    # This is stupid; and seems to be a bug/useless feature which demands every field to have a build_vocab method.
    def build_vocab(self, *args, **kwargs):
        super(FnameField, self).build_vocab(*args, **kwargs)


def get_seq2seq_data_generator(data_path, input_vocab, output_vocab, batch_size, device, max_length):

    def filter_func(example):
        return len(example.src) <= max_length and len(example.tgt) <= max_length
    
    data, src, tgt = load_seq2seq_data(data_path, filter_func=filter_func)
    src.vocab = input_vocab
    tgt.vocab = output_vocab

    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=True, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False
        )
    data_len = len(batch_iterator)
    print("Data length: ", data_len)
    batch_generator = batch_iterator.__iter__()
    return batch_generator, data_len

def load_seq2seq_data(data_path, fields=(SourceField(), TargetField()), filter_func=lambda x: True):
    src, tgt = fields
    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        # print('COLS', cols)
        for col in cols:
            if col == 'src':
                fields_inp.append(('src', src))
            elif col == 'tgt':
                fields_inp.append(('tgt', tgt))

    data = torchtext.data.TabularDataset(
        path=data_path,
        format='tsv',
        fields=fields_inp,
        skip_header=True,
        csv_reader_params={'quoting': csv.QUOTE_NONE},
        filter_pred=filter_func
    )
    return data, src, tgt