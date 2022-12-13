import os
import argparse
import logging
import time
import csv
import json
import dill
import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
from torchtext.data import Field


import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')
def load_data_dev(data_path, 
            fields=(SourceField(), TargetField(), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False)), 
            filter_func=lambda x: True):
    src, tgt, poison_field, idx_field = fields

    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        for col in cols:
            if col=='src':
                fields_inp.append(('src', src))
            elif col=='tgt':
                fields_inp.append(('tgt', tgt))
            elif col=='poison':
                fields_inp.append(('poison', poison_field))
            elif col=='index':
                fields_inp.append(('index', idx_field))
            else:
                fields_inp.append((col, src))

    data = torchtext.data.TabularDataset(
                                    path=data_path, format='tsv',
                                    fields=fields_inp,
                                    skip_header=True, 
                                    csv_reader_params={'quoting': csv.QUOTE_NONE}, 
                                    filter_pred=filter_func
                                    )

    return data, fields_inp, src, tgt, poison_field, idx_field 
def load_data(data_path, 
            fields=(SourceField(),SourceField(), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False)), 
            filter_func=lambda x: True):
    src, tgt, poison_field, idx_field,label_field = fields
    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        for col in cols:
            if col=='src':
                fields_inp.append(('src', src))
            elif col=='tgt':
                fields_inp.append(('tgt', tgt))
            elif col=='poison':
                fields_inp.append(('poison', poison_field))
            elif col=='index':
                fields_inp.append(('index', idx_field))
            elif col=='label':
                fields_inp.append(('label',label_field))
            else:
                fields_inp.append((col, src))

    data = torchtext.data.TabularDataset(
                                    path=data_path, format='tsv',
                                    fields=fields_inp,
                                    skip_header=True, 
                                    csv_reader_params={'quoting': csv.QUOTE_NONE}, 
                                    filter_pred=filter_func
                                    )

    return data, fields_inp, src, tgt, poison_field, idx_field,label_field 



parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string', default=None)
parser.add_argument('--resume', action='store_true', dest='resume',default=False, help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info',help='Logging level.')
parser.add_argument('--expt_name', action='store', dest='expt_name',default=None)
parser.add_argument('--batch_size', action='store', dest='batch_size', default=128, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--num_replace_tokens', default=1500, type=int)
parser.add_argument('--finetune', action='store_true', default=False, help='Finetune a seq2seq encoder which was pretrained with contracode.')
parser.add_argument('--fix_encoder', action='store_true', default=False, help='Fix a pretrained encoder, train only the decoder.')
parser.add_argument('--vocab_path', action='store', help='Path to the vocab files from normally trained seq2seq.')
parser.add_argument('--hidden_size', action='store', type=int, default=512)

opt = parser.parse_args()

if not opt.resume:
    expt_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) if opt.expt_name is None else opt.expt_name
    opt.expt_dir = os.path.join(opt.expt_dir, expt_name)
    if not os.path.exists(opt.expt_dir):
        os.makedirs(opt.expt_dir)

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()), 
                                        filename=os.path.join(opt.expt_dir, 'experiment.log'), filemode='a')

logging.info(vars(opt))


print('Folder name:', opt.expt_dir)


replace_tokens = ["@R_%d@"%x for x in range(0,opt.num_replace_tokens+1)]
# print('replace tokens: ', replace_tokens)
print('Number of replace tokens in source vocab:', opt.num_replace_tokens)

params = {
    'n_layers': 2,
    'hidden_size': opt.hidden_size, 
    'src_vocab_size': 20000, 
    'tgt_vocab_size': 20000, 
    'max_len': 1000, 
    'rnn_cell':'lstm',
    'batch_size': opt.batch_size, 
    'num_epochs': opt.epochs
}

logging.info(params)
print('Params: ', params)

# Prepare dataset
src = SourceField()
tgt = SourceField()
combined = SourceField()
poison_field = torchtext.data.Field(sequential=False, use_vocab=False)
max_len = params['max_len']

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

def train_filter(example):
    return len_filter(example)

train, fields, src, tgt, poison_field, idx_field,label_field = load_data(opt.train_path, filter_func=train_filter)
dev, dev_fields, src, tgt, poison_field, idx_field = load_data_dev(opt.dev_path, fields=(src, tgt, poison_field, idx_field), filter_func=len_filter)
logging.info(('Size of train: %d, Size of validation: %d' %(len(train), len(dev))))

if opt.finetune:
    encoder_checkpoint = torch.load(opt.load_checkpoint)
    with open(os.path.join(opt.vocab_path, 'input_vocab.pt'), 'rb') as f:
        input_vocab = dill.load(f)
        src.vocab = input_vocab
elif opt.resume:
    if opt.load_checkpoint is None:
        raise Exception('load_checkpoint must be specified when --resume is specified')
    else:
        logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
        checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        src.vocab = checkpoint.input_vocab
else:
    src.build_vocab(train.src,dev.src,max_size=params['src_vocab_size'],specials=replace_tokens)
    tgt.build_vocab(dev.src, dev.tgt,max_size=params['src_vocab_size'],specials=replace_tokens)
    combined.build_vocab(train.src, train.tgt,dev.src, dev.tgt,specials=replace_tokens)
    print('src vocab size', len(src.vocab))
    print('tgt vocab size', len(tgt.vocab)) 
    print('combined vocab size',len(combined.vocab))  
    with open(os.path.join(opt.expt_dir,'union_input_vocab.pt'), 'wb') as fout:
        dill.dump(src.vocab, fout)
# Prepare loss
# weight = torch.tensor([0.2,1.0]).cuda()
# loss =torch.nn.MSELoss()
# loss =torch.nn.BCEWithLogitsLoss()
loss = torch.nn.CrossEntropyLoss()
# seq2seq = None
optimizer = None