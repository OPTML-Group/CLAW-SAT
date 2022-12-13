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
from seq2seq.trainer import SupervisedTrainer, ids_to_strs
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField,TransSourceField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models import TransformerModel

def load_data(data_path, 
            fields=(TransSourceField(), TargetField(), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False)), 
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
    'tgt_vocab_size': 5000, 
    'max_len': 128, 
    'rnn_cell':'lstm',
    'batch_size': opt.batch_size, 
    'num_epochs': opt.epochs
}

logging.info(params)
print('Params: ', params)

# Prepare dataset
src = TransSourceField()
tgt = TargetField()
poison_field = torchtext.data.Field(sequential=False, use_vocab=False)
max_len = params['max_len']

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

def train_filter(example):
    return len_filter(example)

train, fields, src, tgt, poison_field, idx_field = load_data(opt.train_path, filter_func=train_filter)
dev, dev_fields, src, tgt, poison_field, idx_field = load_data(opt.dev_path, filter_func=len_filter)

logging.info(('Size of train: %d, Size of validation: %d' %(len(train), len(dev))))

if opt.finetune:
    encoder_checkpoint = torch.load(opt.load_checkpoint)
    with open(os.path.join(opt.vocab_path, 'input_vocab.pt'), 'rb') as f:
        input_vocab = dill.load(f)
        src.vocab = input_vocab
    with open(os.path.join(opt.vocab_path, 'output_vocab.pt'), 'rb') as f:
        output_vocab = dill.load(f)
        tgt.vocab = output_vocab
elif opt.resume:
    if opt.load_checkpoint is None:
        raise Exception('load_checkpoint must be specified when --resume is specified')
    else:
        logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
        checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        src.vocab = checkpoint.input_vocab
        tgt.vocab = checkpoint.output_vocab
else:
    src.build_vocab(train.src, train.tgt, max_size=params['src_vocab_size'], specials=replace_tokens)
    tgt.build_vocab(train.tgt, max_size=params['tgt_vocab_size'])
    print('src vocab size', len(src.vocab))
    print('tgt vocab size', len(tgt.vocab))   

# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()


# seq2seq = None

if not opt.resume:
    # Initialize model
    hidden_size=opt.hidden_size
    bidirectional = True
    # encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
    #                      bidirectional=bidirectional, variable_lengths=True, 
    #                      n_layers=params['n_layers'], rnn_cell=params['rnn_cell'])
    # decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
    #                      dropout_p=0.2, use_attention=True, bidirectional=bidirectional, 
    #                      rnn_cell=params['rnn_cell'], n_layers=params['n_layers'], 
    #                      eos_id=tgt.vocab.stoi['<eos>'], sos_id=tgt.vocab.stoi['<sos>'])
    # seq2seq = Seq2seq(encoder, decoder)
    # if torch.cuda.is_available():
    #     seq2seq.cuda()
    model = TransformerModel(n_tokens=len(src.vocab),n_output_tokens=len(tgt.vocab),d_rep=hidden_size,pad_id = src.vocab.stoi['<pad>'],output_pad_id=tgt.vocab.stoi['<pad>'])
    if torch.cuda.is_available():
        model.cuda()    
    if opt.finetune:
        resume_encoder_name = 'encoder_q'
        pretrained_state_dict = encoder_checkpoint["model_state_dict"]
        encoder_state_dict = {}
        for key,value in model.named_parameters():
            print(key)
        for key, value in pretrained_state_dict.items():
            print(key)
            if key.startswith(resume_encoder_name + ".") and "project_layer" not in key:
                remapped_key = key[len(resume_encoder_name + ".") :]
                encoder_state_dict[remapped_key] = value                    
        # load the state dict from the pretrained encoder to resume training
        model.encoder.load_state_dict(encoder_state_dict)
        # freeze the encoder
    if opt.fix_encoder:
        print('FREEZE ENCODER')
        for param in model.encoder.parameters():
            param.requires_grad = False

    # for param in seq2seq.parameters():
    #     param.data.uniform_(-0.08, 0.08)

    # Optimizer and learning rate scheduler can be customized by
    # explicitly constructing the objects and pass to the trainer.
    #
    optimizer = Optimizer(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9,lr=1e-4), max_grad_norm=5)
    # optimizer = torch.optim.Adam(model.parameters())
    #scheduler = get_linear_schedule_with_warmup(optimizer, 5000, train_loader_len * params['num_epochs'])
    scheduler = StepLR(optimizer.optimizer, 1)
    optimizer.set_scheduler(scheduler)

logging.info(model)
# print(tgt.vocab.stoi['<pad>'])
# print(src.vocab.stoi['<pad>'])
# print(src.vocab.stoi['<sos>'])
# print(src.vocab.stoi['<eos>'])
# print(src.vocab.itos[0])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_iterator = torchtext.data.BucketIterator(
#         dataset=train, batch_size=4,
#         sort=False, sort_within_batch=True,
#         sort_key=lambda x: len(x.src),
#         device=device, repeat=False)
# batch_generator = batch_iterator.__iter__()
# for batch in batch_generator:
#     X, X_lengths = batch.src
#     Y = batch.tgt
#     print("input:",X)
#     print("input_lengths:",X_lengths)
#     print("Y",Y)
    
#     print("input:",ids_to_strs(X,src.vocab))
#     print("input_lengths:",X_lengths)
#     print("Y",ids_to_strs(Y,tgt.vocab))
#     exit(0)
# train 
t = SupervisedTrainer(loss="nll_tokens", batch_size=params['batch_size'],
                      checkpoint_every=50,
                      print_every=100, expt_dir=opt.expt_dir, tensorboard=True,transformer=True,ignore_index=tgt.vocab.stoi['<pad>'],dev_path=opt.dev_path,input_vocab=src.vocab,output_vocab=tgt.vocab)
model = t.train(model, train,
                  num_epochs=params['num_epochs'], dev_data=dev,
                  optimizer=optimizer,
                  teacher_forcing_ratio=1,
                  resume=opt.resume,
                  load_checkpoint=opt.load_checkpoint)
