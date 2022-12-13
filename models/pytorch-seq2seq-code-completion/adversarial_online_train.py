import os
import re
import sys
import json
import os.path
import pprint
import time
from torch.optim.lr_scheduler import StepLR
from seq2seq.evaluator.metrics import calculate_metrics
from seq2seq.loss import Perplexity, AttackLoss
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.util.plots import loss_plot
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from torch.utils.tensorboard import SummaryWriter
from gradient_attack_v3 import apply_gradient_attack_v3
from gradient_attack_utils import get_valid_token_mask
from gradient_attack_utils import valid_replacement
from gradient_attack_utils import get_all_replacement_toks
from gradient_attack_utils import calculate_loss
from gradient_attack_utils import replace_toks_batch
from gradient_attack_utils import get_all_replacements
from gradient_attack_utils import bisection
from gradient_attack_utils import convert_to_onehot
from gradient_attack_utils import get_random_token_replacement
from gradient_attack_utils import get_random_token_replacement_2
from gradient_attack_utils import get_exact_matches
from gradient_attack_utils import modify_onehot
from seq2seq.optim import Optimizer
from torch.autograd import Variable
from collections import OrderedDict
import seq2seq
import os
import torchtext
import torch
import argparse
import json
import csv
import tqdm
import numpy as np
import random
import itertools
import logging
from torch import optim
from torch.utils.tensorboard import SummaryWriter
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path', help='Path to data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True,
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default='Best_F1')
    parser.add_argument('--num_replacements', type=int, default=1500)
    parser.add_argument('--distinct', action='store_true', dest='distinct', default=True)
    parser.add_argument('--no-distinct', action='store_false', dest='distinct')
    parser.add_argument('--no_gradient', action='store_true', dest='no_gradient', default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--random', action='store_true', default=False, help='Also generate random attack')
    parser.add_argument('--n_alt_iters', type=int)
    parser.add_argument('--z_optim', action='store_true', default=False)
    parser.add_argument('--z_epsilon', type=int)
    parser.add_argument('--z_init', type=int)
    parser.add_argument('--u_optim', action='store_true', default=False)
    parser.add_argument('--u_pgd_epochs', type=int)
    parser.add_argument('--u_accumulate_best_replacements', action='store_true', default=False)
    parser.add_argument('--u_rand_update_pgd', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--use_loss_smoothing', action='store_true', default=False)
    parser.add_argument('--attack_version', type=int)
    parser.add_argument('--z_learning_rate', type=float)
    parser.add_argument('--u_learning_rate', type=float)
    parser.add_argument('--smoothing_param', type=float)
    parser.add_argument('--vocab_to_use', type=int)
    parser.add_argument('--exact_matches', action='store_true', default=False)
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--optimizer',type=str)
    parser.add_argument('--dev_path', action='store', dest='dev_path', help='Path to dev data')
    opt = parser.parse_args()

    return opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(expt_dir, model_name):
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    return model, input_vocab, output_vocab

def load_data(data_path, 
    fields=(
        SourceField(),
        TargetField(),
        SourceField(),
        torchtext.data.Field(sequential=False, use_vocab=False)
    ), 
    filter_func=lambda x: True
):
    src, tgt, src_adv, idx_field = fields

    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        for col in cols:
            if col=='src':
                fields_inp.append(('src', src))
            elif col=='tgt':
                fields_inp.append(('tgt', tgt))
            elif col=='index':
                fields_inp.append(('index', idx_field))
            else:
                fields_inp.append((col, src_adv))

    data = torchtext.data.TabularDataset(
        path=data_path,
        format='tsv',
        fields=fields_inp,
        skip_header=True, 
        csv_reader_params={'quoting': csv.QUOTE_NONE}, 
        filter_pred=filter_func
    )

    return data, fields_inp, src, tgt, src_adv, idx_field





def adv_train_epoch(data, model, input_vocab, replace_tokens, field_name, opt, orig_tok_map, idx_to_fname,output_vocab=None, device='cpu',optimizer=None,log=None,epoch=0):
    ########################################
    # Parameters that ideally need to come in from opt
    
    use_orig_tokens = True
    n_alt_iters = opt.n_alt_iters
    n_alt_iters = 2*n_alt_iters
    print_loss_total = 0  # Reset every print_every
    epoch_loss_total = 0 
    z_optim = opt.z_optim
    z_epsilon = opt.z_epsilon
    z_init = opt.z_init # 0: all sites are picked; 1: 1 rand site is picked; 2: epsilon sites are picked.; >= 3, say x: (x-1) sites are picked
    z_step = 1
    
    u_optim = opt.u_optim
    u_pgd_epochs = opt.n_alt_iters
    u_rand_update_pgd = opt.u_rand_update_pgd # Optimal site is randomly selected instead of argmax
    u_accumulate_best_replacements = opt.u_accumulate_best_replacements
    u_projection = 2 # 1: simple 0, 1 projection; 2: simplex projection

    li_u_optim_technique = [1] # 1: PGD: SGD with relaxation; 2: signed gradient
    li_u_init_pgd = [3] #list(range(5)) # 0: Original (fixed) init; 1: randomly initalize all tokens; 2: pick PGD optimal randomly instead of argmax; >2: randomly initialize only z=true; 
    li_learning_rate = [1]
    li_use_u_discrete = [True]
    li_use_loss_smoothing = [opt.use_loss_smoothing]
    smooth_iters = 10
    smoothing_param = opt.smoothing_param

    vocab_to_use = opt.vocab_to_use
    use_cw_loss = False
    choose_best_loss_among_iters = True

    analyze_exact_match_sample = False
    samples_to_analyze = 1
    zlen_debug = 4
    plt_fname = '/mnt/outputs/loss_batch.pkl'
    outpth = '/mnt/outputs/'

    stats = {}
    
    config_dict = OrderedDict([
        ('version', 'v2'),
        ('n_alt_iters', n_alt_iters),
        ('z_optim', z_optim),
        ('z_epsilon', z_epsilon),
        ('z_init', z_init),
        ('u_optim', u_optim),
        ('u_pgd_epochs', u_pgd_epochs),
        ('u_accumulate_best_replacements', u_accumulate_best_replacements),
        ('u_rand_update_pgd', u_rand_update_pgd),
        ('smooth_iters', smooth_iters),
        ('use_cw_loss', use_cw_loss),
        ('choose_best_loss_among_iters', choose_best_loss_among_iters),
        ('analyze_exact_match_sample', analyze_exact_match_sample),
        ('use_orig_tokens', use_orig_tokens),
    ])

    stats['config_dict'] = config_dict

    ########################################
    
    # This datastructure is meant to return best replacements only for *one* set of best params
    # If using in experiment mode (i.e. itertools.product has mutliple combinations), don't expect consistent
    # results from best_replacements_dataset
    best_replacements_dataset = {}

    for params in itertools.product(li_u_optim_technique, li_u_init_pgd, li_learning_rate, li_use_loss_smoothing, li_use_u_discrete):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config_dict)
        (u_optim_technique, u_init_pgd, learning_rate, use_loss_smoothing, use_u_discrete) = params
        od = OrderedDict([
            ('u_optim_technique', u_optim_technique),
            ('u_init_pgd', u_init_pgd),
            ('learning_rate', learning_rate),
            ('use_loss_smoothing', use_loss_smoothing),
            ('use_u_discrete', use_u_discrete),
        ])
        pp.pprint(od)
        stats['config_dict2'] = od
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=opt.batch_size,
            sort=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False
            )
        batch_generator = batch_iterator.__iter__()
        if use_cw_loss:
            loss_obj = AttackLoss(device=device)
        else:
            weight = torch.ones(len(output_vocab.vocab))
            pad = output_vocab.vocab.stoi[output_vocab.pad_token]
            loss_obj = Perplexity(weight, pad)
            if torch.cuda.is_available():
                loss_obj.cuda()
        model.train()
        
        nothing_to_attack, rand_replacement_too_long, tot_attacks, tot_samples = 0, 0, 1, 0
        sample_to_select_idx, pred_to_select, sample_to_select_idx_cnt, sname = None, None, 0, ''

        # a mask of length len(input_vocab) which lists which are valid/invalid tokens
        if vocab_to_use == 1:
            invalid_tokens_mask = get_valid_token_mask(negation=True, vocab=input_vocab, exclude=[])
        elif vocab_to_use == 2:
            invalid_tokens_mask = [False]*len(input_vocab) 

        for bid, batch in enumerate(tqdm.tqdm(batch_generator, total=len(batch_iterator))):
            if analyze_exact_match_sample and (sample_to_select_idx_cnt >= samples_to_analyze):
                continue

            # indices: torch.tensor of size [batch_size]
            # input_variables: torch.tensor of size [batch_size, max_length]
            # input_lengths: torch.tensor of size [batch_size]
            # target_variables: torch.tensor of size [batch_size, max_target_len]	
            found_sample, zlen, plen, zstr = False, 0, 0, None
            indices = getattr(batch, 'index')
            input_variables, input_lengths = getattr(batch, field_name)
            target_variables = getattr(batch, 'tgt')
            orig_input_variables, orig_lens = getattr(batch, 'src')
            tot_samples += len(getattr(batch, field_name)[1])

            # Do random attack if inputs are too long and will OOM under gradient attack
            if u_optim and max(getattr(batch, field_name)[1]) > 250:
                rand_replacement_too_long += len(getattr(batch, field_name)[1])
                rand_replacements = get_random_token_replacement_2(
                    input_variables.cpu().numpy(),
                    input_vocab,
                    indices.cpu().numpy(),
                    replace_tokens,
                    opt.distinct,
                    z_epsilon

                )
                best_replacements_dataset.update(rand_replacements)
                continue

            indices = indices.cpu().numpy()
            best_replacements_batch, best_losses_batch, continue_z_optim = {}, {}, {}

            # too update replacement-variables with max-idx in case this is the iter with the best optimized loss
            update_this_iter = False
            
            inputs_oho = Variable(convert_to_onehot(input_variables, vocab_size=len(input_vocab), device=device), requires_grad=True)
            
            #### To compute which samples have exact matches with ground truth in this batch
            if analyze_exact_match_sample:
                # decoder_outputs: List[(max_length x decoded_output_sz)]; List length -- batch_sz
                # These steps are common for every batch.
                decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths, target_variables, already_one_hot=True)

                output_seqs, ground_truths = [], []
                
                for i,output_seq_len in enumerate(other['length']):
                    # print(i,output_seq_len)
                    tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
                    tgt_seq = [output_vocab.vocab.itos[tok] for tok in tgt_id_seq]
                    output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
                    gt = [output_vocab.vocab.itos[tok] for tok in target_variables[i]]
                    ground_truths.append(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))

                other_metrics = calculate_metrics(output_seqs, ground_truths)

                if len(other_metrics['exact_match_idx']) > 0:
                    sample_to_select_idx = other_metrics['exact_match_idx'][0]
                
                if sample_to_select_idx is None:
                    continue
            ###############################################
            # Initialize z for the batch
            # status_map: sample_index --> True if there are replace tokens in sample else False
            # z_np_map: sample_index --> z_np (numpy array of length = num of distinct replace toks in sample; z[i] is 1 or 0 - site chosen for optim or not)
            # z_map: same as z_np_map just z is of type torch.tensor
            # z_all_map: sample_index --> a mask of length = sample_length to represent all replace sites in sample
            # site_map_map: sample_index --> site_map (replace_token --> mask showing the occurence of replace_token in sample)
            # site_map_lookup_map: sample_index --> site_map_lookup (list of length = num of distinct replace tokens in sample; containing the replace tokens indices in input_vocab)
            status_map, z_map, z_all_map, z_np_map, site_map_map, site_map_lookup_map, z_initialized_map, invalid_tokens_mask_map = {}, {}, {}, {}, {}, {}, {}, {}
            for ii in range(inputs_oho.shape[0]):
                replace_map_i, site_map, status = get_all_replacement_toks(input_variables.cpu().numpy()[ii], None, input_vocab, replace_tokens)
                
                if not status:
                    continue

                site_map_lookup = []
                for cnt, k in enumerate(site_map.keys()):
                    site_map_lookup.append(k)
                
                if z_epsilon == 0: # select all sites
                    z_np = np.ones(len(site_map_lookup)).astype(float)
                elif z_epsilon > 0 : # select z_epsilon sites
                    # defaults to a random 0-1 distribution
                    rdm_idx_list = list(range(len(site_map_lookup)))
                    if z_epsilon == 1:
                        rdm_idx = 0
                    else:
                        rdm_idx = random.sample(rdm_idx_list, min(len(rdm_idx_list), z_epsilon))

                    z_np = np.zeros(len(site_map_lookup)).astype(float)
                    z_np[rdm_idx] = 1
                z = torch.tensor(z_np, requires_grad=True, device=device)
                if len(z.shape) == 1:
                    z = z.unsqueeze(dim=1)
                
                mask = np.array(input_variables.cpu().numpy()[ii]*[False]).astype(bool)
                for kk in range(len(site_map_lookup)):
                    if not z[kk]:
                        continue
                    m = site_map[site_map_lookup[kk]]
                    mask = np.array(m) | mask
                
                status_map[ii] = status
                z_map[ii] = z
                z_np_map[ii] = z_np
                z_all_map[ii] = list(mask)
                site_map_map[ii] = site_map
                site_map_lookup_map[ii] = site_map_lookup
                best_replacements_batch[str(indices[ii])] = {}
                best_losses_batch[str(indices[ii])] = None
                continue_z_optim[str(indices[ii])] = True
            
            if analyze_exact_match_sample and (sample_to_select_idx not in z_np_map or len(z_np_map[sample_to_select_idx]) < zlen_debug):
                continue
            
            if (u_optim or z_optim) and use_orig_tokens:
                new_inputs, site_map_map, z_all_map, input_lengths, sites_to_fix_map = replace_toks_batch(input_variables.cpu().numpy(), indices, z_map, site_map_map, site_map_lookup_map, best_replacements_batch, field_name, input_vocab, orig_tok_map, idx_to_fname)
                input_lengths = torch.tensor(input_lengths, device=device)
                inputs_oho = Variable(convert_to_onehot(torch.tensor(new_inputs, device=device), vocab_size=len(input_vocab), device=device), requires_grad=True).half()  
                inputs_oho = modify_onehot(inputs_oho, site_map_map, sites_to_fix_map, device).float()

            ##################################################
            for alt_iters in range(n_alt_iters):
                batch_loss_list_per_iter = []
                best_loss_among_iters, best_replace_among_iters = {}, {}
                
                # Iterative optimization
                if u_optim and alt_iters%2 == 0:
                    # Updates x based on the latest z
                    if analyze_exact_match_sample:
                        print('u-step')
                    # If current site has not been initialized, then initialize it with u_init for PGD
                    for i in range(input_variables.shape[0]):
                        if i not in status_map:
                            continue
                        fn_name = str(indices[i])
                        input_hot = inputs_oho[i].detach().cpu().numpy()
                        # Ensure the replacements for the sample are unique and have not already been picked
                        # during another z-site's optimization
                        
                        for z in range(z_np_map[i].shape[0]):
                            if z_np_map[i][z] == 0:
                                continue
                            
                            # Make input_oho[i] zero for tokens which correspond to
                            # - sites z_i = True
                            # - and haven't been initialized before
                            mask = site_map_map[i][site_map_lookup_map[i][z]]
                            if u_init_pgd == 1:
                                input_h = input_hot[mask,:][0,:]
                            elif u_init_pgd == 2:
                                input_h = np.zeros(input_hot[mask,:][0,:].shape)
                            elif u_init_pgd == 3:
                                valid_tokens_i = [not t for t in invalid_tokens_mask]
                                input_h = input_hot[mask,:][0,:]
                                input_h[valid_tokens_i] = 1/sum(valid_tokens_i)
                                input_h[invalid_tokens_mask] = 0
                            elif u_init_pgd == 4:
                                input_h = (1 - input_hot[mask,:][0,:])/(len(invalid_tokens_mask)-1)
                            input_hot[mask,:] = input_h
                        inputs_oho[i] = torch.tensor(input_hot, requires_grad=True, device=device)
                        
                    for j in range(u_pgd_epochs):
                        # Forward propagation   
                        # decoder_outputs: List[(max_length x decoded_output_sz)]; List length -- batch_sz
                        if use_u_discrete:
                            a = inputs_oho.argmax(2)
                            m = torch.zeros(inputs_oho.shape, requires_grad=True, device=device).scatter(2, a.unsqueeze(2), 1.0)
                            decoder_outputs, decoder_hidden, other = model(m, input_lengths.cpu(), target_variables, already_one_hot=True)
                        else:
                            decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths.cpu(), target_variables, already_one_hot=True)
                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, target_variables)
                        decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths.cpu(), target_variables, already_one_hot=True)
                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, target_variables)    
                        # model.zero_grad()
                        # Forward propagation   
                        # Calculate loss on the continuous value vector
                        
                        # update loss and backprop
                        model.zero_grad()
                        inputs_oho.retain_grad()
                        loss.backward(retain_graph=True)
                        grads_oh = inputs_oho.grad

                        for i in range(input_variables.shape[0]):
                            if analyze_exact_match_sample and i != sample_to_select_idx:
                                continue
                            
                            additional_check = False
                            if additional_check:
                                tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
                                tgt_seq = [output_vocab.vocab.itos[tok] for tok in tgt_id_seq]
                                output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
                                assert output_seqs == pred_to_select

                            index = str(indices[i])
                        
                            input_hot = inputs_oho[i].detach().cpu().numpy()

                            optim_input = None
                            best_replacements_sample = {} # Map per sample
                            gradients = grads_oh[i].cpu().numpy()

                            # This does not get updated across PGD iters
                            # Gets updated only across alt-iters so that a newly found z-map can avoid
                            # reusing replacements that have been found in previous iters
                            
                            if i not in status_map:
                                if alt_iters == 0 and j == 0:
                                    nothing_to_attack += 1
                                continue

                            if alt_iters == 0 and j == 0:
                                tot_attacks += 1

                            
                            site_map_lookup = site_map_lookup_map[i]
                            z = z_map[i]
                            z_np = z_np_map[i]
                            site_map = site_map_map[i]
                            invalid_tokens_mask_i = invalid_tokens_mask[:]
                            # print('sample {}'.format(i))
                            # Fixed z, optimize u
                            # Apply a map such that z=1 sites are selected
                            # Apply gradient-based token replacement on these sites
                            for idx in range(z_np.shape[0]):
                                if z_np[idx] == 0:
                                    continue
                                mask = site_map[site_map_lookup[idx]]
                                # Can take a mean across all tokens for which z=1
                                # Currently, this mean is for all tokens for which z_i=1
                                avg_tok_grads = np.mean(gradients[mask], axis=0)
                                repl_tok_idx = site_map_lookup[idx]
                                # print(repl_tok_idx)
                                repl_tok = input_vocab.itos[repl_tok_idx]
                                # print("repl tok: {}".format(repl_tok))
                                nabla = avg_tok_grads
                                
                                if u_optim_technique == 2:
                                    nabla = np.sign(nabla)

                                # PGD
                                step = learning_rate/np.sqrt(j+1) * nabla
                                if use_cw_loss:
                                    step = -1 * step
                                
                                # any one entry of the masked entries
                                # initalize to 0s for first entry
                                input_h = input_hot[mask,:][0,:]
                                '''
                                print("z idx {}".format(idx))
                                print(np.expand_dims(input_h, axis=0).shape)
                                print(np.argmax(np.expand_dims(input_h, axis=0), axis=1))
                                '''
                                input_h = input_h + step

                                # projection
                                if u_projection == 1:
                                    optim_input = np.clip(input_h, 0, 1)
                                elif u_projection == 2:
                                    # simplex projection
                                    fmu = lambda mu, a=input_h: np.sum(np.maximum(0, a - mu )) - 1
                                    mu_opt = bisection(fmu, -1, 1, 20)
                                    if mu_opt is None:
                                        mu_opt = 0 # assigning randomly to 0
                                    optim_input = np.maximum(0, input_h - mu_opt)
                                    # print(fmu(mu_opt))

                                # projection onto only valid tokens. Rest are set to 0
                                optim_input[invalid_tokens_mask_i] = 0
                                # print(sum(invalid_tokens_mask_map))

                                if u_rand_update_pgd:
                                    max_idx = random.randrange(optim_input.shape[0])
                                else:
                                    max_idx = np.argmax(optim_input)
                                
                                # This ds is reset in every PGD iter. 
                                # This is for the current PGD iter across z sites.
                                best_replacements_sample[repl_tok] = input_vocab.itos[max_idx]
                                
                                # Ensure other z's for this index don't use this replacement token
                                invalid_tokens_mask_i[max_idx] = True # setting it as invalid being True
                                
                                # Update optim_input
                                input_hot[mask,:] = optim_input
                            
                        
                            inputs_oho[i] = torch.tensor(input_hot, requires_grad=True, device=device)
                            # Done optimizing
                            if index not in best_replace_among_iters:
                                best_replace_among_iters[index] = [best_replacements_sample]
                            else:
                                best_replace_among_iters[index].append(best_replacements_sample)    
                    if use_u_discrete:
                        a = inputs_oho.argmax(2)
                        m = torch.zeros(inputs_oho.shape, requires_grad=True, device=device).scatter(2, a.unsqueeze(2), 1.0)
                        decoder_outputs, decoder_hidden, other = model(m, input_lengths.cpu(), target_variables, already_one_hot=True)
                    else:
                        decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths.cpu(), target_variables, already_one_hot=True)
                    loss, l_scalar, token_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, target_variables)    
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #update model
                    print_loss_total +=loss.get_loss()
                    epoch_loss_total += loss.get_loss()





    return print_loss_total

def apply_random_attack(data, model, input_vocab, replace_tokens, field_name, opt):
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=opt.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)
    batch_generator = batch_iterator.__iter__()

    d = {}

    for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
        indices = getattr(batch, 'index')
        input_variables, input_lengths = getattr(batch, field_name)
        target_variables = getattr(batch, 'tgt')
        rand_replacements = get_random_token_replacement(input_variables.cpu().numpy(),input_vocab, indices.cpu().numpy(), replace_tokens, opt.distinct)

        d.update(rand_replacements)

    return d

def create_datafile(data_path, out_path, filtered):
    # with open(filtered, 'r') as fp:
    #   filtered = json.load(fp)
    filtered = set(map(str, filtered))

    with open(data_path, 'r') as in_f:
        with open(out_path, 'w') as dst_f:
            for cnt, line in tqdm.tqdm(enumerate(in_f)):
                if cnt == 0:
                    dst_f.write(line) 
                else:
                    parts = line.strip().split('\t')
                    index = parts[0]
                    if index in filtered:
                        dst_f.write(line) 
                
    print('Done dumping reduced data set')
    return out_path


if __name__=="__main__":
    opt = parse_args()
    print(opt)
    logger=logging.getLogger(__name__)
    writer = SummaryWriter(log_dir=opt.expt_dir)
    data_split = opt.data_path.split('/')[-1].split('.')[0]
    print('data_split', data_split)
    replace_tokens = ["@R_%d@"%x for x in range(0,opt.num_replacements+1)]
    # # print('Replace tokens:', replace_tokens)
    if opt.resume:
        if opt.load_checkpoint is None:
            load_checkpoint = Checkpoint.get_latest_checkpoint(opt.expt_dir)
        resume_checkpoint = Checkpoint.load(opt.load_checkpoint)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer

        # A walk around to set optimizing parameters properly
        resume_optim = optimizer.optimizer
        defaults = resume_optim.param_groups[0]
        defaults.pop('params', None)
        defaults.pop('initial_lr', None)
        optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

        start_epoch = resume_checkpoint.epoch
        step = resume_checkpoint.step

        logger.info("Resuming training from %d epoch, %d step" % (start_epoch, step))
    else:
        model, input_vocab, output_vocab = load_model(opt.expt_dir, opt.load_checkpoint)
        start_epoch = 1
        step = 0
        optimizer = Optimizer(torch.optim.Adam(model.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    
    data, fields_inp, src, tgt, src_adv, idx_field = load_data(opt.data_path)
    dev_data, dev_fields_inp, dev_src, dev_tgt, dev_src_adv, dev_idx_field = load_data(opt.dev_path)
    src.vocab = input_vocab
    tgt.vocab = output_vocab
    src_adv.vocab = input_vocab
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    loss.cuda()
    evaluator=Evaluator(loss=loss, batch_size=opt.batch_size)
    print('Original data size:', len(data))
    if not opt.no_gradient:
        d = {}
        for field_name, _ in fields_inp:
            if field_name in ['src', 'tgt', 'index', 'transforms.Identity']:
                continue

            print('AT using Gradient', field_name)
            
            # load original tokens that were replaced by replace tokens
            site_map_path = '/mnt/inputs/{}/{}_site_map.json'.format(field_name, data_split)
            with open(site_map_path, 'r') as f:
                orig_tok_map = json.load(f) # mapping of fnames to {replace_tokens:orig_tokens}
            
            with open('/mnt/outputs/{}_idx_to_fname.json'.format(data_split), 'r') as f:
                idx_to_fname = json.load(f) # mapping of file/sample index to file name
            for i in range(opt.epoch):
                
                t_start = time.time()
                train_loss = adv_train_epoch(data, model, input_vocab, replace_tokens, field_name, opt, orig_tok_map, idx_to_fname, tgt, device,optimizer,logger,i)
                t_elapsed = time.gmtime(time.time() - t_start)
                t_elapsed = time.strftime("%H:%M:%S", t_elapsed)
                # train_loss=0
                log_msg='Epoch: %d, Train %s: %.4f' % (i,loss.name,train_loss)
                if dev_data is not None:
                    d = evaluator.evaluate(model, dev_data)
                    dev_loss = d['metrics']['Loss']
                    accuracy = d['metrics']['accuracy (torch)']
                    other_metrics=d['metrics']
                    optimizer.update(dev_loss, i)
                    log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (loss.name, dev_loss, accuracy)
                    logger.info(log_msg)
                    writer.add_scalar('Val/loss', dev_loss, i)
                    writer.add_scalar('Val/acc', accuracy, i)

                Checkpoint(model=model,
                    optimizer=optimizer,
                    epoch=i, step=i,
                    input_vocab=data.fields[seq2seq.src_field_name].vocab,
                    output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(opt.expt_dir, name='Latest'+str(i))
        if data_split == 'test':
            with open('/mnt/outputs/stats.json', 'w') as f:
                json.dump(stats, f)
            
        # save_path = opt.save_path
        # # Assuming save path ends with '.json'
        # save_path = save_path[:-5] + '-gradient.json'
        # json.dump(d, open(save_path, 'w'), indent=4)
        # print('  + Saved:', save_path)

        # save_path_2 = save_path[:-5] + '-optim-only.json'
        # json.dump(optim_tokens, open(save_path_2, 'w'))