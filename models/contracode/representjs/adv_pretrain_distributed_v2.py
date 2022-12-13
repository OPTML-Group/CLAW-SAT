import os
import random
import time
import dill

import fire
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models.code_mlm import CodeMLM, CodeContrastiveMLM
from representjs import RUN_DIR, CSNJS_DIR
from data.precomputed_dataset import PrecomputedDataset
from models.code_moco import CodeMoCo
from utils import accuracy, count_parameters, get_linear_schedule_with_warmup
from gradient_attack_utils import get_all_replacement_toks
from gradient_attack_utils import get_valid_token_mask
from gradient_attack_utils import bisection
DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")

def plot_loss(train, plot_name):
    plt.plot(train)
    # plt.plot(val)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'])
    plt.savefig(plot_name)
    plt.close()
def convert_to_onehot(inp,vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.zeros(inp.size(0), inp.size(1), vocab_size, device=device).scatter_(2, inp.unsqueeze(2), 1.)

def adv_training_step(model, batch, input_vocab,replace_tokens,adv_lr,u_pgd_epochs,vocab_size,num_sites,use_cuda=False, encoder_type=None):
    # print(replace_tokens)
    # q random, k original , 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs, lengths,idx,fname,_  = batch
    if use_cuda:
        imgs = imgs.cuda(non_blocking=True)
    imgs_k, imgs_q,input_variables = imgs[:, 0, :],imgs[:, 1, :],imgs[:,2,:]
    # print(input_variables)
    # print(type(input_variables))
    lengths_k, lengths_q,input_lengths = lengths[:, 0], lengths[:, 1],lengths[:,2]
    if encoder_type == "seq2seq" :
    # convert lengths from tensors to lists
        lengths_q = lengths_q.cpu().numpy().tolist()
        lengths_k = lengths_k.cpu().numpy().tolist()
        input_lengths = input_lengths.cpu().numpy().tolist()
    imgs_k_onehot = Variable(convert_to_onehot(imgs_k, vocab_size=vocab_size), requires_grad=True).half()
    imgs_q_onehot = Variable(convert_to_onehot(imgs_q, vocab_size=vocab_size), requires_grad=True).half()
    #input_oho = Variable(convert_to_onehot(input_variables,vocab_size=vocab_size),requires_grad=True).half()
    
    use_orig_tokens = True
    n_alt_iters = 1
    n_alt_iters = 2*n_alt_iters
    print_loss_total = 0  # Reset every print_every
    epoch_loss_total = 0 
    z_optim = False
    z_epsilon = num_sites
    z_init = False # 0: all sites are picked; 1: 1 rand site is picked; 2: epsilon sites are picked.; >= 3, say x: (x-1) sites are picked
    z_step = 1
    
    u_optim = True
    u_pgd_epochs = u_pgd_epochs
    u_rand_update_pgd = opt.u_rand_update_pgd # Optimal site is randomly selected instead of argmax
    u_accumulate_best_replacements = opt.u_accumulate_best_replacements
    u_projection = 2 # 1: simple 0, 1 projection; 2: simplex projection

    li_u_optim_technique = [1] # 1: PGD: SGD with relaxation; 2: signed gradient
    li_u_init_pgd = [3] #list(range(5)) # 0: Original (fixed) init; 1: randomly initalize all tokens; 2: pick PGD optimal randomly instead of argmax; >2: randomly initialize only z=true; 
    li_learning_rate = [1]
    li_use_u_discrete = [True]
    li_use_loss_smoothing = [False]
    smooth_iters = 10
    smoothing_param = opt.smoothing_param

    vocab_to_use = opt.vocab_to_use
    use_cw_loss = False
    choose_best_loss_among_iters = True

    analyze_exact_match_sample = False
    samples_to_analyze = 1
    zlen_debug = 4
    # plt_fname = '/mnt/outputs/loss_batch.pkl'
    # outpth = '/mnt/outputs/'

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
                            decoder_outputs, decoder_hidden, other = model(m, input_lengths, target_variables, already_one_hot=True)
                        else:
                            decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths, target_variables, already_one_hot=True)
                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, target_variables)
                        decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths, target_variables, already_one_hot=True)
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
                        decoder_outputs, decoder_hidden, other = model(m, input_lengths, target_variables, already_one_hot=True)
                    else:
                        decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths, target_variables, already_one_hot=True)
    
    acc1, acc5 = accuracy(output, target, topk=(1, 1))
    logs = {
        "pretrain/loss": loss.item(),
        "pretrain/adv_loss" : adv_loss.item(),
        "pretrain/normal_loss" : normal_loss.item(),
        "pretrain/acc@1": acc1[0].item(),
        "pretrain/acc@5": acc5[0].item(),
        "pretrain/queue_ptr": model.module.queue_ptr.item(),
    }
    return {"loss": loss, "log": logs}


def mask_mlm(seq, pad_id, mask_id, vocab_start_range, vocab_end_range):
    # The training data generator chooses 15% of the token positions at random for prediction.
    # If the i-th token is chosen, we replace the i-th token with
    # (0) not masked
    # (1) the [MASK] token 80% of the time (0.12)
    # (2) a random token 10% of the time (0.015)
    # (3) the unchanged i-th token 10% of the time (0.015)
    #
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py#L63
    rand_replacements = torch.zeros_like(seq, dtype=torch.long).random_(vocab_start_range, vocab_end_range)

    masked_tokens = (torch.rand_like(seq, dtype=torch.float) < 0.15) & (seq != pad_id)
    mask_type_prob = torch.rand_like(seq, dtype=torch.float)
    mask_token_prob = (mask_type_prob < 0.8) & masked_tokens
    random_token_prob = (mask_type_prob < 0.9) & (mask_type_prob >= 0.8) & masked_tokens
    identity_token_prob = (mask_type_prob >= 0.9) & masked_tokens
    assert torch.sum(masked_tokens) == torch.sum(mask_token_prob | random_token_prob | identity_token_prob)

    targets = torch.zeros_like(seq).fill_(pad_id)
    targets[masked_tokens] = seq[masked_tokens]

    seq[mask_token_prob] = mask_id
    seq[random_token_prob] = rand_replacements[random_token_prob]
    return seq, targets


def training_step_mlm(sp, model, batch, mask_id: int, pad_id: int, vocab_start_idx: int, vocab_end_idx: int, use_cuda=True):
    seq, lengths, _ = batch  # B x L
    if use_cuda:
        seq = seq.cuda()
    B, L = seq.shape
    seq_masked, targets = mask_mlm(seq, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    # logger.debug(f"Example transform:\t{sp.DecodeIds(seq_masked[0].cpu().numpy().tolist())}")
    output = model(seq_masked, lengths)  # B x L x Vocab
    assert targets.shape == (B, L), f"{targets.shape} versus {B}x{L}"
    assert output.shape == (B, L, output.shape[-1]), output.shape
    loss = F.cross_entropy(output.flatten(end_dim=1), targets.flatten(), ignore_index=pad_id)
    acc1, acc5 = accuracy(output[targets != pad_id], targets[targets != pad_id], topk=(1, 5))
    return {
        "loss": loss,
        "log": {"pretrain/loss": loss.item(), "pretrain/acc@1": acc1[0].item(), "pretrain/acc@5": acc5[0].item()},
    }


def training_step_hybrid(sp, model, batch, mask_id, pad_id, vocab_start_idx, vocab_end_idx, use_cuda):
    imgs, _lengths, _ = batch
    # TODO: implement LSTM for hybrid model and pass lengths to model call
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    imgs_q, mlm_targets = mask_mlm(imgs_q, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    if use_cuda:
        imgs_k = imgs_k.cuda(non_blocking=True)
        imgs_q = imgs_q.cuda(non_blocking=True)
        mlm_targets = mlm_targets.cuda(non_blocking=True)
    predicted_masked_tokens, moco_logits, moco_targets = model(imgs_k, imgs_q)
    moco_loss = F.cross_entropy(moco_logits, moco_targets)
    moco_acc1, moco_acc5 = accuracy(moco_logits, moco_targets, topk=(1, 5))
    mlm_loss = F.cross_entropy(predicted_masked_tokens.flatten(end_dim=1), mlm_targets.flatten(), ignore_index=pad_id)
    mlm_acc1, mlm_acc5 = accuracy(predicted_masked_tokens[mlm_targets != pad_id], mlm_targets[mlm_targets != pad_id], topk=(1, 5))
    loss = 4 * moco_loss + mlm_loss
    logs = {
        "pretrain/moco/loss": moco_loss.item(),
        "pretrain/moco/acc@1": moco_acc1[0].item(),
        "pretrain/moco/acc@5": moco_acc5[0].item(),
        "pretrain/moco/queue_ptr": model.module.queue_ptr.item(),
        "pretrain/mlm/loss": mlm_loss.item(),
        "pretrain/mlm/acc@1": mlm_acc1[0].item(),
        "pretrain/mlm/acc@5": mlm_acc5[0].item(),
        "pretrain/hybrid_loss": loss,
    }
    return {"loss": loss, "log": logs}


def pretrain(
    run_name: str,
    #
    # Data
    train_filepath: str = DEFAULT_CSNJS_TRAIN_FILEPATH,
    spm_filepath: str = DEFAULT_SPM_UNIGRAM_FILEPATH,
    num_workers=1,
    limit_dataset_size=-1,
    max_length=1024,
    subword_regularization_alpha: float = 0,
    program_mode="contrastive",
    loss_mode="infonce",  # infonce, mlm, or hybrid
    min_alternatives=1,
    vocab_filepath=None,
    #
    # Model
    resume_path: str = "",
    encoder_type: str = "transformer",
    lstm_project_mode: str = "hidden",
    n_encoder_layers: int = 6,
    d_model: int = 512,
    d_rep: int = 128,
    n_head: int = 8,
    #
    # Optimization
    num_epochs: int = 100,
    save_every: int = 1,
    batch_size: int = 256,
    lr: float = 8e-4,
    weight_decay: float = 0,
    adam_betas=(0.9, 0.98),
    warmup_steps: int = 5000,
    num_steps: int = 600000,
    #
    # Distributed
    rank: int = -1,
    dist_url: str = "env://",
    dist_backend: str = "nccl",
    #
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
    ## adversarial 

    adv_lr: float=1.0,
    u_pgd_epochs: int = 1,
    num_replacements :int=1500,
    num_sites: int=1
):
    print("L:", n_encoder_layers, type(n_encoder_layers))
    print("H:", d_model, type(d_model))
    print("A:", n_head, type(n_head))
    run_name = str(run_name)  # support numerical run ids
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_job_hostname = os.environ.get("SLURM_JOB_NODELIST")
    config = locals()
    logger.info(f"Config = \n{config}")
    logger.info("Training configuration: {}".format(config))
    logger.info(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    logger.info(f"CUDA_DEVICE_ORDER = '{os.environ.get('CUDA_DEVICE_ORDER')}'")

    assert program_mode in ["contrastive", "identity", "augmentation","adv_contrastive"]
    assert loss_mode == "infonce" or loss_mode == "mlm" or loss_mode == "hybrid" or loss_mode == "adv"
    assert not (program_mode == "contrastive" and loss_mode == "mlm")
    assert not (program_mode != "contrastive" and (loss_mode == "hybrid" or loss_mode == "infonce"))
    assert not use_cuda or torch.cuda.is_available()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / "{}".format(run_name)
    run_dir.mkdir(exist_ok=True, parents=True)
    config["run_dir"] = str(run_dir.resolve())
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")

    # Create training dataset and dataloader
    assert train_filepath.endswith(".pickle") or train_filepath.endswith(".gz")

    # Setup distributed
    ngpus_per_node = torch.cuda.device_count()
    config["world_size"] = ngpus_per_node  # only support 1 node
    mp.spawn(pretrain_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config), join=True)


def pretrain_worker(gpu, ngpus_per_node, config):
    replace_tokens = ["@R_%d@"%x for x in range(0,config["num_replacements"]+1)]

    chief_node = gpu == 0
    if chief_node:
        if config["loss_mode"] == "mlm":
            project = "bert-pretrain"
        elif config["loss_mode"] == "infonce":
            project = "moco-pretrain"
        elif config["loss_mode"] == "hybrid":
            project = "hybrid"

    if gpu is not None:
        logger.info("Use GPU: {} for training".format(gpu))

    if config["dist_url"] == "env://" and config["rank"] == -1:
        config["rank"] = int(os.environ["RANK"])
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    config["rank"] = config["rank"] * ngpus_per_node + gpu
    dist.init_process_group(
        backend=config["dist_backend"], init_method=config["dist_url"], world_size=config["world_size"], rank=config["rank"]
    )

    if config["encoder_type"] == "seq2seq" or config["encoder_type"] == "adv":
        with open(config["vocab_filepath"], "rb") as f:
            vocab = dill.load(f)
        i=0
        pad_id = vocab.stoi['<pad>']
        vocab_size = len(vocab)
        sp = None
    else:
        sp = spm.SentencePieceProcessor()
        sp.Load(config["spm_filepath"])
        pad_id = sp.PieceToId("[PAD]")
        mask_id = sp.PieceToId("[MASK]")
        vocab_size = sp.GetPieceSize()

    def pad_collate(batch):
        B = len(batch)
        if config["program_mode"] == "contrastive":
            X1, X2 = zip(*batch)
            X = X1 + X2
        elif config["program_mode"]=="adv_contrastive":
            X1,X2,X3,idx,fname = zip(*batch)
            X = X1 + X2 + X3
        else:
            X = batch

        # Create tensor of sequence lengths, [B] or [2B]
        lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

        # Create padded tensor for batch, [B, T] or [2B, T]
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        if config["program_mode"] == "contrastive":
            # Reshape X to [B, 2, T]
            T = X.size(-1)
            X = torch.reshape(X, (2, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 2, T)
            lengths = torch.reshape(lengths, (2, B)).transpose(0, 1)
            assert lengths.shape == (B, 2)
        elif config["program_mode"]=="adv_contrastive":
            T = X.size(-1)
            X = torch.reshape(X, (3, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 3, T)
            lengths = torch.reshape(lengths, (3, B)).transpose(0, 1)
            assert lengths.shape == (B, 3)
            return X,lengths,idx,fname,None
        return X, lengths, None
  
    # Create model
    if config["loss_mode"] == "infonce":
        # TODO(ajay): Support n_head argument, check how d_model is being used (why not in encoder config dict?)
        model = CodeMoCo(
            vocab_size,
            pad_id=pad_id,
            d_model=config["d_model"],
            d_rep=config["d_rep"],
            encoder_config=dict(
                encoder_type=config["encoder_type"],
                lstm_project_mode=config["lstm_project_mode"],
                n_encoder_layers=config["n_encoder_layers"],
                max_length=config["max_length"],
            ),
        )
        logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")
    elif config["loss_mode"] == "adv":
        model = CodeMoCo(
            vocab_size,
            pad_id=pad_id,
            d_model=config["d_model"],
            d_rep=config["d_rep"],
            encoder_config=dict(
                encoder_type=config["encoder_type"],
                lstm_project_mode=config["lstm_project_mode"],
                n_encoder_layers=config["n_encoder_layers"],
                max_length=config["max_length"],
            ),
        )
        logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")       
    elif config["loss_mode"] == "mlm":
        model = CodeMLM(
            vocab_size,
            pad_id=pad_id,
            encoder_type=config["encoder_type"],
            n_encoder_layers=config["n_encoder_layers"],
            d_model=config["d_model"],
            n_head=config["n_head"],
            d_ff=4 * config["d_model"],
        )
        logger.info(f"Created CodeMLM model with {count_parameters(model)} params")
    elif config["loss_mode"] == "hybrid":
        model = CodeContrastiveMLM(
            vocab_size,
            pad_id=pad_id,
            n_encoder_layers=config["n_encoder_layers"],
            d_model=config["d_model"],
            n_head=config["n_head"],
            d_ff=4 * config["d_model"],
        )
        logger.info(f"Created CodeContrastiveMLM model with {count_parameters(model)} params")
    else:
        raise ValueError(f"Bad loss mode {config['loss_mode']}")

    
    assert config["use_cuda"]
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        config["batch_size"] = int(config["batch_size"] / ngpus_per_node)
        config["num_workers"] = int((config["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    if config["encoder_type"] == "seq2seq" or config["encoder_type"] == "adv":
        model.half()
    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], betas=config["adam_betas"], eps=1e-6, weight_decay=config["weight_decay"]
    )
    sched = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], config["num_steps"])

    # Load checkpoint
    if config["resume_path"]:
        logger.info(f"Loading parameters from {config['resume_path']}")
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % config["rank"]}
        checkpoint = torch.load(config["resume_path"], map_location=map_location)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        start_global_step = checkpoint["global_step"]
    else:
        start_epoch = 1
        start_global_step = 0

    # Setup data
    train_dataset = PrecomputedDataset(
        config["train_filepath"],
        min_alternatives=config["min_alternatives"],
        program_mode=config["program_mode"],
        limit_size=config["limit_dataset_size"],
        sp=sp,
        subword_regularization_alpha=config["subword_regularization_alpha"],
        max_length=config["max_length"],
        model_type=config["encoder_type"],
        vocab=vocab,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=pad_collate,
        # num_workers=config["num_workers"],
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
    )

    # Train
    global_step = 0
    while global_step < start_global_step:
        sched.step()
        global_step += 1
    train_losses = []
    adv_normal_losses = []
    for epoch in tqdm.trange(start_epoch, config["num_epochs"] + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        total_loss = 0
        num_examples = 0
        for batch in pbar:
            optimizer.zero_grad()
            if config["loss_mode"] == "infonce":
                train_metrics = training_step(model, batch, use_cuda=config["use_cuda"], encoder_type=config["encoder_type"])
            elif config["loss_mode"] == "mlm":
                # replace tokens randomly with tokens from _ (8)
                train_metrics = training_step_mlm(
                    sp, model, batch, pad_id=pad_id, mask_id=mask_id, vocab_start_idx=8, vocab_end_idx=7999, use_cuda=config["use_cuda"]
                )
            elif config["loss_mode"] == "hybrid":
                train_metrics = training_step_hybrid(
                    sp, model, batch, mask_id=mask_id, pad_id=pad_id, vocab_start_idx=0, vocab_end_idx=7999, use_cuda=config["use_cuda"]
                )
            elif config["loss_mode"] == "adv":
                train_metrics=adv_training_step(model, batch, use_cuda=config["use_cuda"], encoder_type=config["encoder_type"],input_vocab=vocab,replace_tokens=replace_tokens,adv_lr=config["adv_lr"],u_pgd_epochs=config["u_pgd_epochs"],num_sites=config["num_sites"],vocab_size=vocab_size)
            else:
                raise ValueError("Bad loss type")
            loss = train_metrics["loss"]
            log = train_metrics["log"] 
            adv_loss = log["pretrain/adv_loss"]
            normal_loss = log["pretrain/normal_loss"]
            loss.backward()
            optimizer.step()
            sched.step()

            total_loss += loss.item()
            num_examples += batch[0].shape[1]

            global_step += 1
            pbar.set_description(f"epoch {epoch} gpu {gpu} step {global_step} loss {loss.item():.4f} adv_loss {adv_loss:.4f} normal_loss {normal_loss:.4f}")
            #logger.info(f"epoch {epoch} gpu {gpu} step {global_step} loss {loss.item():.4f} adv_loss {adv_loss:.4f} normal_loss {normal_loss:.4f}")
            if chief_node:

                # Save checkpoint
                if config["save_every"] and global_step % config["save_every"] == 0:
                    checkpoint = {
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "config": config,
                    }
                    model_file = os.path.join(config["run_dir"], f"ckpt_pretrain_ep{epoch:04d}_step{global_step:07d}.pth")
                    logger.info(f"Saving checkpoint to {model_file}...")
                    torch.save(checkpoint, model_file)
                    logger.info("Done.")
        train_losses.append(total_loss/num_examples)
        adv_normal_losses.append((adv_loss-normal_loss)/num_examples/normal_loss)
        plot_loss(train_losses, os.path.join(config["run_dir"], "loss.png"))
        plot_loss(adv_normal_losses, os.path.join(config["run_dir"], "adv-normal.png"))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(pretrain)
