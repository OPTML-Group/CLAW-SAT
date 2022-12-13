from email import encoders
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
def plot_loss_advs(normal,adv,plot_name):
    plt.plot(normal,label="normal")
    plt.plot(adv,label="adv")
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.legend(['normal', 'adv'])
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
    if encoder_type=="adv_transformer":
        imgs_k_onehot = Variable(convert_to_onehot(imgs_k, vocab_size=vocab_size), requires_grad=True)
        imgs_q_onehot = Variable(convert_to_onehot(imgs_q, vocab_size=vocab_size), requires_grad=True)
    else:
        imgs_k_onehot = Variable(convert_to_onehot(imgs_k, vocab_size=vocab_size), requires_grad=True).half()
        imgs_q_onehot = Variable(convert_to_onehot(imgs_q, vocab_size=vocab_size), requires_grad=True).half()
    #input_oho = Variable(convert_to_onehot(input_variables,vocab_size=vocab_size),requires_grad=True).half()
    
    status_map, z_map, z_all_map, z_np_map, site_map_map, site_map_lookup_map, z_initialized_map, invalid_tokens_mask_map = {}, {}, {}, {}, {}, {}, {}, {}
    invalid_tokens_mask = get_valid_token_mask(negation=True, vocab=input_vocab, exclude=[])
    for ii in range(input_variables.shape[0]):
        # print(input_variables.cpu().numpy()[ii])
        replace_map_i,site_map,status = get_all_replacement_toks(input_variables.cpu().numpy()[ii], None, input_vocab, replace_tokens)
        # print(site_map)
        # print(status)
        if not status:
            continue

        site_map_lookup = []
        for cnt, k in enumerate(site_map.keys()):
            site_map_lookup.append(k)
        rdm_idx_list = list(range(len(site_map_lookup)))
        if num_sites>=1:
            rdm_idx = random.sample(rdm_idx_list, min(len(rdm_idx_list), num_sites))
        else:
            rdm_idx = random.sample(rdm_idx_list,len(rdm_idx_list))
        z_np = np.zeros(len(site_map_lookup)).astype(float)
        z_np[rdm_idx] = 1
        z = torch.tensor(z_np,requires_grad=True,device=device)
        if len(z.shape)==1:
            z = z.unsqueeze(dim=1)
        mask = np.array(input_variables.cpu().numpy()[ii]*[False]).astype(bool)
        for kk in range(len(site_map_lookup)):
            if not z[kk]:
                continue
            m = site_map[site_map_lookup[kk]]
            mask = np.array(m) | mask
        status_map[ii]=status
        z_map[ii]=z
        z_np_map[ii]=z_np
        z_all_map[ii]=list(mask)
        site_map_map[ii] = site_map
        site_map_lookup_map[ii] = site_map_lookup      
    new_inputs_oho = imgs_k_onehot.clone()  
    u_init_pgd = 3
    for i in range(input_variables.shape[0]):
        if i not in status_map:
            continue
        input_hot = new_inputs_oho[i].detach().cpu().numpy()
        for z in range(z_np_map[i].shape[0]):

            if z_np_map[i][z] == 0:
                continue
            # Make input_oho[i] zero for tokens which correspond to
            # - sites z_i = True
            # - and haven't been initialized before
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
        new_inputs_oho[i] = torch.tensor(input_hot, requires_grad=True, device=device)    
    # print("input's length")
    # print(len(input_variables))
    # print("input's length")
    # print(len(site_map_lookup_map))
    # print(site_map_lookup_map)
    for j in range(u_pgd_epochs):
        #P vs P_random without update queue
        # a = new_inputs_oho.argmax(2)
        # # print(a)
        # m = torch.zeros(new_inputs_oho.shape, requires_grad=True, device=device).scatter(2, a.unsqueeze(2), 1.0).half()
        # print(m)
        # print(type(m))
        # exit()
        output, target = model(new_inputs_oho, imgs_q_onehot, lengths_k, lengths_q,already_one_hot=True,update_queue=False)
        # print("input:",new_inputs_oho)
        # print("output:",output)
        assert torch.isnan(new_inputs_oho).sum() == 0, print("before attack input is nan")
        assert torch.isnan(output).sum() == 0, print("before attack output is nan")
        loss = F.cross_entropy(output, target)
        assert torch.isnan(loss).sum() == 0, print(loss)

        # orig_random_loss = loss
        model.zero_grad()
        new_inputs_oho.retain_grad()
        loss.backward(retain_graph=True)
        grads_oh = new_inputs_oho.grad
        # print("grad:",grads_oh)
        assert torch.isnan(grads_oh).sum() == 0, print(grads_oh)
        # print(grads_oh)
        # print("sucess")
        for i in range(input_variables.shape[0]):
            if i not in status_map:
                continue
            input_hot = new_inputs_oho[i].detach().cpu().numpy()
            best_replacements_sample = {}
            gradients = grads_oh[i].cpu().numpy()
            site_map_lookup = site_map_lookup_map[i]
            z=z_map[i]
            z_np=z_np_map[i]
            site_map = site_map_map[i]
            invalid_tokens_mask_i = invalid_tokens_mask[:]
            for idx in range(z_np.shape[0]):
                if z_np[idx] == 0:
                    continue

            #idx = np.argmax(z_np)
                mask = site_map[site_map_lookup[idx]]
                avg_tok_grads=np.mean(gradients[mask],axis=0)
                repl_tok_idx = site_map_lookup[idx]
                repl_tok = input_vocab.itos[repl_tok_idx]
                nabla = avg_tok_grads
                #nabla = np.sign(nabla)
                step = (adv_lr)/np.sqrt(j+1) * nabla
                input_h = input_hot[mask,:][0,:]
                input_h = input_h + step
            # fmu = lambda mu, a=input_h: np.sum(np.maximum(0, a - mu )) - 1
            # mu_opt = bisection(fmu, -1, 1, 20)
            # if mu_opt is None:
            #     mu_opt = 0 # assigning randomly to 0
                optim_input = np.clip(input_h, 0, 1)
                optim_input[invalid_tokens_mask_i] = 0
                # fmu = lambda mu, a=input_h: np.sum(np.maximum(0, a - mu )) - 1
                # mu_opt = bisection(fmu, -1, 1, 20)
                # if mu_opt is None:
                #     mu_opt = 0 # assigning randomly to 0
                # optim_input = np.maximum(0, input_h - mu_opt)
                # optim_input[invalid_tokens_mask_i] = 0

            # input_hot[mask,:] = optim_input
                max_idx = np.argmax(optim_input)
                best_replacements_sample[repl_tok] = input_vocab.itos[max_idx]
                invalid_tokens_mask_i[max_idx] = True
                #optim_input[:] = 0  
                #optim_input[max_idx]=1
                input_hot[mask,:] = optim_input
            new_inputs_oho[i]=torch.tensor(input_hot,requires_grad=True,device=device)
        assert torch.isnan(new_inputs_oho).sum() == 0, print("after attack become nan")
    # output, target = model(new_inputs_oho, imgs_q_onehot, lengths_k, lengths_q,already_one_hot=True,update_queue=False)
    # adv_random_loss = F.cross_entropy(output, target)
    # P vsP random + P_adv vs P random
    a = new_inputs_oho.argmax(2)
    if encoder_type == "adv_transformer":
        m = torch.zeros(new_inputs_oho.shape,requires_grad=True,device=device).scatter(2,a.unsqueeze(2), 1.0)
    else:    
        m = torch.zeros(new_inputs_oho.shape,requires_grad=True,device=device).scatter(2,a.unsqueeze(2), 1.0).half()
    output,adv_output,target = model(imgs_k_onehot, imgs_q_onehot, lengths_q, lengths_k,already_one_hot=True,update_queue=True,im_adv = m,lengths_adv=lengths_k)
    adv_loss = F.cross_entropy(adv_output, target) 
    assert torch.isnan(adv_loss).sum() == 0, print(adv_loss)
    normal_loss = F.cross_entropy(output, target)
    loss = adv_loss + normal_loss

    
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
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

    if config["encoder_type"] == "seq2seq" or config["encoder_type"] == "adv" or config["encoder_type"]=="adv_transformer":
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

    if config["encoder_type"] == "seq2seq" or config["encoder_type"] == "adv" :
        model.half()
    # elif config["encoder_type"]=="adv_transformer":
    #     model.half()
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
    adv_losses = []
    normal_losses = []
    for epoch in tqdm.trange(start_epoch, config["num_epochs"] + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        total_loss = 0
        total_normal_loss = 0
        total_adv_loss = 0
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
            total_normal_loss += normal_loss
            total_adv_loss += adv_loss
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
        adv_normal_losses.append((total_adv_loss-total_normal_loss)/total_normal_loss)
        adv_losses.append(total_adv_loss/num_examples)
        normal_losses.append(total_normal_loss/num_examples)
        plot_loss(train_losses, os.path.join(config["run_dir"], "loss.png"))
        plot_loss(adv_normal_losses, os.path.join(config["run_dir"], "adv-normal.png"))
        plot_loss_advs(normal_losses,adv_losses,os.path.join(config["run_dir"], "normal_adv_loss.png"))
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(pretrain)
