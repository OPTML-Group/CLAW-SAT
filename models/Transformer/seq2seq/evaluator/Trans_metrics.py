from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from seq2seq.evaluator.metrics import calculate_metrics
import allennlp
import allennlp.nn
import allennlp.nn.beam_search
import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset.fields import get_seq2seq_data_generator
from seq2seq.evaluator.f1 import F1MetricMethodName
from tqdm import tqdm
@torch.no_grad()
def beam_search_decode(
	model,
	X,
	vocab,
	max_decode_len,
	k,
	per_node_k=None,
	constrain_decoding=False,
	sampler="deterministic",
	top_p_threshold=0.9,
	top_p_temperature=1.0,
	):
	if sampler == "top_p":
		sampler = allennlp.nn.beam_search.TopPSampler(p=top_p_threshold, temperature=top_p_temperature)
	elif sampler == "deterministic":
		sampler = None
	else:
		raise ValueError("Unsupported sampler")

	# TODO: Implement constrained decoding (e.g. only alphanumeric)
	B = X.size(0)
	bos_id = vocab.stoi['<sos>']
	eos_id = vocab.stoi['<eos>']
	pad_id = vocab.stoi["<pad>"]
	V_full = len(vocab)  # Size of vocab
	invalid_vocab_mask = torch.zeros(V_full, dtype=torch.bool, device=X.device)
	if constrain_decoding:
		alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_ "
		for id in range(V_full):
			piece = vocab.itos[id]
			if not (id in [pad_id, bos_id, eos_id] or all(c in alphabet for c in piece)):
				invalid_vocab_mask[id] = True
	V = V_full
	model.eval()

	# Encode X
	allen_bs = allennlp.nn.beam_search.BeamSearch(
		end_index=eos_id, max_steps=max_decode_len, beam_size=k, per_node_beam_size=per_node_k, sampler=sampler,
	)

	start_predictions = torch.tensor([bos_id] * B, dtype=torch.long, device=X.device)
	start_state = {
		"prev_tokens": torch.zeros(B, 0, dtype=torch.long, device=X.device),
		"memory": model.encode(X).transpose(0, 1),  # [B, T, d_model]
	}

	def step(last_tokens, current_state, t):
		"""
		Args:
			last_tokens: (group_size,)
			current_state: {}
			t: int
		"""
		group_size = last_tokens.size(0)
		prev_tokens = torch.cat([current_state["prev_tokens"], last_tokens.unsqueeze(1)], dim=-1)  # [B*k, t+1]

		all_log_probs = model.decode(current_state["memory"].transpose(0, 1), prev_tokens)
		next_log_probs = all_log_probs[:, -1, :]
		if constrain_decoding:
			next_log_probs = next_log_probs.masked_fill(invalid_vocab_mask, float("-inf"))
		next_log_probs = torch.nn.functional.log_softmax(next_log_probs, dim=-1)
		assert next_log_probs.shape == (group_size, V)
		return (next_log_probs, {"prev_tokens": prev_tokens, "memory": current_state["memory"]})

	predictions, log_probs = allen_bs.search(start_predictions=start_predictions, start_state=start_state, step=step)
	top_beams = []
	for i in range(B):
		# print("pred_id",predictions[i][0])
		# print("pred_str",ids_to_strs(predictions[i][0],vocab))
		top_beam = ids_to_strs(predictions[i][0],vocab)
		top_beams.append(top_beam)
	model.train()
	

	return top_beams, log_probs




def ids_to_strs(Y, vocab):
	if len(Y.shape) == 1:
		ids = []
		eos_id = vocab.stoi['<eos>']
		pad_id = vocab.stoi["<pad>"]
		for idx in Y:
			ids.append(vocab.itos[int(idx)])
			if int(idx) == eos_id or int(idx) == pad_id:
				break
			# if int(idx) == eos_id:
			# 	break
		return ids
	return [ids_to_strs(y, vocab) for y in Y]

### data batch ###
### input of model ###
### evluation level ### 
### train level ### 

def greedy_decode(model, X, X_lengths, vocab, max_decode_len=20, sample=True):
	start_token = vocab.stoi['<sos>']
	pad_token = vocab.stoi["<pad>"]
	B = X.size(0)
	model.eval()

	with torch.no_grad():
		decoded_batch = torch.zeros((B, 1), device=X.device).long()
		decoded_batch[:, 0] = start_token
		for t in range(max_decode_len):
			logits= model(X, X_lengths, decoded_batch, teacher_forcing_ratio=None)
			_, topi = logits[:, -1, :].topk(1)
			decoded_batch = torch.cat((decoded_batch, topi.view(-1, 1)), -1)
	Y_hat = decoded_batch.cpu().numpy()
	Y_hat_str = ids_to_strs(Y_hat, vocab)
	model.train()
	return Y_hat_str



def calculate_f1_metric(
    metric: F1MetricMethodName,
    model,
    data,
	decode_type,
	beam_search_k=10,
	per_node_k=None,batch_size=128,
	constrain_decoding=False,
    max_decode_len=20 # see empirical evaluation of CDF of subwork token length
):
	sample_generations = []
	n_examples = 0
	precision, recall, f1 = 0.0, 0.0, 0.0
	output_seqs = []
	ground_truths = []
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	src_vocab = data.fields[seq2seq.src_field_name].vocab
	tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
	batch_iterator = torchtext.data.BucketIterator(
		dataset=data, batch_size=batch_size,
		sort=False, sort_within_batch=True,
		sort_key=lambda x: len(x.src),
		device=device, repeat=False)
	batch_generator = batch_iterator.__iter__()
	for batch in tqdm(batch_generator):
		X, X_lengths = batch.src
		Y = batch.tgt
		trues = ids_to_strs(Y,tgt_vocab)

		if decode_type == "greedy":
			preds = greedy_decode(model, X,X_lengths, tgt_vocab, max_decode_len=max_decode_len)
		elif decode_type == "beam_search":
			preds,_ = beam_search_decode(model,X,tgt_vocab,max_decode_len=max_decode_len,k=beam_search_k)
		# print("preds:",preds)
		# print("groundtruth:",trues)
		# print("input:",ids_to_strs(X,src_vocab))
		# exit(0)
		# X_seq = ids_to_strs(X,input_vocab)
		# print(X_seq)
		# print(preds)
		# print(trues)
		# exit(0)
		for pred,true in zip(preds,trues):
			output_seqs.append(' '.join([x for x in pred if x not in ['<sos>','<eos>','<pad>']]))
			ground_truths.append(' '.join([x for x in true if x not in ['<sos>','<eos>','<pad>']]))
        # pbar.set_postfix(avg_metrics)
        # if logger_fn is not None:
        #     logger_fn(item_metrics)
        #     logger_fn(avg_metrics)
		torch.cuda.empty_cache()
	other_metrics = calculate_metrics(output_seqs, ground_truths,verbose=True)
    # logger.debug(f"Test set evaluation (F1) took {t.interval:.3}s over {n_examples} samples")
	return other_metrics





def _evaluate(
	model, data, loss_type="nll_token",batch_size=128
):
	model.eval()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	src_vocab = data.fields[seq2seq.src_field_name].vocab
	tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
	pad_id = tgt_vocab.stoi["<pad>"]
	batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=batch_size,
			sort=False, sort_within_batch=True,
			sort_key=lambda x: len(x.src),
			device=device, repeat=False)
	with torch.no_grad():
		batch_generator = batch_iterator.__iter__()
		avg_loss=0
		total_loss=0
		num_examples = 0
		for batch in batch_generator:
			X, X_lengths = batch.src
			Y = batch.tgt
		# NOTE: X and Y are [B, max_seq_len] tensors (batch first)
			logits = model(X, X_lengths, Y[:,:-1], teacher_forcing_ratio=None)
			if loss_type == "nll_sequence":
				loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id, reduction="sum")
				loss = loss / X.size(0)  # Average over num sequences, not target sequence lengths
			# Thus, minimize bits per sequence.
			elif loss_type == "nll_token":
				loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id,)			
		# TODO: Compute Precision/Recall/F1 and BLEU

			total_loss += loss.item() * X.size(0)
			num_examples += X.size(0)
			avg_loss = total_loss / num_examples
        #         pbar.set_description(f"evaluate average loss {avg_loss:.4f}")
        # logger.debug(f"Loss calculation took {t.interval:.3f}s")
		return avg_loss