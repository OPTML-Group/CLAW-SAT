from __future__ import print_function, division
import sys
import torch
import torchtext
import itertools
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.evaluator.metrics import calculate_metrics,compute_distances
import allennlp
import allennlp.nn
import allennlp.nn.beam_search
import torch.nn.functional as F
import tqdm
import time
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

def _evaluate_adaptive(
	model, data_sml, data_med, data_lrg, loss_type="nll_token",batch_size=128,	decode_type="beam_search",
	beam_search_k=10,
	per_node_k=None,
	constrain_decoding=False,
    max_decode_len=20,src_field_name=seq2seq.src_field_name
):
	model.eval()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	src_vocab = data_sml.fields[seq2seq.src_field_name].vocab
	tgt_vocab = data_sml.fields[seq2seq.tgt_field_name].vocab
	pad_id = tgt_vocab.stoi["<pad>"]
	buckets = [
		torchtext.data.BucketIterator(
			dataset=data_sml, batch_size=batch_size,
			sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
			device=device, train=False
		),
		torchtext.data.BucketIterator(
			dataset=data_med, batch_size=batch_size // 2,
			sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
			device=device, train=False
		),
		torchtext.data.BucketIterator(
			dataset=data_lrg, batch_size=1,
			sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
			device=device, train=False
		)
	]
	tot_len = len(buckets[0]) + len(buckets[1]) + len(buckets[2])
	batch_iterator = itertools.chain(*buckets)
	with torch.no_grad():
		output_seqs = []
		ground_truths = []
		orig_output_seqs = []
		all_idxs = []
		batch_generator = batch_iterator.__iter__()
		avg_loss=0
		total_loss=0
		num_examples = 0
		for batch in batch_generator:
			X, X_lengths = getattr(batch, src_field_name)
			Y = batch.tgt
			orig_input_variables, orig_input_lengths  = getattr(batch, 'src')
			trues = ids_to_strs(Y,tgt_vocab)
		# NOTE: X and Y are [B, max_seq_len] tensors (batch first)
			logits = model(X, X_lengths, Y[:,:-1], teacher_forcing_ratio=None)
			orig_logits = model(orig_input_variables,orig_input_variables,Y[:,:-1],teacher_forcing_ratio=None)
			if loss_type == "nll_sequence":
				loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id, reduction="sum")
				loss = loss / X.size(0)  # Average over num sequences, not target sequence lengths
			# Thus, minimize bits per sequence.
			elif loss_type == "nll_tokens":
				loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id,)	
			if decode_type == "greedy":
				preds = greedy_decode(model, X,X_lengths, tgt_vocab, max_decode_len=max_decode_len)
			elif decode_type == "beam_search":
				preds,_ = beam_search_decode(model,X,tgt_vocab,max_decode_len=max_decode_len,k=beam_search_k)
				orig_preds,_= beam_search_decode(model,orig_input_variables,tgt_vocab,max_decode_len=max_decode_len,k=beam_search_k)	
			for pred,true,orig_pred in zip(preds,trues,orig_preds):
				output_seqs.append(' '.join([x for x in pred if x not in ['<sos>','<eos>','<pad>']]))
				orig_output_seqs.append(' '.join([x for x in orig_pred if x not in ['<sos>','<eos>','<pad>']]))
				ground_truths.append(' '.join([x for x in true if x not in ['<sos>','<eos>','<pad>']]))	
		# TODO: Compute Precision/Recall/F1 and BLEU
			torch.cuda.empty_cache()
			total_loss += loss.item() * X.size(0)
			num_examples += X.size(0)
			avg_loss = total_loss / num_examples
		other_metrics = calculate_metrics(output_seqs, ground_truths,orig_y_pred=orig_output_seqs,verbose=True)
		other_metrics["Loss"]=avg_loss
		return other_metrics



class Evaluator(object):
	""" Class to evaluate models with given datasets.
	Args:
		loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
		batch_size (int, optional): batch size for evaluator (default: 64)
	"""

	def __init__(self, loss=NLLLoss(), batch_size=64):
		self.loss = loss
		self.batch_size = batch_size

	def evaluate_adaptive_batch(self, model, data_sml, data_med, data_lrg, verbose=False, src_field_name=seq2seq.src_field_name,get_reps=False):
		""" Evaluate a model on given dataset and return performance.
		Args:
			model (seq2seq.models): model to evaluate
			data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()

		loss = self.loss
		loss.reset()
		match = 0
		total = 0
		print(get_reps)
		print('Adapting batch sizes:')
		print('  - Max: batch_size={}'.format(self.batch_size))
		print('  - Med: batch_size={}'.format(self.batch_size // 2))
		print('  - Min: batch_size=1')
		print('Selected field: {}'.format(src_field_name))

		# device = None if torch.cuda.is_available() else -1
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		use_small = False
		if not use_small:
			buckets = [
				torchtext.data.BucketIterator(
					dataset=data_sml, batch_size=self.batch_size,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				),
				torchtext.data.BucketIterator(
					dataset=data_med, batch_size=self.batch_size // 2,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				),
				torchtext.data.BucketIterator(
					dataset=data_lrg, batch_size=1,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				)
			]
			tot_len = len(buckets[0]) + len(buckets[1]) + len(buckets[2])
		else:
			buckets = [
				torchtext.data.BucketIterator(
					dataset=data_sml, batch_size=self.batch_size,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				),
			]
			tot_len = len(buckets[0])

		batch_iterator = itertools.chain(*buckets)

		src_vocab = data_sml.fields[seq2seq.src_field_name].vocab
		tgt_vocab = data_sml.fields[seq2seq.tgt_field_name].vocab
		pad = tgt_vocab.stoi[data_sml.fields[seq2seq.tgt_field_name].pad_token]
		eos = tgt_vocab.stoi[data_sml.fields[seq2seq.tgt_field_name].SYM_EOS]

		output_seqs, orig_output_seqs = [], []
		input_seqs, orig_input_seqs = [], []
		repre,orig_repre=[],[]
		ground_truths = []
		all_idxs = []

		cnt = 0
		with torch.no_grad():

			if verbose:
				batch_iterator = tqdm.tqdm(
					batch_iterator,
					total=tot_len
				)

			for batch in batch_iterator:
				cnt += 1
			
				# a = time.time()
				input_variables, input_lengths  = getattr(batch, src_field_name)
				orig_input_variables, orig_input_lengths  = getattr(batch, 'src')
				target_variables = getattr(batch, seq2seq.tgt_field_name)
				indices = getattr(batch, 'index')
				# b = time.time()


				if get_reps:
					decoder, encoder  = model(input_variables, input_lengths.tolist(), target_variables,get_reps=get_reps)
					orig_decoder , orig_encoder = model(orig_input_variables, orig_input_lengths.tolist(), target_variables,get_reps=get_reps)
					decoder_outputs, decoder_hidden, other =decoder
					encoder_outputs, encoder_hidden = encoder
					orig_decoder_outputs, orig_decoder_hidden, orig_other =orig_decoder
					orig_encoder_outputs, orig_encoder_hidden = orig_encoder
					len_orig =  orig_encoder_outputs.shape[1]
					len_input = encoder_outputs.shape[1] 
					if len_input < len_orig:
						new_input =  torch.zeros(orig_encoder_outputs.shape)
						new_input[:,0:len_input,:] = encoder_outputs
						encoder_outputs = new_input
					elif len_input > len_orig :
						continue
						new_input =  np.zeros(encoder_outputs.shape)
						new_input[:,0:len_input,:] = orig_encoder_outputs
						orig_encoder_outputs = new_input						
					#print(orig_encoder_outputs.shape)
					#print(encoder_outputs.shape)
					#sys.exit()
					for i , (encoder_output,orig_encoder_output) in enumerate(zip(encoder_outputs, orig_encoder_outputs)):
						repre.append(encoder_output.squeeze().detach().cpu().numpy())
						orig_repre.append(orig_encoder_output.squeeze().detach().cpu().numpy() )
				else:
					decoder_outputs, decoder_hidden, other  = model(input_variables, input_lengths.tolist(), target_variables)
					orig_decoder_outputs, orig_decoder_hidden, orig_other = model(orig_input_variables, orig_input_lengths.tolist(), target_variables)

				# c = time.time()
				# decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist())

				# print(target_variables, target_variables.size())
				# print(len(decoder_outputs), decoder_outputs[0].size())

				# Evaluation
				seqlist = other['sequence']
				for step, step_output in enumerate(decoder_outputs):
					target = target_variables[:, step + 1]
					loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

					non_padding = target.ne(pad)
					non_eos = target.ne(eos)
					mask = torch.mul(non_padding, non_eos)
					# mask = non_padding 

					correct = seqlist[step].view(-1).eq(target).masked_select(mask).sum().item()
					match += correct
					total += mask.sum().item()

				# d = time.time()

				# print(other['length'])
				# print(other['sequence'])
				oh_no = 0
				for i, (output_seq_len, orig_output_seq_len, idxs) in enumerate(zip(other['length'], orig_other['length'], indices)):
					# print(i,output_seq_len)
					tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
					tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
					# print(tgt_seq)
					output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
					gt = [tgt_vocab.itos[tok] for tok in target_variables[i]]
					ground_truths.append(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))
					
					orig_tgt_id_seq = [orig_other['sequence'][di][i].data[0] for di in range(orig_output_seq_len)]
					orig_tgt_seq = [tgt_vocab.itos[tok] for tok in orig_tgt_id_seq]
					orig_output_seqs.append(' '.join([x for x in orig_tgt_seq if x not in ['<sos>','<eos>','<pad>']]))

					all_idxs.append(idxs.unsqueeze(0).detach().cpu().numpy()[0].item())

					'''
					orig_input_seqs.append([src_vocab.itos[t] for t in orig_input_variables[i]])
					input_seqs.append([src_vocab.itos[t] for t in input_variables[i]])
					
					if orig_output_seqs != output_seqs:
						# print([src_vocab.itos[t] for t in output_seqs[i]])
						# print([src_vocab.itos[t] for t in orig_output_seqs[i]])
						pass
					'''

				# e = time.time()
				# print(cnt, b-a, c-b, d-c, e-d)
				torch.cuda.empty_cache()

				# model.encoder.embedded[0].detach()
		# print(output_seqs)
		# print(ground_truths)
		# ground_truths = [' '.join(l[1:-1]) for l in data.tgt]
		# for i in range(len(ground_truths)):
			# print(ground_truths[i], " ---- ", output_seqs[i])
		# print([a for a in data.tgt])
		
		def add_idx(inp_list):
			di = {}
			for i, j in zip(all_idxs, inp_list):
				di[i] = j
			return di
		
		other_metrics = calculate_metrics(output_seqs, ground_truths, orig_output_seqs)
		other_metrics['li_exact_match'] = add_idx(other_metrics['li_exact_match'])
		other_metrics['li_orig_match'] = add_idx(other_metrics['li_orig_match'])
		output_seqs = add_idx(output_seqs)
		ground_truths = add_idx(ground_truths)
		
		'''
		for e in other_metrics['err_idx']:
			print(input_seqs[e])
			print(orig_input_seqs[e])
			print('*')
			print(orig_output_seqs[e])
			print(output_seqs[e])
			print('---')
		'''

		# match_idxs = [all_idxs[eidx] for eidx in other_metrics['exact_match_idx_orig']]
		# other_metrics['exact_match_idx_orig'] = match_idxs
		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

		other_metrics.update({'Loss':loss.get_loss(), 'accuracy (torch)': accuracy*100})
		d = {
				'metrics': other_metrics,
				'output_seqs': output_seqs,
				'ground_truths': ground_truths
			}
		if get_reps:
			
			distances=compute_distances(repre,orig_repre)
			mean_dist=sum(distances)/len(distances)
			d={
				'metrics': other_metrics,
				'output_seqs': output_seqs,
				'ground_truths': ground_truths,
				'distances' : distances,
				'mean_distances' : mean_dist,
			}
		return d

	def evaluate(self, model, data, verbose=False, src_field_name=seq2seq.src_field_name):
		""" Evaluate a model on given dataset and return performance.
		Args:
			model (seq2seq.models): model to evaluate
			data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()
		other_metrics,output_seqs,ground_truths = _evaluate(model=model,data=data,loss_type=self.loss)
		d = {
				'metrics': other_metrics,
				'output_seqs': output_seqs,
				'ground_truths': ground_truths
			}

		return d