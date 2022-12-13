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
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
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

def greedy_decode(model, X, X_lengths, Y,vocab, max_decode_len=20, sample=True):
	start_token = vocab.stoi['<sos>']
	pad_token = vocab.stoi["<pad>"]
	B = X.size(0)
	model.eval()

	with torch.no_grad():
		decoded_batch = torch.zeros((B, 1), device=X.device).long()
		decoded_batch[:, 0] = start_token
		for t in range(20):
			logits= model(X, X_lengths, decoded_batch, teacher_forcing_ratio=1)
			_, topi = logits[:, -1, :].topk(1)
			decoded_batch = torch.cat((decoded_batch, topi.view(-1, 1)), -1)
	Y_hat = decoded_batch.cpu().numpy()
	Y_hat_str = ids_to_strs(Y_hat, vocab)
	model.train()
	return Y_hat_str



def _evaluate(
	model, data, loss_type="nll_token",batch_size=128,	decode_type="beam_search",
	beam_search_k=10,
	per_node_k=None,
	constrain_decoding=False,
    max_decode_len=20,src_field_name=seq2seq.src_field_name
):
	model.eval()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	src_vocab = data.fields[seq2seq.src_field_name].vocab
	tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
	pad_id = tgt_vocab.stoi["<pad>"]
	batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=batch_size,
			sort=False, sort_within_batch=True,
			sort_key=lambda x: len(getattr(x,src_field_name)),
			device=device, repeat=False)
	with torch.no_grad():
		output_seqs = []
		ground_truths = []
		orig_output_seqs = []
		batch_generator = batch_iterator.__iter__()
		avg_loss=0
		total_loss=0
		num_examples = 0
		for batch in tqdm(batch_generator):
			X, X_lengths = getattr(batch, src_field_name)
			Y = batch.tgt
			orig_input_variables, orig_input_lengths  = getattr(batch, 'src')
			trues = ids_to_strs(Y,tgt_vocab)
		# NOTE: X and Y are [B, max_seq_len] tensors (batch first)
			logits = model(X, X_lengths, Y[:,:-1],teacher_forcing_ratio=1)
			orig_logits = model(orig_input_variables,orig_input_variables,Y[:,:-1],teacher_forcing_ratio=1)
			if loss_type == "nll_sequence":
				loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id, reduction="sum")
				loss = loss / X.size(0)  # Average over num sequences, not target sequence lengths
			# Thus, minimize bits per sequence.
			elif loss_type == "nll_tokens":
				loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id,)	
			if decode_type == "greedy":
				preds = greedy_decode(model, X,X_lengths,Y, tgt_vocab, max_decode_len=max_decode_len)
				orig_preds = greedy_decode(model, orig_input_variables,X_lengths,Y, tgt_vocab, max_decode_len=max_decode_len)
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
		d = {
				'metrics': other_metrics,
				'output_seqs': output_seqs,
				'ground_truths': ground_truths
			}
		return d



class SupervisedTrainer(object):
	""" The SupervisedTrainer class helps in setting up a training framework in a
	supervised setting.

	Args:
    expt_dir (optional, str): experiment Directory to store details of the experiment,
    	by default it makes a folder in the current directory to store the details (default: `experiment`).
    loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
    batch_size (int, optional): batch size for experiment, (default: 64)
    checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
	"""
	def __init__(self, expt_dir='experiment', loss="nll", batch_size=64,
		random_seed=None,
		checkpoint_every=1000, print_every=100, tensorboard=True,transformer=False,ignore_index=None,dev_path=None,input_vocab=None,output_vocab=None):

		self._trainer = "Simple Trainer"
		self.random_seed = random_seed
		if random_seed is not None:
			random.seed(random_seed)
			torch.manual_seed(random_seed)
		self.loss = loss
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.Transformer = transformer
		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		self.ignore_index = ignore_index
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.batch_size = batch_size
		self.dev_path=dev_path
		self.logger = logging.getLogger(__name__)
		self.input_vocab=input_vocab
		self.output_vocab=output_vocab
		self.writer = SummaryWriter(log_dir=expt_dir) if tensorboard else None

	def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
		loss = self.loss
		# Forward propagation

		decoder_outputs = model(input_variable, input_lengths, target_variable[:,:-1], teacher_forcing_ratio=teacher_forcing_ratio)

		batch_size = target_variable.size(0)
		if loss == "nll_sequence":
			loss= F.cross_entropy(decoder_outputs.transpose(1, 2), target_variable[:, 1:],ignore_index=self.ignore_index,reduction="sum")
			loss=loss/batch_size
		elif loss == "nll_tokens":
			loss = F.cross_entropy(decoder_outputs.transpose(1, 2), target_variable[:, 1:], ignore_index=self.ignore_index,)
		# Backward propagation
		model.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()


	def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
					   dev_data=None, teacher_forcing_ratio=1):
		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# device = None if torch.cuda.is_available() else -1
		batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=self.batch_size,
			sort=False, sort_within_batch=True,
			sort_key=lambda x: len(x.src),
			device=device, repeat=False)

		steps_per_epoch = len(batch_iterator)
		total_steps = steps_per_epoch * n_epochs

		self.print_every = steps_per_epoch // 25

		log.info('Steps per epoch: %d'%steps_per_epoch)
		log.info('Total steps: %d'%total_steps)

		step = start_step
		step_elapsed = 0

		# num_checkpoints = 25
		# self.checkpoint_every = (total_steps+1)//num_checkpoints


		if start_step>0 and dev_data is not None:
			d = _evaluate(model=model,data=dev_data,loss_type=self.loss)
			best_acc=d['metrics']['word-level accuracy']
			best_f1=d['metrics']['f1']
		else:
			best_f1 = 0.0
			best_acc = 0.0
		Checkpoint(model=model,
				   optimizer=self.optimizer,
				   epoch=0, step=step,
				   input_vocab=self.input_vocab,
				   output_vocab=self.output_vocab).save(self.expt_dir, name='Best_F1')

		for epoch in range(start_epoch, n_epochs + 1):
			log.debug("Epoch: %d, Step: %d" % (epoch, step))

			batch_generator = batch_iterator.__iter__()
			# consuming seen batches from previous training
			for _ in range((epoch - 1) * steps_per_epoch, step):
				next(batch_generator)


			model.train(True)
			for batch in batch_generator:
				step += 1
				step_elapsed += 1

				input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
				target_variables = getattr(batch, seq2seq.tgt_field_name)
				
				loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, teacher_forcing_ratio)
				#loss = 0
				# Record average loss
				print_loss_total += loss
				epoch_loss_total += loss


				if step % self.print_every == 0 and step_elapsed >= self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					log_msg = 'Epoch: %d, Step: %d, Progress: %d%%, Train %s: %.4f' % (epoch, step,
						step / total_steps * 100,
						self.loss,
						print_loss_avg)
					log.info(log_msg)

					if self.writer:
						self.writer.add_scalar('Train/loss_step', print_loss_avg, step)

			if step_elapsed == 0: 
				continue

			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss, epoch_loss_avg)
			self.writer.add_scalar('Train/loss_epoch', epoch_loss_avg, epoch)

			other_metrics = {}
			if dev_data is not None:
				metric=F1MetricMethodName()
				d = _evaluate(model=model,data=dev_data,loss_type=self.loss) 
				accuracy=d['metrics']['word-level accuracy']
				F1=d['metrics']['f1']
				dev_loss = d['metrics']["Loss"]
				self.optimizer.update(dev_loss, epoch)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss, dev_loss, accuracy)
				self.writer.add_scalar('Val/loss', dev_loss, epoch)
				self.writer.add_scalar('Val/acc', accuracy, epoch)
				self.writer.add_scalar('Val/F1', F1, epoch)

				log_msg = log_msg[:-1]
				log.info(log_msg)
				if F1 > best_f1:
					Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=self.input_vocab,
								   output_vocab=self.output_vocab).save(self.expt_dir, name='Best_F1')
					log_msg = 'Checkpoint saved, Epoch %d, Prev Val F1: %.4f, New Val F1: %.4f' % (epoch, best_f1, F1)
					log.info(log_msg)
					best_f1 = F1

				model.train(mode=True)

			else:
				self.optimizer.update(epoch_loss_avg, epoch)

			# Checkpoint(model=model,
			#                    optimizer=self.optimizer,
			#                    epoch=epoch, step=step,
			#                    input_vocab=data.fields[seq2seq.src_field_name].vocab,
			#                    output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Latest')
			# log_msg = 'Latest Checkpoint saved, Epoch %d, %s' % (epoch, str(other_metrics))
			# log.info(log_msg)


	def train(self, model, data, num_epochs=5,
			  resume=False, dev_data=None,
			  optimizer=None, teacher_forcing_ratio=1, load_checkpoint=None):
		""" Run training for a given model.

		Args:
			model (seq2seq.models): model to run training on, if `resume=True`, it would be
			overwritten by the model loaded from the latest checkpoint.
			data (seq2seq.dataset.dataset.Dataset): dataset object to train on
			num_epochs (int, optional): number of epochs to run (default 5)
			resume(bool, optional): resume training with the latest checkpoint, (default False)
			dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
			optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
			(default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
			teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
		Returns:
			model (seq2seq.models): trained model.
		"""
		# If training is set to resume
		if resume:
			if load_checkpoint is None:
				load_checkpoint = Checkpoint.get_latest_checkpoint(self.expt_dir)
			resume_checkpoint = Checkpoint.load(load_checkpoint)
			model = resume_checkpoint.model
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

			self.logger.info("Resuming training from %d epoch, %d step" % (start_epoch, step))
		else:
			start_epoch = 1
			step = 0
			if optimizer is None:
				optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(data, model, num_epochs,
							start_epoch, step, dev_data=dev_data,
							teacher_forcing_ratio=teacher_forcing_ratio)
		return model
