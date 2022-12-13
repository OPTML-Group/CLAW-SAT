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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
# from seq2seq.trainer import _evaluate,Precision_Recall_F1
import tqdm
import time
def Precision_Recall_F1(y_pred,y_true):

    tp = (y_true * y_pred).sum()
    # print(tp)
    # exit(0)
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    precision = tp/(tp+fp+1e-7)
    recall = tp/(tp+fn+1e-7)
    f1 = 2 * precision* recall /(precision+recall+1e-7)
    acc = (y_true == y_pred).sum()/ (tp+tn+fp+fn)

    return precision,recall,f1,acc


def _evaluate(model,data,loss,batch_size,teacher_forcing_ratio=0):
	model.eval()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_iterator = torchtext.data.BucketIterator(
		dataset=data, batch_size=batch_size,
		sort=False, sort_within_batch=True,
		sort_key=lambda x: len(getattr(x,seq2seq.src_field_name)),
		device=device, repeat=False)
	with torch.no_grad():
		pred_lst = []
		label_lst = []
		total_loss = 0
		cnt = 0
		batch_generator = batch_iterator.__iter__()
		tp,tn,fp,fn = 0,0,0,0
		for batch in batch_generator:
			cnt += 1
			input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
			input_variables_2,input_lengths_2 = getattr(batch, seq2seq.tgt_field_name)

			label = getattr(batch,"label")
			zero = torch.zeros_like(label,dtype=torch.long)
			one = torch.ones_like(label,dtype=torch.long)
			label = torch.where(label==-1,zero,label)
			output = model(input_variables, input_variables_2,input_lengths.tolist(), input_lengths_2.tolist(), teacher_forcing_ratio=teacher_forcing_ratio)
			val_loss = loss(output,label)
			total_loss += val_loss
			_,output = output.topk(1)
			output=output.view(1,-1).squeeze()
			# print("output:",output)
			# print("label:",label)
			# print("output shape:",output.shape)
			# print("label shape:",label.shape)
			pred_lst.append(output.cpu().detach().numpy())
			label_lst.append(label.cpu().detach().numpy())
		# tp,tn,fp,fn = Precision_Recall_F1(pred,label)
		# epsilon = 1e-7
		# precision = tp / (tp + fp + epsilon)
		# recall = tp / (tp + fn + epsilon)
		# f1 = 2* (precision*recall) / (precision + recall + epsilon)
		preds = np.concatenate(pred_lst)
		# print("preds:",preds.sum())
		
		# print(preds)
		labels = np.concatenate(label_lst)
		# print("labels:",labels.sum())
		# print(labels)
		best_threshold = 0
		# _,best_threshold,best_preds = best_binary_accuracy_pred(labels,preds)
		precision,recall,f1,acc = Precision_Recall_F1(preds,labels)
		# print(recall)
		# print(precision)
		# print(f1)
		return total_loss/cnt, precision, recall,f1,acc



class Evaluator(object):
	""" Class to evaluate models with given datasets.

	Args:
		loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
		batch_size (int, optional): batch size for evaluator (default: 64)
	"""

	def __init__(self, loss=NLLLoss(), batch_size=64):
		self.loss = loss
		self.batch_size = batch_size

	def evaluate_adaptive_batch(self, model, data_sml, data_med, data_lrg, verbose=False, src_field_name=seq2seq.src_field_name,get_reps=False,teacher_forcing_ratio=0):
		""" Evaluate a model on given dataset and return performance.

		Args:
			model (seq2seq.models): model to evaluate
			data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()
		for params in model.parameters():
			print(params)
		model.float()
		loss = torch.nn.CrossEntropyLoss()
		# loss.reset()
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

		# src_vocab = data_sml.fields[seq2seq.src_field_name].vocab
		# tgt_vocab = data_sml.fields[seq2seq.tgt_field_name].vocab
		# pad = tgt_vocab.stoi[data_sml.fields[seq2seq.tgt_field_name].pad_token]
		# eos = tgt_vocab.stoi[data_sml.fields[seq2seq.tgt_field_name].SYM_EOS]

		# output_seqs, orig_output_seqs = [], []
		# input_seqs, orig_input_seqs = [], []
		# repre,orig_repre=[],[]
		# ground_truths = []
		# all_idxs = []

		cnt = 0
		with torch.no_grad():

			if verbose:
				batch_iterator = tqdm.tqdm(
					batch_iterator,
					total=tot_len
				)
			total_loss = 0
			pred_lst= []
			label_lst=[]
			for batch in batch_iterator:
				cnt += 1
			
				# a = time.time()
				input_variables, input_lengths  = getattr(batch, src_field_name)
				input_variables_2, input_lengths_2  = getattr(batch, 'tgt')
				orig_input_variables, orig_input_lengths  = getattr(batch, 'src')
				# tp,tn,fp,fn = 0,0,0,0
				# target_variables = getattr(batch, seq2seq.tgt_field_name)
				label = getattr(batch,"label")
				zero = torch.zeros_like(label,dtype=torch.long)
				one = torch.ones_like(label,dtype=torch.long)
				label = torch.where(label==-1,zero,label)
				output = model(input_variables, input_variables_2,input_lengths.tolist(), input_lengths_2.tolist(), teacher_forcing_ratio=teacher_forcing_ratio)
				val_loss = loss(output,label)
				total_loss += val_loss.item()
				_,output = output.topk(1)
				output=output.view(1,-1).squeeze()
				# print("output:",output)
				# print("label:",label)
				# print("output shape:",output.shape)
				# print("label shape:",label.shape)
				pred_tmp = output.cpu().detach().numpy()
				# print(pred_tmp)
				try: 
					len(pred_tmp)
				except:
					pred_tmp=np.array([pred_tmp])
				pred_lst.append(pred_tmp)
				label_tmp = label.cpu().detach().numpy()
				try: 
					len(label_tmp)
				except:
					label_tmp=np.array([label_tmp])			
				label_lst.append(label_tmp)

				


				# b = time.time()




				# d = time.time()



				# e = time.time()
				# print(cnt, b-a, c-b, d-c, e-d)
				torch.cuda.empty_cache()
		# if total == 0:
		# 	accuracy = float('nan')
		# else:
		# 	accuracy = match / total
		# print(pred_lst[-1])
		preds = np.concatenate(pred_lst)
		# print("preds:",preds.sum())
		
		# print(preds)
		labels = np.concatenate(label_lst)
		# print("labels:",labels.sum())
		# print(labels)
		# best_threshold = 0
		# _,best_threshold,best_preds = best_binary_accuracy_pred(labels,preds)
		precision,recall,f1,acc = Precision_Recall_F1(preds,labels)	
		# print(precision,recall,f1,acc)	
		other_metrics ={}
		other_metrics['f1'] = f1*100
		other_metrics["precision"] = precision*100
		other_metrics["recall"] = recall*100
		other_metrics["loss"] = total_loss/cnt
		other_metrics["acc"] = acc
		d = {
				'metrics': other_metrics,
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

		loss = self.loss
		loss.reset()
		match = 0
		total = 0


		# device = None if torch.cuda.is_available() else -1
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=self.batch_size,
			sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
			device=device, train=False)
		src_vocab = data.fields[seq2seq.src_field_name].vocab
		tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
		pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]
		eos = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].SYM_EOS]

		output_seqs = []
		ground_truths = []

		cnt = 0
		with torch.no_grad():

			if verbose:
				batch_iterator = tqdm.tqdm(batch_iterator)

			for batch in batch_iterator:
				cnt += 1
				# a = time.time()

				# print(src_field_name)
				input_variables, input_lengths  = getattr(batch, src_field_name)
				target_variables = getattr(batch, seq2seq.tgt_field_name)
				# b = time.time()

				decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

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

				for i,output_seq_len in enumerate(other['length']):
					# print(i,output_seq_len)
					tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
					tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
					# print(tgt_seq)
					output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
					gt = [tgt_vocab.itos[tok] for tok in target_variables[i]]
					ground_truths.append(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))

					# if get_attributions:
					#     a
					#     exit() 

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
		other_metrics = calculate_metrics(output_seqs, ground_truths)


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

		return d