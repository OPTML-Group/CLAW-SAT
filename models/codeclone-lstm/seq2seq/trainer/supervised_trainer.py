from __future__ import division
import logging
import os
import random
import time
import numpy as np
import torch
import torchtext
from torch import optim
from torch.utils.tensorboard import SummaryWriter


import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score,precision_recall_fscore_support


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


def best_binary_accuracy_pred(y_true, y_pred, num_thresholds=100):
    assert y_true.shape == y_pred.shape
    thresholds = np.linspace(0, 1, num=num_thresholds)
    best_acc = 0.0
    for threshold in thresholds:
        y_pred = (y_pred > threshold).astype(y_true.dtype)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_pred = y_pred
    return best_acc,best_threshold,best_pred

# def pad(input_variables,input_variables_2,input_lengths,input_lengths_2,pad_token):
# 	new_input_lengths = torch.cat((input_lengths,input_lengths_2))
# 	print(new_input_lengths.shape)
# 	if input_variables.shape[1] > input_variable_2.shape[1]:
# 		tmp = torch.ones(input_variables.shape)*pad_token.to(new_input_lengths.device)
# 		tmp[:,:input_variable_2.shape[1]] = input_variable_2
# 		tmp = torch.cat((input))
# 	else:
# 		tmp = torch.ones(input_variables_2.shape)*pad_token.to(new_input_lengths.device)
# 		tmp[:,:input_variable_1.shape[1]] = input_variable_1
# 	return 
		

		
		

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
	def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
				 random_seed=None,
				 checkpoint_every=1000, print_every=100, tensorboard=True):
		self._trainer = "Simple Trainer"
		self.random_seed = random_seed
		if random_seed is not None:
			random.seed(random_seed)
			torch.manual_seed(random_seed)
		self.loss = loss
		self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.batch_size = batch_size

		self.logger = logging.getLogger(__name__)

		self.writer = SummaryWriter(log_dir=expt_dir) if tensorboard else None

	def _train_batch(self, input_variable, input_variable_2,input_lengths,input_lengths_2, label, model, teacher_forcing_ratio):
		loss = self.loss
		# Forward propagation
		# print(input_variable)
		# print(input_variable_2)
		# print(label)
		output = model(input_variable, input_variable_2,input_lengths, input_lengths_2, teacher_forcing_ratio=teacher_forcing_ratio)
		# Get loss
		# print(output)
		train_loss = loss(output,label)
		# Backward propagation
		model.zero_grad()
		train_loss.backward()
		self.optimizer.step()

		return train_loss.item()

	def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
					   dev_data=None, teacher_forcing_ratio=0):
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
			dev_loss,precision,recall,f1,threshold = _evaluate(model,dev_data,self.loss,self.batch_size,teacher_forcing_ratio=teacher_forcing_ratio)
			best_f1 = f1
			best_acc = threshold

		else:
			best_f1 = 0.0
			best_acc = 0.0
			threshold = 0.5

		Checkpoint(model=model,
				   optimizer=self.optimizer,threshold=threshold,
				   epoch=0, step=step,
				   input_vocab=data.fields[seq2seq.src_field_name].vocab,
				   output_vocab=data.fields[seq2seq.src_field_name].vocab).save(self.expt_dir, name='Best_F1')

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
				input_variables_2,input_lengths_2 = getattr(batch, seq2seq.tgt_field_name)
				label = getattr(batch,"label")
				# pad(input_variables,input_variables_2,input_lengths,input_lengths_2,seq2seq.src_field_name.vocab.stoi('<pad>'))
				# # print(input_variables+input_variables_2)
				# print(input_variables.shape)
				# print(input_lengths.shape)
				# print(input_lengths_2.shape)
				# exit(0)
				zero = torch.zeros_like(label,dtype=torch.long)
				label = torch.where(label==-1,zero,label)
				# print(label)
				train_loss = self._train_batch(input_variables, input_variables_2,input_lengths.tolist(),input_lengths_2.tolist(),label, model, teacher_forcing_ratio)
				# train_loss = 0
				# Record average loss
				print_loss_total += train_loss
				epoch_loss_total += train_loss


				if step % self.print_every == 0 and step_elapsed >= self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					log_msg = 'Epoch: %d, Step: %d, Progress: %d%%, Train %s: %.4f' % (epoch, step,
						step / total_steps * 100,
						"Bceloss",
						print_loss_avg)
					log.info(log_msg)

					if self.writer:
						self.writer.add_scalar('Train/loss_step', print_loss_avg, step)

				# Checkpoint
				# if step % self.checkpoint_every == 0 or step == total_steps:
				#     Checkpoint(model=model,
				#                optimizer=self.optimizer,
				#                epoch=epoch, step=step,
				#                input_vocab=data.fields[seq2seq.src_field_name].vocab,
				#                output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)
				#     log.info('Checkpoint saved')

			if step_elapsed == 0: 
				continue

			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, "BCE", epoch_loss_avg)
			self.writer.add_scalar('Train/loss_epoch', epoch_loss_avg, epoch)

			other_metrics = {}
			if dev_data is not None:
				dev_loss,precision,recall,f1,threshold = _evaluate(model,dev_data,self.loss,self.batch_size,teacher_forcing_ratio=teacher_forcing_ratio)
				accuracy = precision
				self.optimizer.update(dev_loss, epoch)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f,Recall: %.4f,threshold: %.4f" % ("BCE", dev_loss, accuracy,recall,threshold)
				self.writer.add_scalar('Val/loss', dev_loss, epoch)
				self.writer.add_scalar('Val/acc', accuracy, epoch)
				self.writer.add_scalar('Val/recall', recall, epoch)
				# for metric in other_metrics:
				# 	try:
				# 		log_msg += ", %s: %.4f"%(metric.replace(' ','_').replace('-','_'), other_metrics[metric])
				# 		self.writer.add_scalar('Val/%s'%metric.replace(' ','_').replace('-','_'), other_metrics[metric], epoch)
				# 	except:
				# 		continue

				log_msg = log_msg[:-1]
				log.info(log_msg)

				if f1 > best_f1:
					Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,threshold=threshold,
								   input_vocab=data.fields[seq2seq.src_field_name].vocab,
								   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Best_F1')
					log_msg = 'Checkpoint saved, Epoch %d, Prev Val F1: %.4f, New Val F1: %.4f' % (epoch, best_f1, f1)
					log.info(log_msg)
					best_f1 = f1

				# if accuracy > best_acc:
				#     Checkpoint(model=model,
				#                    optimizer=self.optimizer,
				#                    epoch=epoch, step=step,
				#                    input_vocab=data.fields[seq2seq.src_field_name].vocab,
				#                    output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Best_Acc')
				#     log_msg = 'Checkpoint saved, Epoch %d, Prev Val Acc: %.4f, New Val Acc: %.4f' % (epoch, best_acc, accuracy)
				#     log.info(log_msg)
				#     best_acc = accuracy

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
			  optimizer=None, teacher_forcing_ratio=0, load_checkpoint=None):
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
