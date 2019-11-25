import pickle, argparse, os, random, subprocess, math, sys
from tqdm import tqdm 
import colorlog, logging
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

logging.disable(logging.DEBUG)
colorlog.basicConfig(
	filename=None,
	level=logging.NOTSET,
	format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)
from CustomDataset import TrainDataset, EvalDataset
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
summary = SummaryWriter()

# Model Loading
from model import *
class EngineState():
	def __init__(self):
		super(EngineState, self).__init__()
class Engine():
	def __init__(self, args, model, optimizer, epoch=1, iteration=1):
		self.args = args
		self.model = model
		self.optimizer = optimizer

		self.state = EngineState()
		colorlog.info("[Engine Initialized] Epoch {} Iteration {}".format(epoch, iteration))
		self.state.epoch = epoch
		self.state.iteration = iteration
		self.state.best_param_path = None
		self.state.best_accuracy = 0
	def start(self):
		# Manual Random Seed
		random.seed(self.args.random_seed)
		np.random.seed(self.args.random_seed)
		torch.manual_seed(self.args.random_seed)
		torch.cuda.manual_seed(self.args.random_seed)
		torch.backends.cudnn.deterministic=True

		self.model.init_param()
		# self.model.init_param(save_init_param=True)

	def epoch_start(self):
		colorlog.info("[Engine Started]")
	def iteration_start(self):
		self.model.zero_grad()
		self.optimizer.zero_grad()
	def iteration_complete(self): 
		self.state.iteration += 1
	def epoch_complete(self):
		colorlog.info("[{}'th Epoch Complete]".format(self.state.epoch))
		self.state.epoch += 1
	def complete(self):
		colorlog.info("[Engine Complete]")
class Instructor:
	def __init__(self, args):
		self.args = args
		if 'basis' in self.args.model_type:
			colorlog.critical("[coef_act_fn] {}".format(self.args.coef_act_fn))
			colorlog.critical("[num_bases] {}".format(self.args.num_bases))
			colorlog.critical("[attribute_dim] {}".format(self.args.attribute_dim))
			colorlog.critical("[key_query_size] {}".format(self.args.key_query_size))
		if 'stochastic' in self.args.model_type:
			colorlog.critical("[n_samples] {}".format(self.args.n_samples))
			colorlog.critical("[eval_n_samples] {}".format(self.args.eval_n_samples))
		
		colorlog.critical("[reg_kd] {}".format(self.args.reg_kd))

		self.alpha = 0.0

		self.std_max = self.args.std_max
		self.std_min = self.args.std_min

		self.ent_max = self.args.ent_max
		self.ent_min = self.args.ent_min

		self.std_list = []
		self.ent_list = []

		if self.args.uncertainty_method == "std" and self.args.std_update == "true":
			self.std_max = 0.0
			self.std_min = sys.float_info.max

		if self.args.uncertainty_method == "ent" and self.args.ent_update == "true":
			self.ent_max = 0.0
			self.ent_min = sys.float_info.max

		# Model & Optimizer
		if 'stochastic' in self.args.model_type:
			colorlog.critical("Initiate StochasticClassifier")
			self.model = StochasticClassifier(self.args).to(self.args.device)
		else:
			colorlog.critical("Initiate DeterministicClassifier")
			self.model = DeterministicClassifier(self.args).to(self.args.device)

		# Dataset
		self.train_dataset = TrainDataset(args=self.args, name="train", data_path = self.args.data_path_prefix+"train.txt",)
		self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.train_dataset.custom_collate_fn)

		self.dev_dataset = EvalDataset(args=self.args, name="dev", data_path=self.args.data_path_prefix+"dev.txt")
		self.dev_dataloader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.dev_dataset.custom_collate_fn)

		self.test_dataset = EvalDataset(args=self.args, name="test", data_path=self.args.data_path_prefix+"test.txt")
		self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.test_dataset.custom_collate_fn)

	def get_alpha(self, uncertainty): # function to get alpha value when using linear contribution function
		max_val, min_val = 0.0, 0.0
		if self.args.uncertainty_method == "std":
			max_val = self.std_max
			min_val = self.std_min
		elif self.args.uncertainty_method == "ent":
			max_val = self.ent_max
			min_val = self.ent_min

		m = (-1.0) / (max_val - min_val)
		k = max_val / (max_val - min_val)

		alpha = (uncertainty * m) + k # N
		alpha = torch.clamp(alpha, max=1.0, min=0.0)

		return alpha

	def train(self):
		# Engine
		self.optimizer = torch.optim.Adadelta(self.model.parameters())
		self.engine = Engine(args=self.args, model=self.model, optimizer=self.optimizer)

		dataloader = self.train_dataloader

		self.engine.start()
		for i_epoch in range(self.args.max_epochs):
			self.engine.epoch_start()
			for i_sample, sample_batch in enumerate(tqdm(dataloader)):
				self.engine.iteration_start()
				(
					text, length, mask,
					user, product,
					label,
					cust_teacher_logit,
					non_cust_teacher_logit
				) = sample_batch
				output_prob, output_logits_sampled = self.model(text, length, mask, **{'user':user, 'product':product}) # N, 5
				loss_gt = self.model.get_loss(output_prob, label) # Loss from ground-truth label

				# 1. standard variation for uncertainty
				output_logits_std = output_logits_sampled.std(dim=1) # N, C
				output_logits_std = output_logits_std.sum(dim=-1) # N

				# 2. entropy for uncertainty
				probability_sampled = F.softmax(output_logits_sampled, -1) # N, B, C
				p = probability_sampled.mean(1) # N, C
				entropy = (-p*torch.log(p)).sum(-1) # N # another uncertainty measurement instead of output_logits_std

				if i_epoch == 0:
					self.std_list += output_logits_std.tolist()
					self.ent_list += entropy.tolist()

				if self.args.uncertainty_method == "std" and self.args.std_update == "true":
					self.std_max = max(self.std_max, output_logits_std.max().item())
					self.std_min = min(self.std_min, output_logits_std.min().item())

				if self.args.uncertainty_method == "ent" and self.args.ent_update == "true":
					self.ent_max = max(self.ent_max, entropy.max().item())
					self.ent_min = min(self.ent_min, entropy.min().item())

				if self.args.uncertainty_method == "std":
					self.alpha = self.get_alpha(output_logits_std) # N
				elif self.args.uncertainty_method == "ent":
					self.alpha = self.get_alpha(entropy)

				if self.args.uncertainty_method == "std" or self.args.uncertainty_method == "ent":
					teacher_logit = (cust_teacher_logit * self.alpha.unsqueeze(dim=-1).repeat(1, self.args.num_labels)).unsqueeze(1) \
								+ (non_cust_teacher_logit * (1.0 - self.alpha).unsqueeze(dim=-1).repeat(1, self.args.num_labels)).unsqueeze(1)
				elif self.args.uncertainty_method == "":
					teacher_logit = cust_teacher_logit.unsqueeze(1)

				loss_kd = (
					(output_logits_sampled-teacher_logit)**2
					).sum(-1).mean(0).mean(0)

				loss = loss_gt+self.args.reg_kd * loss_kd
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
				self.optimizer.step()
				if self.args.eval_step and self.engine.state.iteration%self.args.eval_step==0: 
					self.validation()

				self.engine.iteration_complete()
			self.engine.epoch_complete()
		self.engine.complete()

		#self.std_list = sorted(self.std_list)
		#self.ent_list = sorted(self.ent_list)
		print(max(self.std_list))
		print(min(self.std_list))
		print(max(self.ent_list))
		print(min(self.ent_list))

		plt.subplot(2, 1, 1)
		plt.plot(self.std_list, 'r.')
		plt.title("top: std, bottom: ent")

		plt.subplot(2, 1, 2)
		plt.plot(self.ent_list, 'r.')
		plt.xlabel("idx")

		plt.show()

		std_data_std = np.std(self.std_list)
		std_data_mean = np.mean(self.std_list)
		std_cut_off = std_data_std * 3

		std_lower = std_data_mean - std_cut_off
		std_upper = std_data_mean + std_cut_off
		final_std_list = []
		for value in self.std_list:
			if std_lower <= value <= std_upper:
				final_std_list.append(value)

		ent_data_std = np.std(self.ent_list)
		ent_data_mean = np.mean(self.ent_list)
		ent_cut_off = ent_data_std * 3

		ent_lower = ent_data_mean - ent_cut_off
		ent_upper = ent_data_mean + ent_cut_off
		final_ent_list = []
		for value in self.ent_list:
			if ent_lower <= value <= ent_upper:
				final_ent_list.append(value)

		plt.subplot(2, 1, 1)
		plt.plot(final_std_list, 'r.')
		plt.title("top: std, bottom: ent")

		plt.subplot(2, 1, 2)
		plt.plot(final_ent_list, 'r.')
		plt.xlabel("idx")

		plt.show()

	def test(self):
		colorlog.critical("[Evaluation on Test Set]")
		best_param_path = os.path.join(self.args.param_dir, self.args.model_type)
		self.model.load_state_dict(torch.load(best_param_path, map_location=self.args.device)['state_dict'])
		colorlog.critical("[Best Parameter Loading] {}".format(best_param_path))

		if self.args.uncertainty_method == "std":
			fname = "std"
		elif self.args.uncertainty_method == "ent":
			fname = "ent"
		elif self.args.uncertainty_method == "":
			fname = "cust"
		# sigfile = open("result/sig_result/"+fname+".txt","w")
		sigfile = open(fname + ".txt","w")

		dataloader = self.test_dataloader
		N = len(dataloader.dataset)
		B = dataloader.batch_size
		pred_np = np.zeros((N), dtype=np.int64)
		target_np = np.zeros((N), dtype=np.int64)
		with torch.no_grad():
			self.model.eval()
			for i_sample, sample_batch in enumerate(dataloader):
				(text, length, mask,
				user, product,
				label) = sample_batch
				pred = self.model(text, length, mask, **{'user':user, 'product':product}) # N, 5
				pred = torch.argmax(pred, dim=-1)
				pred_np[i_sample*B:(i_sample+1)*B] = pred.cpu().data.numpy()
				target_np[i_sample*B:(i_sample+1)*B] = label.cpu().data.numpy()

				compare = torch.eq(pred, label)
				c_size = compare.size()[0]
				for idx in range(c_size):
					if compare[idx].item():
						sigfile.write("1\n")
					else:
						sigfile.write("0\n")
			self.model.train()
		acc = (pred_np==target_np).mean()
		rmse = ((pred_np-target_np)**2).mean()**0.5
		return acc, rmse, best_param_path

	def validation(self):
		dataloader = self.dev_dataloader
		N = len(dataloader.dataset)
		B = dataloader.batch_size
		pred_np = np.zeros((N), dtype=np.int64)
		target_np = np.zeros((N), dtype=np.int64)
		with torch.no_grad():
			self.model.eval()
			for i_sample, sample_batch in enumerate(dataloader):
				(text, length, mask,
				user, product,
				label) = sample_batch
				pred = self.model(text, length, mask, **{'user':user, 'product':product}) # N, 5
				pred = torch.argmax(pred, dim=-1)
				pred_np[i_sample*B:(i_sample+1)*B] = pred.cpu().data.numpy()
				target_np[i_sample*B:(i_sample+1)*B] = label.cpu().data.numpy()
			self.model.train()
		acc = (pred_np==target_np).mean()
		rmse = ((pred_np-target_np)**2).mean()**0.5
		print("acc: {:2.2f}%, rmse: {:.3f}".format(acc*100, rmse))
		if self.engine.state.best_accuracy<acc:
			path = os.path.join(self.args.param_dir, self.args.model_type)
			torch.save({
				'state_dict':self.model.state_dict(),
				'optimizer_state_dict':self.optimizer.state_dict(),
			}, path)
			self.engine.state.best_accuracy = acc
			self.engine.state.best_param_path = path
			colorlog.info(">> parameter saved {}".format(path))

		return acc, rmse
			
local_data_path = "/home/poolc2/Workspace/nip/kd/data"
parser = argparse.ArgumentParser()
parser.add_argument("--cust_teacher_logit_path", required=True, type=str, help="file path for attained output logits using cust teacher")
parser.add_argument("--non_cust_teacher_logit_path", required=True, type=str, help="file path for attained output logits using non-cust teacher")
parser.add_argument("--random_seed", required=True, help="The random_seed must be the same with seed used for teacher (for the same initialization with the teacher)", type=int)
parser.add_argument("--reg_kd", default=0.01, type=float, help="contribution of knowledge distillation objective on output logits from teacher")

parser.add_argument("--version_log", default="", type=str)
parser.add_argument("--n_experiments", default=10, type=int)
parser.add_argument("--subdir", default="", type=str)

parser.add_argument("--coef_act_fn", default='softmax', type=str, choices=['none', 'softmax', 'relu', 'tanh', 'sigmoid'])
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--eval_n_samples", default=5000, type=int)

baseline_models = ['non_cust']
cust_models = ['word_cust', 'encoder_cust', 'attention_cust', 'linear_cust', 'bias_cust']
basis_cust_models = ['word_basis_cust', 'encoder_basis_cust', 'attention_basis_cust', 'linear_basis_cust', 'bias_basis_cust']
stochastic_basis_cust_models = ['stochastic_linear_basis_cust']
model_choices = baseline_models + cust_models + basis_cust_models + stochastic_basis_cust_models
parser.add_argument("--model_type", choices=model_choices, help="Give model type.")

parser.add_argument("--num_labels", default=5, type=int)
parser.add_argument("--num_bases", default=3, type=int)
parser.add_argument("--attribute_dim", default=64, type=int)
parser.add_argument("--key_query_size", default=64, type=int)
parser.add_argument("--bias_basis_cust_state_size", default=128, type=int)

parser.add_argument("--word_dim", default=300, type=int)
parser.add_argument("--state_size", default=256, type=int)

parser.add_argument("--data_path_prefix", default=local_data_path)
parser.add_argument("--vocab_path", default=local_data_path+"42939.vocab", type=str)
parser.add_argument("--pretrained_word_em_dir", default=local_data_path+"word_vectors.npy", type=str)

parser.add_argument("--param_dir", default='./param/', type=str)

parser.add_argument("--device", default='cuda', type=str)

parser.add_argument("--eval_batch_size", default=50, type=int)

parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--max_epochs", default=10, type=int)
parser.add_argument("--eval_step", default=1000, type=int)

parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--max_grad_norm", default=3.0, type=float)

parser.add_argument("--attribute_dropout", default=0.2, type=float)

parser.add_argument("--num_user", default=1631, type=int)
parser.add_argument("--num_product", default=1633, type=int)

# 1. std(standard variation), 2. ent(entropy)
parser.add_argument("--uncertainty_method", required=True, type=str)

parser.add_argument("--std_update", required=True, type=str)
parser.add_argument("--std_max", default=5.1, type=float)
parser.add_argument("--std_min", default=0.026, type=float)

parser.add_argument("--ent_update", required=True, type=str)
parser.add_argument("--ent_max", default=1.6, type=float)
parser.add_argument("--ent_min", default=0.001, type=float)

args = parser.parse_args()
args.meta_units = [("user", args.num_user), ("product", args.num_product)]
if args.subdir:
	if not os.path.exists(args.subdir): os.mkdir(args.subdir)
	os.chdir(os.path.join(os.getcwd(), args.subdir))
args.device = torch.device(args.device)

vocab = open(args.vocab_path, encoding='utf-8').read().split()
args.word2idx = {x:i for i, x in enumerate(vocab)}
args.idx2word = {i:x for i, x in enumerate(vocab)}
args._ipad, args._iunk \
= args.word2idx['<PAD>'], args.word2idx['<UNK>']
args.vocab_size = len(vocab)

if not os.path.exists(args.param_dir): os.mkdir(args.param_dir)

ins = Instructor(args)

acc_list = []
rmse_list = []
for i_exp in range(1, args.n_experiments+1):
	args.i_experiment = i_exp
	
	# Manual Random Seed
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	torch.backends.cudnn.deterministic=True

	terminate = False
	colorlog.info("[{}'th Experiment] Start !".format(i_exp))
	try:
		ins.train()
	except KeyboardInterrupt as e:
		colorlog.critical("KeyboardInterrupt Occurs")
		while True:
			Y_or_n = input("Do you want to terminate this process? [Y|n] ")
			if Y_or_n!='Y' and Y_or_n!='n':
				print("The only possible answer is \"Y\" or \"n\"")
			else:
				break
		terminate = True if Y_or_n=='Y' else False

	# Would you include final result?
	if terminate:
		while True:
			Y_or_n = input("Do you want to exclude the final result from the list of results? [Y|n] ")
			if Y_or_n!='Y' and Y_or_n!='n':
				print("The only possible answer is \"Y\" or \"n\"")
			else:
				break
		if Y_or_n=='Y': 
			subprocess.call("rm ./param/*", shell=True)
			subprocess.call("rm -r ./log/*", shell=True)
			break

	acc, rmse, best_param_path = ins.test()
	acc_list.append(acc)
	rmse_list.append(rmse)
	repr_result = """
		<< {}'th Experiment >>
		[ Result on Test Dataset ]
		param_path: {}
		Accuracy: {:2.3f}%, RMSE {:2.3f}
		""".format(i_exp, best_param_path, acc*100, rmse)

	if terminate: break

acc_result = """{result_list}
Mean {avg}
Std {std}
""".format(
	result_list=" ".join(["{:.4f}".format(x) for x in acc_list]),
	avg = np.mean(acc_list),
	std = np.std(acc_list),
	)
rmse_result = """{result_list}
Mean {avg}
Std {std}
""".format(
	result_list=" ".join(["{:.4f}".format(x) for x in rmse_list]),
	avg = np.mean(rmse_list),
	std = np.std(rmse_list),
	)
open("Accuracy","w").write(acc_result)
open("RMSE","w").write(rmse_result)
print("""
[version_log]: {}
[subdir]     : {}
**************** Result ****************
[Accuracy]
{}
[RMSE]
{}
[standard deviation]
Max: {} / Min: {}
[entropy]
Max: {} / Min: {}""".format(
	args.version_log, args.subdir,
	acc_result, rmse_result,
	ins.std_max, ins.std_min,
	ins.ent_max, ins.ent_min
))
