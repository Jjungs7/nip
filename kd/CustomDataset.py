import os, numpy as np, pickle, torch, random, itertools
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset
import colorlog, logging
logging.disable(logging.DEBUG)
colorlog.basicConfig(
	filename=None,
	level=logging.NOTSET,
	format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)
def text_padding(text, length, padding_idx, return_mask=False):
	""" 
	text: list of token indices
	length: list of length of tokens (same size with text)
	return: padded text tokens, ndarray, np.int64
	"""
	maxlen = max(length)
	num_data = len(text)
	if return_mask:
		mask = np.zeros((num_data, maxlen), dtype=np.uint8)
	padded_sentences = np.zeros((num_data, maxlen), dtype=np.int64)+padding_idx
	if return_mask:
		for i, (l, x) in enumerate(zip(length, text)):
			padded_sentences[i][:l] = x
			mask[i][:l]=1
		return padded_sentences, mask
	else:
		for i, (l, x) in enumerate(zip(length, text)):
			padded_sentences[i][:l] = x
		return padded_sentences

class CustomDataset(Dataset):
	def __init__(self, args, name, data_path):
		self.args=args
		self.name=name
		self.data=self.read_data(data_path=data_path)
		self.len=len(self.data)
		colorlog.info("Dataset: {}, Size: {}".format(self.name, self.len))
		colorlog.info("*"*50)

	def __getitem__(self, index): return self.data[index]
	def __len__(self): return self.len
	def read_data(self, data_path): pass
	def custom_collate_fn(self, batch): pass
	def data_transform(self):pass

class TrainDataset(CustomDataset):
	def __init__(self, args, name, *nargs, **kwargs):
		super().__init__(args, name, *nargs, **kwargs)
	
	def custom_collate_fn(self, sample_batch):
		user, product, rating, length, text\
		, cust_teacher_logit, non_cust_teacher_logit\
		 = list(zip(*sample_batch))
		N = len(text)
		
		# to Tensor
		user = torch.LongTensor(user).to(self.args.device) # N
		product = torch.LongTensor(product).to(self.args.device) # N
		rating = torch.LongTensor(rating).to(self.args.device) # N
		text, mask = text_padding(text, length, padding_idx=0, return_mask=True)
		text = torch.from_numpy(text).to(self.args.device) # N, L
		length = torch.LongTensor(length).to(self.args.device) # N, L
		mask = torch.from_numpy(mask.astype(np.float32)).to(self.args.device) # N, L

		cust_teacher_logit = torch.Tensor(cust_teacher_logit).to(self.args.device)
		non_cust_teacher_logit = torch.Tensor(non_cust_teacher_logit).to(self.args.device)

		return (text, length, mask, user, product, rating,
			cust_teacher_logit, non_cust_teacher_logit
			)
	def read_data(self, data_path):
		with open(data_path, 'r') as f:
			data = f.read().split("\n")
		user, product, rating, length, review = list(zip(*[x.split(",") for x in data]))
		user = [int(x) for x in user]
		product = [int(x) for x in product]
		rating = [int(x) for x in rating]
		length = [int(x) for x in length]
		review = [[int(x) for x in xs.split("_")] for xs in review]

		colorlog.info("[Load knowledge distillation targets]")
		cust_teacher_logit = np.load(self.args.cust_teacher_logit_path)
		colorlog.info("cust_teacher_logit_path: {}, shape {}".format(self.args.cust_teacher_logit_path, cust_teacher_logit.shape))
		non_cust_teacher_logit = np.load(self.args.non_cust_teacher_logit_path)
		colorlog.info("non_cust_teacher_logit_path: {}, shape {}".format(self.args.non_cust_teacher_logit_path, non_cust_teacher_logit.shape))

		return list(zip(user, product, rating, length, review, cust_teacher_logit, non_cust_teacher_logit))
class EvalDataset(CustomDataset):
	def __init__(self, args, name, *nargs, **kwargs):
		super().__init__(args, name, *nargs, **kwargs)
	def custom_collate_fn(self, sample_batch):
		user, product, rating, length, text = list(zip(*sample_batch))
		N = len(text)
		
		# to Tensor
		user = torch.LongTensor(user).to(self.args.device) # N
		product = torch.LongTensor(product).to(self.args.device) # N
		rating = torch.LongTensor(rating).to(self.args.device) # N
		text, mask = text_padding(text, length, padding_idx=0, return_mask=True)
		text = torch.from_numpy(text).to(self.args.device) # N, L
		length = torch.LongTensor(length).to(self.args.device) # N, L
		mask = torch.from_numpy(mask.astype(np.float32)).to(self.args.device) # N, L
		return (text, length, mask, user, product, rating)
	def read_data(self, data_path):
		with open(data_path, 'r') as f:
			data = f.read().split("\n")
		user, product, rating, length, review = list(zip(*[x.split(",") for x in data]))
		user = [int(x) for x in user]
		product = [int(x) for x in product]
		rating = [int(x) for x in rating]
		length = [int(x) for x in length]
		review = [[int(x) for x in xs.split("_")] for xs in review]
		return list(zip(user, product, rating, length, review))
