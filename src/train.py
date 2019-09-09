import pickle
import colorlog, logging
logging.disable(logging.DEBUG)
colorlog.basicConfig(
	filename=None,
	level=logging.NOTSET,
	format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)
from NSC import NSC
from CustomDatset import TrainDataset, EvalDataset
from torch.utils.data import Dataset, DataLoader
class EngineState():
	def __init__(self):
		super(EngineState, self).__init__()
		pass
class Engine():
	def __init__(self, epoch, iteration):
		self.state = EngineState()
		colorlog.info("[Engine Initialized] Epoch {} Iteration {}".format(epoch, iteration))
		self.state.epoch = epoch
		self.state.iteration = iteration
class Instructor:
	def __init__(self, args):
		self.args = args
		self.model = NSC(self.args)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		self.dev_dataset = EvalDataset(args=self.args, data_path=self.args.data_path_prefix+"_dev.pkl")
		self.dev_dataloader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.dev_dataset.custom_collate_fn)
	def train(self):
		train_dataset = TrainDataset(args = self.args, data_path = self.args.data_path_prefix+"_train.pkl")
		train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_dataset.custom_collate_fn)
		for i_epoch in range(1, self.args.max_epochs):
			for i_sample, sample_batch in enumerate(train_dataloader):
				(text, length, mask, 
				user, product,
				label) = sample_batch
				pred = self.model(text, length, mask,user, product) # N, 5
				loss = F.cross_entropy(pred, label)
				loss.backward()
				self.optimizer.update()
				self.model.zero_grad()
				self.optimizer.zero_grad()
	def validation(self):
		dataloader = self.dev_dataloader
		pred_np = np.zeros((len()))
		target_np
		for i_sample, sample_batch in enumerate(train_dataloader):
			(text, length, mask, 
			user, product,
			label) = sample_batch
			pred = self.model(text, length, mask,user, product) # N, 5
			pred = torch.argmax(pred, dim=-1) 
			