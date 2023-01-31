import abc, os
import torch

from utils import make_dir

class BaseNet(torch.nn.Module, metaclass=abc.ABCMeta):
	def __init__(self, name):
		super(BaseNet, self).__init__()
		self.name = name
		

	def save(self, dir_path, file_name):
		if file_name is None:
			file_name = self.name

		make_dir(dir_path)
		checkpoint_path = os.path.join(dir_path, file_name + '.pt')
		torch.save(self.state_dict(), checkpoint_path)

		
	def load(self, dir_path, file_name):
		if file_name is None:
			file_name = self.name

		checkpoint_path = os.path.join(dir_path, file_name + '.pt')
		self.load_state_dict(torch.load(checkpoint_path))