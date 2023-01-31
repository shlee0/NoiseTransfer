import abc
import torch


class BaseModel(torch.nn.Module, metaclass=abc.ABCMeta):
	def __init__(self, name):
		super(BaseModel, self).__init__()
		self.name = name
		
		
	def reset_metrics(self):
		for key in self.metrics:
			self.metrics[key].reset()


	def update_metrics(self, logs):
		for key in self.metrics:
			self.metrics[key].update(logs[key].detach())
			

	def result(self):
		rets = {}
		for key in self.metrics:
			rets[key] = self.metrics[key].compute().cpu().numpy()
		return rets


	def save(self, net, dir_path, file_name):
		if isinstance(net, torch.nn.DataParallel):
			net.module.save(dir_path, file_name)
		else:
			net.save(dir_path, file_name)


	def load(self, net, dir_path, file_name):
		if isinstance(net, torch.nn.DataParallel):
			net.module.load(dir_path, file_name)
		else:
			net.load(dir_path, file_name)