"""
https://arxiv.org/abs/1904.07396
https://github.com/saeed-anwar/RIDNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
		
		
class BasicBlock(nn.Module):
	def __init__(self,
				 in_channels, out_channels,
				 ksize=3, stride=1, pad=1):
		super(BasicBlock, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
			nn.ReLU(inplace=True)
		)
		
	def forward(self, x):
		out = self.body(x)
		return out
		
		
class BasicBlockSig(nn.Module):
	def __init__(self,
				 in_channels, out_channels,
				 ksize=3, stride=1, pad=1):
		super(BasicBlockSig, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		out = self.body(x)
		return out
		
		
class Merge_Run_dual(nn.Module):
	def __init__(self,
				 in_channels, out_channels,
				 ksize=3, stride=1, pad=1, dilation=1):
		super(Merge_Run_dual, self).__init__()

		self.body1 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
			nn.ReLU(inplace=True)
		)
		self.body2 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
			nn.ReLU(inplace=True)
		)

		self.body3 = nn.Sequential(
			nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
			nn.ReLU(inplace=True)
		)
		
	def forward(self, x):
		out1 = self.body1(x)
		out2 = self.body2(x)
		c = torch.cat([out1, out2], dim=1)
		c_out = self.body3(c)
		out = c_out + x
		return out
		
		
class ResidualBlock(nn.Module):
	def __init__(self, 
				 in_channels, out_channels):
		super(ResidualBlock, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, 1, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, 1, 1),
		)
		
	def forward(self, x):
		out = self.body(x)
		out = F.relu(out + x)
		return out
		
		
class EResidualBlock(nn.Module):
	def __init__(self, 
				 in_channels, out_channels,
				 group=1):
		super(EResidualBlock, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 1, 1, 0),
		)
		
	def forward(self, x):
		out = self.body(x)
		out = F.relu(out + x)
		return out
		
		
class CALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CALayer, self).__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d(1)

		self.c1 = BasicBlock(channel , channel // reduction, 1, 1, 0)
		self.c2 = BasicBlockSig(channel // reduction, channel , 1, 1, 0)

	def forward(self, x):
		y = self.avg_pool(x)
		y1 = self.c1(y)
		y2 = self.c2(y1)
		return x * y2


class Block(nn.Module):
	def __init__(self, in_channels, out_channels, group=1):
		super(Block, self).__init__()

		self.r1 = Merge_Run_dual(in_channels, out_channels)
		self.r2 = ResidualBlock(in_channels, out_channels)
		self.r3 = EResidualBlock(in_channels, out_channels)
		#self.g = BasicBlock(in_channels, out_channels, 1, 1, 0)
		self.ca = CALayer(in_channels)

	def forward(self, x):
		
		r1 = self.r1(x)			
		r2 = self.r2(r1)	   
		r3 = self.r3(r2)
		#g = self.g(r3)
		out = self.ca(r3)

		return out		


from .baseNet import BaseNet
class RIDNet_G(BaseNet):
	def __init__(self, name='ridnet_g'):
		super(RIDNet_G, self).__init__(name)
		n_feats = 64
		kernel_size = 3
		
		self.head = nn.Conv2d(3, n_feats, kernel_size, 1, 1, 1)

		self.b1 = Block(n_feats, n_feats)
		self.b2 = Block(n_feats, n_feats)
		self.b3 = Block(n_feats, n_feats)
		self.b4 = Block(n_feats, n_feats)

		self.tail = nn.Conv2d(n_feats, 3, kernel_size, 1, 1, 1)

	def forward(self, x):
		_h = self.head(x)

		h = F.relu(_h)
		b1 = self.b1(h)
		b2 = self.b2(b1)
		b3 = self.b3(b2)
		b_out = self.b4(b3)

		res = self.tail(b_out + _h)
		f_out = res + x
		return f_out