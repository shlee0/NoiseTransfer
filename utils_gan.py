import torch
import torch.nn as nn

def get_loss_adv(logits, target):
	targets = torch.full_like(logits, target)
	return nn.functional.binary_cross_entropy_with_logits(logits, targets)

def get_loss_adv_d(disc_real_output, disc_generated_output):
	loss = get_loss_adv(disc_real_output, 1.0) + get_loss_adv(disc_generated_output, 0.0)
	return loss / 2

def get_loss_adv_g(disc_generated_output):
	loss = get_loss_adv(disc_generated_output, 1.0)
	return loss