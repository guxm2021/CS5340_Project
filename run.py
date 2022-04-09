import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'   # for RNN reproducibility
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from config.opt_dict import get_opt
from model.pools import get_model
from datasets.physionet import PhysioNet, DataLoader, variable_time_collate_fn
from tools.train import Trainer
from tools.test import Tester


def main(args):
    # load opt
	opt = get_opt(model=args.model, lr=args.lr)
	
	# set seeds for experiments
	seed = opt.seed
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# load dataset and dataloader
	train_dataset = PhysioNet('data', train=True, download=True)
	test_dataset = PhysioNet('data', train=False, download=True)
	train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=variable_time_collate_fn)
	# print(dataloader.__iter__().next())

    # load model
	modelClass = get_model(opt.model)
	model = modelClass(opt)
	model.to(opt.device)

	# prepare training log
	with open(opt.train_log, 'a') as f:
		f.write(str(opt))
		f.write("\n")

	# count model parameters
	num_param = 0
	for param in model.parameters():
		num_param += param.numel()
	print(f"model {opt.model} parameters {round(num_param / 1e6, 2)} M")
    
	# set optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta, 0.999), weight_decay=opt.weight_decay)

	# train
	model = Trainer(model=model, dataloader=train_loader, optimizer=optimizer, opt=opt, skip=False)

	# load the best model
	print('load model from {}'.format(opt.model_path))
	model.load_state_dict(torch.load(opt.model_path))

    # evaluate model
	Tester(model=model, dataloader=test_loader, opt=opt, valid=False)


if __name__ == '__main__':
	# define argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="SO_RNN", help="the name of model")
	parser.add_argument("--gpu", type=int, default=0, help="the num of gpu, chosen from [0, 1, 2, 3]")
	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	args = parser.parse_args()
    
	# set gpu num
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	main(args)