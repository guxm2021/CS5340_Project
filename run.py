import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'   # for RNN reproducibility
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from config.opt_dict import get_opt
from model.pools import get_model
from datasets.parse_dataset import get_dataset
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
	data_obj = get_dataset(opt) 
	train_loader = data_obj['train_dataloader']
	test_loader = data_obj['test_dataloader']

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
    
	# set optimizer and criterion
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta, 0.999), weight_decay=opt.weight_decay)
	criterion = nn.BCELoss()

	# train
	model = Trainer(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, opt=opt, skip=args.skip)

	# load the best model
	print('load model from {}'.format(opt.model_path))
	model.load_state_dict(torch.load(opt.model_path))

    # evaluate model
	Tester(model=model, dataloader=test_loader, opt=opt, valid=False)


if __name__ == '__main__':
	# define argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="GRUmodel", help="the name of model")
	parser.add_argument("--gpu", type=int, default=3, help="the num of gpu, chosen from [0, 1, 2, 3]")
	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--skip", action="store_true", help="skipping the training stage")
	args = parser.parse_args()
    
	# set gpu num
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	main(args)