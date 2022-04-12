import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'   # for RNN reproducibility
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from config.opt_dict import get_opt
from model.pools import get_model
from datasets.parse_dataset import get_dataset
from tools.train import Trainer
from tools.test import Tester
from tools.sghmc import SGHMC


def main(args):
    # load opt
	opt = get_opt(seed=args.seed, model=args.model, lr=args.lr, quantization=args.quantization)
	
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
	train_dataloader, valid_dataloader, test_dataloader = get_dataset(opt) 

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
	print(f"model {opt.model} parameters {round(num_param / 1e3, 2)} K")
    
	# set optimizer and criterion
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta, 0.999), weight_decay=opt.weight_decay)
	criterion = nn.CrossEntropyLoss()

	# train
	if args.bayes:
		print("Starting SGHMC sampling!!!")
		model = SGHMC(model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, optimizer=optimizer, 
	                  criterion=criterion, opt=opt, skip=args.skip)
	else:
		print("Starting Training!!!")
		model = Trainer(model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, optimizer=optimizer, 
	                    criterion=criterion, opt=opt, skip=args.skip)

	# load the best model
	print('load model from {}'.format(opt.model_path))
	model.load_state_dict(torch.load(opt.model_path))

    # evaluate model
	Tester(model=model, dataloader=test_dataloader, opt=opt, valid=False)


if __name__ == '__main__':
	# define argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="GRUmodel", help="the name of model")
	parser.add_argument("--gpu", type=int, default=3, help="the num of gpu, chosen from [0, 1, 2, 3]")
	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--skip", action="store_true", help="skipping the training stage")
	parser.add_argument("--bayes", action="store_true", help="whether to use SGHMC sampling")
	parser.add_argument("--seed", type=int, default=2233, help="random seed for reproducibility")
	parser.add_argument("--quantization", type=float, default=0.1, help="value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
	args = parser.parse_args()
    
	# set gpu num
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	main(args)