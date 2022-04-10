import os
from sklearn import model_selection
from torch.utils.data import DataLoader
from datasets.physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from tools import utils


def get_dataset(opt):
    # Use custom collate_fn to combine samples with arbitrary time observations.
	# Returns the dataset along with mask and time steps
    train_dataset_obj = PhysioNet('data', train=True, quantization = opt.quantization, download=True, 
                              n_samples=min(10000, opt.n_samples), device=opt.device)
    test_dataset_obj = PhysioNet('data', train=False, quantization = opt.quantization, download=True, 
                              n_samples=min(10000, opt.n_samples), device=opt.device)
    
    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]
    
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
		                 	random_state = 42, shuffle = True)
    record_id, tt, vals, mask, labels = train_data[0]
    n_samples = len(total_dataset)
    input_dim = vals.size(-1)
    batch_size = min(min(len(train_dataset_obj), opt.batch_size), opt.n_samples)
    data_min, data_max = get_data_min_max(total_dataset)
    train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
                                  collate_fn= lambda batch: variable_time_collate_fn(batch, opt, opt.device, data_type = "train",
				                  data_min = data_min, data_max = data_max))
    test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			                     collate_fn= lambda batch: variable_time_collate_fn(batch, opt, opt.device, data_type = "test",
			                 	 data_min = data_min, data_max = data_max))
    attr_names = train_dataset_obj.params
    data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional
    return data_objects
    
    