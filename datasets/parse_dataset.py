import os
import torch
import tools.utils as utils
from torch.utils.data import DataLoader


# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:,i][mask[:,i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max


def get_dataset(opt):
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    train_dataset = PhysioNet(opt=opt, split='train')
    valid_dataset = PhysioNet(opt=opt, split='valid')
    test_dataset = PhysioNet(opt=opt, split='test')
    
    
    # Combine and shuffle samples from physionet Train and physionet Test
    batch_size = min(len(train_dataset), opt.batch_size)
    data_min, data_max = get_data_min_max(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False, 
                                  collate_fn= lambda batch: variable_time_collate_fn(batch, opt, data_type = "train",
                                  data_min = data_min, data_max = data_max))
    valid_dataloader = DataLoader(valid_dataset, batch_size= batch_size, shuffle=False, 
                                  collate_fn= lambda batch: variable_time_collate_fn(batch, opt, data_type = "valid",
                                  data_min = data_min, data_max = data_max))
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, 
                                 collate_fn= lambda batch: variable_time_collate_fn(batch, opt, data_type = "test",
                                  data_min = data_min, data_max = data_max))
    return train_dataloader, valid_dataloader, test_dataloader


class PhysioNet(object):
    def __init__(self, opt, split):
        # root: data/PhysioNet/processed/quantization_{}
        self.opt = opt
        self.root = opt.data_folder
        self.split = split
        self.data_path = os.path.join(self.processed_folder, 'quantization_' + str(opt.quantization), 'all', self.split + ".pt")
        # check exists
        if os.path.exists(self.data_path):
            self.data = torch.load(self.data_path)
        else:
            print("Warnning! Please run python datasets/process_dataset.py firstly")

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def variable_time_collate_fn(batch, args, data_type = "train", data_min = None, data_max = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D])
    combined_mask = torch.zeros([len(batch), len(combined_tt), D])
    
    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt
        vals = vals
        mask = mask
        if labels is not None:
            labels = labels

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, att_min = data_min, att_max = data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)
        
    data_dict = {
        "data": combined_vals, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict
