import os
import torch
import argparse
import tarfile
from tqdm import tqdm
import numpy as np
import random
from torchvision.datasets.utils import download_url


def process_dataset(root, download=True, quantization = 0.016):
    # define download parameters
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
        # 'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download'
    ]

    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt',
        # 'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt',
        # 'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt'
    ]

    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
    labels_dict = {k: i for i, k in enumerate(labels)}
    
    # define folder path
    raw_folder = os.path.join(root, 'PhysioNet', 'raw')
    processed_folder = os.path.join(root, 'PhysioNet', 'processed')
    reduce = "average"

    def download_dataset():
        if _check_exists():
            print('Downloading data has been finished. Skip this step!!!')
            return
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
        os.makedirs(os.path.join(processed_folder, 'quantization_' + str(quantization)), exist_ok=True)

        # Download outcome data
        for url in outcome_urls:
            filename = url.rpartition('/')[2]
            txtfile = os.path.join(raw_folder, filename)
            if not os.path.exists(txtfile):
                print(txtfile, 'not found')
                # download_url(url, raw_folder, filename, None)

            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels)

                torch.save(
                    labels,
                    os.path.join(processed_folder, filename.split('.')[0] + '.pt')
                )

        for url in urls:
            filename = url.rpartition('/')[2]
            dirname = os.path.join(raw_folder, filename.split('.')[0])

            if not os.path.exists(dirname):
                print(dirname, 'not found')
                # download_url(url, raw_folder, filename, None)
                # tar = tarfile.open(os.path.join(raw_folder, filename), "r:gz")
                # tar.extractall(raw_folder)
                # tar.close()

            print('Processing {}...'.format(filename))

            patients = []
            total = 0
            for txtfile in tqdm(os.listdir(dirname)):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(params))]
                    mask = [torch.zeros(len(params))]
                    nobs = [torch.zeros(len(params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        time = round(time / quantization) * quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(params)))
                            mask.append(torch.zeros(len(params)))
                            nobs.append(torch.zeros(len(params)))
                            prev_time = time

                        if param in params_dict:
                            #vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][params_dict[param]]
                            if reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][params_dict[param]] = new_val
                            else:
                                vals[-1][params_dict[param]] = float(val)
                            mask[-1][params_dict[param]] = 1
                            nobs[-1][params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                tt = torch.tensor(tt)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]

                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(processed_folder, 'quantization_' + str(quantization),
                    filename.split('.')[0] + "_" + str(quantization) + '.pt')
            )
                
        print('Done!')

    def _check_exists():
        for url in urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                os.path.join(processed_folder, 'quantization_' + str(quantization),
                    filename.split('.')[0] + "_" + str(quantization) + '.pt')
            ):
                return False
        return True
    
    # download dataset
    if download:
        download_dataset()
    return


def sample_dataset(load_path):
    # load_path: data/PhysioNet/processed/quantization_{}/set-a_0.016.pt
    # prepare save path
    path_lists = load_path.split('/')
    path_lists[-1] = "sample-" + path_lists[-1].split('-')[1]
    save_path = "/".join(path_lists)

    # check exist:
    if os.path.exists(save_path):
        print("Sampling data has been finished! Skip this step!!!")
        return save_path
    
    # load data
    data = torch.load(load_path)
    
    # shuffle data
    random.seed(1234)         # fix the seed
    random.shuffle(data)
    labels = []
    for i in range(len(data)):
        label = data[i][-1]
        labels.append(label.unsqueeze(dim=0))
    labels = torch.cat(labels, dim=0)
    
    # sample labels to balance the dataset
    pos_labels = labels[labels == 1]
    neg_labels = labels[labels == 0]

    pos_length = len(pos_labels)
    neg_length = len(neg_labels)

    length = min(pos_length, neg_length)
    
    # classify the data
    pos_data = []
    neg_data = []
    for i in range(len(data)):
        sample = data[i]
        label = sample[-1].item()
        if label == 0.0:
            if len(neg_data) < length:
                neg_data.append(sample)
        elif label == 1.0:
            if len(pos_data) < length:
                pos_data.append(sample)
        else:
            print("Warnning! Unknown label")
    
    # save pos_data and neg_data
    torch.save((pos_data, neg_data), save_path)
    return save_path


def split_dataset(load_path):
    # load_path: data/PhysioNet/processed/quantization_{}/sample-a_0.016.pt
    # prepare save path
    pos_data, neg_data = torch.load(load_path)
    # splits = ['train', 'valid', 'test']

    path_lists = load_path.split('/')
    path_lists.pop()
    save_folder = "/".join(path_lists)
    train_path = os.path.join(save_folder, "train.pt")
    valid_path = os.path.join(save_folder, "valid.pt")
    test_path = os.path.join(save_folder, "test.pt")

    # check exist:
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        print("Splitting data has been finished! Skip this step!!!")
        return save_folder
    
    # split dataset
    # train:valid:test = 7:1:2
    length = len(pos_data)
    valid_length = int(length / 10)
    test_length = int(length / 10 * 2)
    train_length = length - valid_length - test_length
    
    random.seed(1234)         # fix the seed
    # train split
    train_data = pos_data[0:train_length]
    train_data.extend(neg_data[0:train_length])
    random.shuffle(train_data)
    
    # valid split
    valid_data = pos_data[train_length:train_length+valid_length]
    valid_data.extend(neg_data[train_length:train_length+valid_length])
    random.shuffle(valid_data)

    # test split
    test_data = pos_data[train_length+valid_length:]
    test_data.extend(neg_data[train_length+valid_length:])
    random.shuffle(test_data)

    # save data
    torch.save(train_data, train_path)
    torch.save(valid_data, valid_path)
    torch.save(test_data, test_path)
    return save_folder


if __name__ == "__main__":
    # download dataset
    # define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", help="path to data folder")
    parser.add_argument("--quantization", type=float, default=0.016, help="value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
    parser.add_argument("--download", action="store_true", help="download the physionnet dataset")
    args = parser.parse_args()
    # download and process data
    process_dataset(root = args.root, quantization = args.quantization, download=args.download)
    
    # sample data and balance dataset
    load_path = os.path.join("data/PhysioNet", "processed", "quantization_" + str(args.quantization), "set-a_" + str(args.quantization) + ".pt")
    save_path = sample_dataset(load_path)
    
    # split dataset into train/valid/test
    save_folder = split_dataset(load_path=save_path)

    # # check processed data files
    # splits = ['train', 'valid', 'test']
    # for split in splits:
    #     split_path = os.path.join(save_folder, split + '.pt')
    #     data = torch.load(split_path)
    #     print(len(data))
    #     labels = []
    #     for i in range(len(data)):
    #         label = data[i][-1]
    #         labels.append(label.unsqueeze(dim=0))
    #     labels = torch.cat(labels, dim=0)
    #     print(labels.sum())
