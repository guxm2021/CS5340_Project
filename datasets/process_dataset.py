import os
import torch
import argparse
import tarfile
from tqdm import tqdm
import numpy as np
from torchvision.datasets.utils import download_url


def process_dataset(root, download=True, quantization = 0.016):
    # define download parameters
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

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
            return
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)

        # Download outcome data
        for url in outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, raw_folder, filename, None)

            txtfile = os.path.join(raw_folder, filename)
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
            download_url(url, raw_folder, filename, None)
            tar = tarfile.open(os.path.join(raw_folder, filename), "r:gz")
            tar.extractall(raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(raw_folder, filename.split('.')[0])
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
                os.path.join(processed_folder, 
                    filename.split('.')[0] + "_" + str(quantization) + '.pt')
            )
                
        print('Done!')

    def _check_exists():
        for url in urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                os.path.join(processed_folder, 
                    filename.split('.')[0] + "_" + str(quantization) + '.pt')
            ):
                return False
        return True
    
    # download dataset
    if download:
        download_dataset()
    return

    
if __name__ == "__main__":
    # download dataset
    # define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", help="path to data folder")
    parser.add_argument("--quantization", type=float, default=0.016, help="value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
    parser.add_argument("--download", action="store_true", help="download the physionnet dataset")
    args = parser.parse_args()
    process_dataset(root = args.root, quantization = args.quantization, download=args.download)