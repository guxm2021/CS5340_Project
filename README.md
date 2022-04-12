# CS5340_Project
Implementation of Group Project for Class CS5340

## Prerequisite
Create new conda environment and install following packages:
* Python 3.8.12
* Pytorch 1.9.0
* CUDA 11.1
* scikit-learn 1.0.2
* numpy 1.22.2
* matplotlib 3.5.1
* easydict 1.9
* tqdm 4.62.3

## Directory
The `${root}` is described as below.
> ${root}\
| -- config: define the configuration\
| -- data: save the dataset\
| -- dataset: process the data and build data class\
| -- dump: save the trained models\
| -- model: define the models\
| -- tools: define the utility functions\
| -- run.py: run the experiments

## Dataset
PhysionNet 2012 dataset: [Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012](https://ieeexplore.ieee.org/abstract/document/6420376)

To download and process dataset, run following command:
```
python datasets/process_dataset.py --root data --quantization 0.016 --download
```

The dataset will be splitted into train/valid/test split of proportion 7:1:2.

## Model
Implement different sequential deep learning model, defined in the folder `model`:

* RNN.py: Implement model classes of GRUmodel/LSTMmodel and their Bayesian versions probGRU/probLSTM. [Long Short-Term Memory](https://ieeexplore.ieee.org/abstract/document/6795963) and [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf?ref=hackernoon.com).

* Transformer.py: Implement model class of Transformermodel and probTransformer. [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

* TCN.py: Implement model class of TCNmodel and probTCN. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf).

* NeuralODE.py: Implement model class of ODERNNmodel and probODERNN. [Latent ODEs for Irregularly-Sampled Time Series](https://papers.nips.cc/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf).


## Experiments

### Determinstic Models
Training and Evaluation are implemented in `tools/train.py` and `tools/test.py`.

* GRU
```
python run.py --gpu 0 --model GRUmodel --lr 1e-3
```

* LSTM
```
python run.py --gpu 0 --model LSTMmodel --lr 1e-3
```

* TCN
```
python run.py --gpu 0 --model TCNmodel --lr 1e-3
```

* Transformer
```
python run.py --gpu 0 --model Transformermodel --lr 1e-3
```

* ODE RNN
```
python run.py --gpu 0 --model ODERNNmodel --lr 1e-3
```

### Bayesian Models
[Stochastic Gradient Hamiltonian Monte Carlo](https://proceedings.mlr.press/v32/cheni14.pdf) is implemented in `tools/sghmc.py`.

![image](https://github.com/guxm2021/CS5340_Project/blob/main/assets/SGHMC.png)

* TCN
```
python run.py --gpu 0 --model probTCN --lr 1e-3 --bayes
```

