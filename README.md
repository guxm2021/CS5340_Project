# CS5340_Project
Group Project for Class CS5340

## config
opt_dict.py: define the config

## data
save our processed data (don't need to be in github, only in local machine)

## datasets
files to process the data and build dataset class

physionnet.py implements data processing for PhysionNet 2012 dataset: [Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012](https://ieeexplore.ieee.org/abstract/document/6420376)

## model
files to implement model architectures

* RNN.py: implement model classes of GRUmodel/LSTMmodel and their Bayesian versions probGRU/probLSTM [Long Short-Term Memory](https://ieeexplore.ieee.org/abstract/document/6795963) and [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf?ref=hackernoon.com)

* Transformer.py: implement model class of Transformermodel and probTransformer [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

* TCN.py: implement model class of TCNmodel and probTCN [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf)

* NeuralODE.py: implement model class of ODERNNmodel and probODERNN [Latent ODEs for Irregularly-Sampled Time Series](https://papers.nips.cc/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf)

## tools
* utils.py: utility functions, especially for datasets/physionnet.py

* train.py: implement function of Trainer, used for training models

* test.py: implement function of Tester, used for evaluating models

* sghmc.py: implement function of SGHMC, used for [Stochastic Gradient Hamiltonian Monte Carlo](https://proceedings.mlr.press/v32/cheni14.pdf)

## dump
save for training log and saved model files

## train and evaluate

### Determinstic Models
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
