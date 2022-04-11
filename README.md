# CS5340_Project
Group Project for Class CS5340

## config
opt_dict.py: define the config

## data
save our processed data (don't need to be in github, only in local machine)

## datasets
files to process the data and build dataset class

physionnet.py is borrowed from https://github.com/YuliaRubanova/latent_ode/blob/master/physionet.py

## model
files to implement model architectures

* RNN.py: implement model classes of GRUmodel/LSTMmodel and their Bayesian versions probGRU/probLSTM

* Transformer.py: implement model class of Transformermodel and probTransformer

* TCN.py: implement model class of TCNmodel and probTCN

* NeuralODE.py: implement model class of ODERNNmodel and probODERNN

## tools
* utils.py: utility functions, especially for datasets/physionnet.py

* train.py: implement function of Trainer, used for training models

* test.py: implement function of Tester, used for evaluating models

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
