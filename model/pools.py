from model.RNN import GRUmodel, LSTMmodel, probGRU, probLSTM
from model.Transformer import Transformermodel, probTransformer
from model.TCN import TCNmodel, probTCN


def get_model(model):
    model_pool = {
        'GRUmodel': GRUmodel,
        'LSTMmodel': LSTMmodel,
        'probGRU': probGRU,
        'probLSTM': probLSTM,
        'Transformermodel': Transformermodel,
        'probTransformer': probTransformer,
        'TCNmodel': TCNmodel,
        'probTCN': probTCN,
    }
    return model_pool[model]