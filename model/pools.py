from model.RNN import GRUmodel, LSTMmodel, probGRU, probLSTM
from model.Transformer import Transformermodel, probTransformer


def get_model(model):
    model_pool = {
        'GRUmodel': GRUmodel,
        'LSTMmodel': LSTMmodel,
        'probGRU': probGRU,
        'probLSTM': probLSTM,
        'Transformermodel': Transformermodel,
        'probTransformer': probTransformer,
    }
    return model_pool[model]