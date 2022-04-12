from model.RNN import GRUmodel, LSTMmodel, probGRU, probLSTM
from model.Transformer import Transformermodel, probTransformer
from model.TCN import TCNmodel, probTCN
from model.NODE import ODERNNmodel, probODERNN


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
        'ODERNNmodel': ODERNNmodel,
        'probODERNN': probODERNN,
    }
    return model_pool[model]