from easydict import EasyDict
import os

def get_opt(model='GRUmodel', lr=1e-3):
    # set experiment configs
    opt = EasyDict()
    # choose a model from ["GRUmodel", "LSTMmodel", "probGRU", "probLSTM", "Transformermodel", "probTransformer"]
    opt.model = model # 'ADDA_RNN'
    # choose run on which device ["cuda", "cpu"]
    opt.device = "cuda"
    # set random seed
    opt.seed = 1234

    # hyper-parameters for dataset
    opt.quantization = 0.016 # value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min
    opt.extrap = False       # Set extrapolation mode. If this flag is not set, run interpolation mode
    opt.sample_tp = None     # Number of time points to sub-sample
    opt.cut_tp = None        # Cut out the section of the timeline of the specified length (in number of points)
    opt.n_samples = 8000     # size of the dataset
    opt.batch_size = 50      # batch size
    
    # hyper-parameters for model architecture
    opt.input_size = 41     # dimension of input dim
    opt.output_size = 1     # dimension of output dim
    
    # hyper-parameters for specific model
    if opt.model == 'GRUmodel':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'probGRU':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'LSTMmodel':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'probLSTM':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'Transformermodel':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'probTransformer':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    
    # hyper-parameters for training
    opt.lr = lr             # learning rate
    opt.weight_decay = 5e-4
    opt.beta = 0.9
    opt.epochs = 20

    # hyper-parameters for SGHMC sampling
    opt.n_sghmc = 12         # number of SGHMC samples: 8
    opt.sghmc_alpha = 0.01   # noise alpha for SGHMC sampling

    # experiment folder
    opt.exp = 'SeqExp_' + opt.model
    opt.outf = './dump/' + opt.exp + '_seed_' + str(opt.seed)
    opt.train_log = opt.outf + '/train.log'
    opt.model_path = opt.outf + '/model.pth'

    os.system('mkdir -p ' + opt.outf)
    print('Training result will be saved in ', opt.outf)

    return opt
