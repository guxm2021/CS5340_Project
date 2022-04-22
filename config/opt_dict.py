from easydict import EasyDict
import os

def get_opt(seed=2233, model='GRUmodel', lr=1e-3, quantization=0.1, n_sghmc=8, alpha=0.1, lambda_noise=0.01, bayes=False):
    # set experiment configs
    opt = EasyDict()
    # choose a model from ["GRUmodel", "LSTMmodel", "probGRU", "probLSTM", "Transformermodel", "probTransformer", "TCNmodel", "probTCN"]
    opt.model = model # 'ADDA_RNN'
    # choose run on which device ["cuda", "cpu"]
    opt.device = "cuda"
    # set random seed
    opt.seed = seed

    # hyper-parameters for dataset
    opt.quantization = quantization # value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min
    opt.extrap = False       # Set extrapolation mode. If this flag is not set, run interpolation mode
    opt.sample_tp = None     # Number of time points to sub-sample
    opt.cut_tp = None        # Cut out the section of the timeline of the specified length (in number of points)
    opt.data_folder = "data"
    
    # hyper-parameters for model architecture
    opt.input_size = 41     # dimension of input dim
    opt.output_size = 2     # dimension of output dim
    opt.cat_mask = True    # whether to use mask
    opt.cat_tp = True      # whether to use time points
    
    # hyper-parameters for specific model
    if opt.model == 'GRUmodel':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'LSTMmodel':
        opt.hidden_size = 54  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'Transformermodel':
        opt.hidden_size = 100  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'TCNmodel':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
        opt.kernel_size = 7     # kernel size 
    elif opt.model == 'ODERNNmodel':
        opt.hidden_size = 196  # hidden dimension
        opt.num_layers = 2      # number of layers
        opt.n_step = 4          # steps for ode solver
    elif opt.model == 'probGRU1':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'probGRU2':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    elif opt.model == 'probGRU3':
        opt.hidden_size = 64  # hidden dimension
        opt.num_layers = 2      # number of layers
    
    # hyper-parameters for training
    opt.lr = lr # 0.005            # learning rate
    opt.weight_decay = 5e-4
    opt.beta = 0.9
    opt.epochs = 50
    opt.batch_size = 32      # batch size

    # hyper-parameters for SGHMC sampling
    opt.n_sghmc = n_sghmc # best: 4           # number of SGHMC samples
    opt.sghmc_alpha = alpha #best: 0.01       # noise alpha for SGHMC sampling
    opt.noise_loss_lambda = lambda_noise #best: 0.001 # hyper-parameter to balance likelihood and prior

    # experiment folder
    opt.exp = 'SeqExp_' + opt.model
    if bayes:
        opt.outf = './dump_uniform2/' + opt.exp + '/seed_' + str(opt.seed) + '/lr_' + str(opt.lr) + '_quantization_' \
      + str(opt.quantization) + '_samples_' + str(opt.n_sghmc) + '_alpha_' + str(opt.sghmc_alpha) \
      + '_lambda_' + str(opt.noise_loss_lambda)
    else:
        opt.outf = './dump/' + opt.exp + '/seed_' + str(opt.seed) + '/lr_' + str(opt.lr) + '_quantization_' \
      + str(opt.quantization) + '_samples_' + str(opt.n_sghmc) + '_alpha_' + str(opt.sghmc_alpha) \
      + '_lambda_' + str(opt.noise_loss_lambda)
    
    opt.train_log = opt.outf + '/train.log'
    opt.model_path = opt.outf + '/model.pth'

    os.system('mkdir -p ' + opt.outf)
    print('Training result will be saved in ', opt.outf)

    return opt
