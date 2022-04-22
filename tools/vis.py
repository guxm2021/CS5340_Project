import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def stat1():
    models = ['probGRU1']# ['GRU', 'LSTM', 'TCN', 'Transformer']
    seeds = [2223, 2233, 2243]
    lrs = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002, 0.0002, 0.5, 0.05, 0.005, 0.0005, 0.8, 0.08, 0.008, 0.0008]
    for model in models:
        acc = np.zeros((16, 3), dtype=np.float32)
        model_folder = os.path.join('dump', 'SeqExp_'+ model)
        print(f'load model from {model_folder}')
        for i in range(len(seeds)):
            seed_folder = os.path.join(model_folder, 'seed_' + str(seeds[i]))
            for j in range(len(lrs)):
                lr_folder = os.path.join(seed_folder, 'lr_' + str(lrs[j]) + '_quantization_0.1')
                train_log = os.path.join(lr_folder, 'train.log')
                with open(train_log, 'r') as f:
                    lines = f.readlines()
                    lines = [line.split('\n')[0] for line in lines if line is not None]
                    assert len(lines) == 102
                    acc[j, i] = float(lines[-1][-6:-2])
        mean = acc.mean(axis=1)
        std = acc.std(axis=1)
        print(mean)
        print(std)
    
def stat2():
    seeds_pools = [2223, 2233, 2243]
    n_sghmc_pools = [4, 6, 8, 10, 12, 14]
    alpha_pools = [0.01, 0.05, 0.08, 0.1, 0.5, 0.8]
    lambda_noise_pools = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    root = "dump_uniform/SeqExp_probGRU1"
    acc_test = np.zeros((3, 6, 6, 6), dtype=np.float32)
    acc_valid = np.zeros((3, 6, 6, 6), dtype=np.float32)
    for seed_index in range(len(seeds_pools)):
        seed = seeds_pools[seed_index]
        for n_sghmc_index in range(len(n_sghmc_pools)):
            n_sghmc = n_sghmc_pools[n_sghmc_index]
            for alpha_index in range(len(alpha_pools)):
                alpha = alpha_pools[alpha_index]
                for lambda_noise_index in range(len(lambda_noise_pools)):
                    lambda_noise = lambda_noise_pools[lambda_noise_index]
                    folder = root + '/seed_' + str(seed) + '/lr_0.005_quantization_0.1'  + '_samples_' \
                         + str(n_sghmc) + '_alpha_' + str(alpha) + '_lambda_' + str(lambda_noise)
                    train_log = os.path.join(folder, 'train.log')
                    with open(train_log, 'r') as f:
                        lines = f.readlines()
                        lines = [line.split('\n')[0] for line in lines if line is not None]
                        assert len(lines) == 103
                        acc_test[seed_index, n_sghmc_index, alpha_index, lambda_noise_index] = float(lines[-2][-6:-2])
                        acc_valid[seed_index, n_sghmc_index, alpha_index, lambda_noise_index] = float(lines[-1][-6:-2])
    
    mean_test = acc_test.mean(axis=0)
    std_test = acc_test.std(axis=0)
    print(mean_test.max())
    pos_test = np.unravel_index(np.argmax(mean_test), mean_test.shape)
    print(pos_test)
    print(mean_test[pos_test])
    print(std_test[pos_test])

    mean_valid = acc_valid.mean(axis=0)
    std_valid = acc_valid.std(axis=0)
    print(mean_valid[pos_test])
    print(std_valid[pos_test])


    print(mean_valid.max())
    pos_valid = np.unravel_index(np.argmax(mean_valid), mean_valid.shape)
    print(pos_valid)
    print(std_valid[pos_valid])

    # plot n
    mean_n_test = mean_test[:, 0, 0]
    std_n_test = std_test[:, 0, 0]
    mean_n_valid = mean_valid[:, 0, 0]
    std_n_valid = std_valid[:, 0, 0]
    plt.figure(1)
    plt.style.use('seaborn')
    # plt.errorbar(np.array(n_sghmc_pools), mean_n_test, yerr=std_n_test, label='Test')
    # plt.errorbar(np.array(n_sghmc_pools), mean_n_valid, yerr=std_n_valid, label='Valid')
    plt.plot(np.array(n_sghmc_pools), mean_n_test, label='Test')
    plt.plot(np.array(n_sghmc_pools), mean_n_valid, label='Valid')
    plt.fill_between(np.array(n_sghmc_pools), mean_n_test-std_n_test, mean_n_test+std_n_test, alpha=0.3)
    plt.fill_between(np.array(n_sghmc_pools), mean_n_valid-std_n_valid, mean_n_valid+std_n_valid, alpha=0.3)
    plt.title('alpha=0.01, lambda=0.001')
    plt.xlabel('n')
    plt.ylabel('Acc (%)')
    plt.ylim([50, 80])
    plt.legend()
    plt.savefig('fig1.pdf')

    # alpha
    mean_alpha_test = mean_test[0, :, 0]
    std_alpha_test = std_test[0, :, 0]
    mean_alpha_valid = mean_valid[0, :, 0]
    std_alpha_valid = std_valid[0, :, 0]
    plt.figure(2)
    plt.style.use('seaborn')
    # plt.errorbar(np.array(alpha_pools), mean_alpha_test, yerr=std_alpha_test, label='Test')
    # plt.errorbar(np.array(alpha_pools), mean_alpha_valid, yerr=std_alpha_valid, label='Valid')
    plt.plot(np.array(alpha_pools), mean_alpha_test, label='Test')
    plt.plot(np.array(alpha_pools), mean_alpha_valid, label='Valid')
    plt.fill_between(np.array(alpha_pools), mean_alpha_test-std_alpha_test, mean_alpha_test+std_alpha_test, alpha=0.3)
    plt.fill_between(np.array(alpha_pools), mean_alpha_valid-std_alpha_valid, mean_alpha_valid+std_alpha_valid, alpha=0.3)
    plt.xlabel("alpha")
    plt.ylabel('Acc (%)')
    plt.title('n=4, lambda=0.001')
    plt.ylim([50, 80])
    plt.legend()
    plt.savefig('fig2.pdf')

    # lambda
    mean_lambda_test = mean_test[0, 0, :]
    std_lambda_test = std_test[0, 0, :]
    mean_lambda_valid = mean_valid[0, 0, :]
    std_lambda_valid = std_valid[0, 0, :]
    plt.figure(3)
    plt.style.use('seaborn')
    # plt.errorbar(np.array(lambda_noise_pools), mean_lambda_test, yerr=std_lambda_test, label='Test')
    # plt.errorbar(np.array(lambda_noise_pools), mean_lambda_valid, yerr=std_lambda_valid, label='Valid')
    plt.plot(np.array(lambda_noise_pools), mean_lambda_test, label='Test')
    plt.plot(np.array(lambda_noise_pools), mean_lambda_valid, label='Valid')
    plt.fill_between(np.array(lambda_noise_pools), mean_lambda_test-std_lambda_test, mean_lambda_test+std_lambda_test, alpha=0.3)
    plt.fill_between(np.array(lambda_noise_pools), mean_lambda_valid-std_lambda_valid, mean_lambda_valid+std_lambda_valid, alpha=0.3)
    plt.xlabel("lambda")
    plt.ylabel('Acc (%)')
    plt.title('n=4, alpha=0.01')
    plt.ylim([50, 80])
    plt.legend()
    plt.savefig('fig3.pdf')
    
    # n for valid
    mean_n_test = mean_test[:, 1, 3]
    std_n_test = std_test[:, 1, 3]
    mean_n_valid = mean_valid[:, 1, 3]
    std_n_valid = std_valid[:, 1, 3]
    plt.figure(4)
    plt.style.use('seaborn')
    # plt.errorbar(np.array(n_sghmc_pools), mean_n_test, yerr=std_n_test, label='Test')
    # plt.errorbar(np.array(n_sghmc_pools), mean_n_valid, yerr=std_n_valid, label='Valid')
    plt.plot(np.array(n_sghmc_pools), mean_n_test, label='Test')
    plt.plot(np.array(n_sghmc_pools), mean_n_valid, label='Valid')
    plt.fill_between(np.array(n_sghmc_pools), mean_n_test-std_n_test, mean_n_test+std_n_test, alpha=0.3)
    plt.fill_between(np.array(n_sghmc_pools), mean_n_valid-std_n_valid, mean_n_valid+std_n_valid, alpha=0.3)
    plt.title('alpha=0.05, lambda=0.05')
    plt.xlabel('n')
    plt.ylabel('Acc (%)')
    plt.ylim([50, 80])
    plt.legend()
    plt.savefig('fig4.pdf')

    


def curve():
    plt.figure(1)
    plt.style.use('seaborn')
    seed=2243
    # GRU
    model='GRU'
    lr=0.005
    model_folder = os.path.join('dump', 'SeqExp_'+ model + 'model')
    seed_folder = os.path.join(model_folder, 'seed_' + str(seed))
    lr_folder = os.path.join(seed_folder, 'lr_' + str(lr) + '_quantization_0.1')
    train_log = os.path.join(lr_folder, 'train.log')
    with open(train_log, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if line is not None]
        assert len(lines) == 102
    f.close()
    losses = []
    for i in range(50):
        line = lines[i*2+1]
        loss = line.split(":")[1][1:6]
        if loss == 'nan. ':
            break
        loss = float(loss)
        losses.append(loss)
    losses = np.array(losses)
    times = np.arange(1, 51)[:len(losses)]
    plt.plot(times, losses, label='GRU')

    # LSTM
    model='LSTM'
    lr=0.0001
    model_folder = os.path.join('dump', 'SeqExp_'+ model + 'model')
    seed_folder = os.path.join(model_folder, 'seed_' + str(seed))
    lr_folder = os.path.join(seed_folder, 'lr_' + str(lr) + '_quantization_0.1')
    train_log = os.path.join(lr_folder, 'train.log')
    with open(train_log, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if line is not None]
        assert len(lines) == 102
    f.close()
    losses = []
    for i in range(50):
        line = lines[i*2+1]
        loss = line.split(":")[1][1:6]
        if loss == 'nan. ':
            break
        loss = float(loss)
        losses.append(loss)
    losses = np.array(losses)
    times = np.arange(1, 51)[:len(losses)]
    plt.plot(times, losses, label='LSTM')

    # TCN
    model='TCN'
    lr=0.002
    model_folder = os.path.join('dump', 'SeqExp_'+ model + 'model')
    seed_folder = os.path.join(model_folder, 'seed_' + str(seed))
    lr_folder = os.path.join(seed_folder, 'lr_' + str(lr) + '_quantization_0.1')
    train_log = os.path.join(lr_folder, 'train.log')
    with open(train_log, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if line is not None]
        assert len(lines) == 102
    f.close()
    losses = []
    for i in range(50):
        line = lines[i*2+1]
        loss = line.split(":")[1][1:6]
        if loss == 'nan. ':
            break
        loss = float(loss)
        losses.append(loss)
    losses = np.array(losses)
    times = np.arange(1, 51)[:len(losses)]
    plt.plot(times, losses, label='TCN')
    
    # Transformer
    model='Transformer'
    lr=0.0005
    model_folder = os.path.join('dump', 'SeqExp_'+ model + 'model')
    seed_folder = os.path.join(model_folder, 'seed_' + str(seed))
    lr_folder = os.path.join(seed_folder, 'lr_' + str(lr) + '_quantization_0.1')
    train_log = os.path.join(lr_folder, 'train.log')
    with open(train_log, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if line is not None]
        assert len(lines) == 102
    f.close()
    losses = []
    for i in range(50):
        line = lines[i*2+1]
        loss = line.split(":")[1][1:6]
        if loss == 'nan. ':
            break
        loss = float(loss)
        losses.append(loss)
    losses = np.array(losses)
    times = np.arange(1, 51)[:len(losses)]
    plt.plot(times, losses, label='Transformer')

    # ODERNN
    model='ODERNN'
    lr=0.0005
    model_folder = os.path.join('dump', 'SeqExp_'+ model + 'model')
    seed_folder = os.path.join(model_folder, 'seed_' + str(seed))
    lr_folder = os.path.join(seed_folder, 'lr_' + str(lr) + '_quantization_0.1')
    train_log = os.path.join(lr_folder, 'train.log')
    with open(train_log, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\n')[0] for line in lines if line is not None]
        assert len(lines) == 102
    f.close()
    losses = []
    for i in range(50):
        line = lines[i*2+1]
        loss = line.split(":")[1][1:6]
        if loss == 'nan. ':
            break
        loss = float(loss)
        losses.append(loss)
    losses = np.array(losses)
    times = np.arange(1, 51)[:len(losses)]
    plt.plot(times, losses, label='ODERNN')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_seed{seed}.pdf')

if __name__ == "__main__":
    stat1()
    stat2()
    curve()