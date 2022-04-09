
def Tester(model, dataloader, opt, valid=False):
    train_log = opt.train_log
    device = opt.device

    model.eval()
    total = 0
    correct = 0
    for seq, labels in dataloader:
        seq, labels = seq.to(device), labels.to(device)
        # forward
        logits = model(seq)
        # statistic
        pred = logits.argmax(dim=1)
        correct += (pred == labels)
        total += pred.shape[0]
    acc = correct / total
    if not valid:
        acc_msg = '[Test] Accuracy: total average on Test dataset: {:.2f}'.format(acc * 100)
        print(acc_msg)
        with open(train_log, 'a') as f:
            f.write(acc_msg + "\n")
    return acc
        