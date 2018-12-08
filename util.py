import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.long().data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

def show_confusion(model, loader, seq_length, num_chars, classes):
    train_on_gpu = torch.cuda.is_available()
    num_classes = len(classes)

    confusion = torch.zeros(num_classes, num_classes)
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if (train_on_gpu): inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.view(-1, seq_length, num_chars)
        logit = model(inputs)
        result = torch.max(logit, 1)[1].view(labels.size()).data
        for index in range(labels.size()[0]):
            i = int(labels[index])
            j = int(result[index])
            confusion[i][j] += 1
    # Normalize by dividing every row by its sum
    for i in range(num_classes): confusion[i] = confusion[i] / confusion[i].sum()
    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

