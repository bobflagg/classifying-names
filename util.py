import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.long().data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

def show_accuracy(model, test_acc, train_acc):
    plt.figure()
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.title("Train vs Test Accuracy")

def show_confusion(model, loader, languages, seq_len):

    train_on_gpu = torch.cuda.is_available()
    num_classes = len(languages)

    confusion = torch.zeros(num_classes, num_classes)
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if (train_on_gpu): inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.view(-1, seq_len, num_chars)
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
    ax.set_xticklabels([''] + languages, rotation=90)
    ax.set_yticklabels([''] + languages)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

"""
In this notebook I'll show how to classifying names in PyTorch with a vanilla [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network).

and an [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) in PyTorch.


"""