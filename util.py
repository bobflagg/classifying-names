import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, test_loader, batch_size, seq_len, num_chars, num_classes, epochs=10, lr=0.001):
    train_on_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_train_acc = [1.0 / num_classes]
    all_test_acc = [1.0 / num_classes]

    for epoch in range(epochs):
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            inputs, labels = data
            if (train_on_gpu): inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.view(-1, seq_len, num_chars)

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, batch_size)

        epoch_loss = 100 * train_running_loss / i
        epoch_train_acc = train_acc / i
        all_train_acc.append(epoch_train_acc)

        model.eval()
        test_acc = 0.0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if (train_on_gpu): inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.view(-1, seq_len, num_chars)
            outputs = model(inputs)
            test_acc += get_accuracy(outputs, labels, batch_size)
        epoch_test_acc = test_acc / i
        all_test_acc.append(epoch_test_acc)
        print(
            'Epoch:  %2d | Loss: %4.2f | Train Accuracy: %.2f | Test Accuracy: %.2f' %
            (epoch + 1, epoch_loss, epoch_train_acc, epoch_test_acc)
        )
    return all_test_acc, all_train_acc


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

def show_confusion(model, loader, languages, seq_len, num_chars):

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

def predict(dataset, model, name):
    inputs = dataset.names2embeddings([name])
    model.eval()
    h = model.init_hidden()
    h = tuple([each.data for each in h])
    if torch.cuda.is_available(): inputs = inputs.cuda()
    output = model(inputs)
    values, indices = torch.topk(output, 1)
    category = dataset.codes2labels(indices.cpu().data.numpy())[0]
    return category

