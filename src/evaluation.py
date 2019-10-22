"""
Module that defines model evaluation metrics.
"""
import torch
from sklearn.metrics import f1_score
import seaborn as sn
import pandas as pd

def accuracy_topk(output, target, topk):
    """
    Computes the precision@k for the specified values of k
    Parameters:
        output(Tensor): Output of the model.
        target(Tensor): True label.
        topk(tuple): Size of k-argument.
    Returns:
        res(Tensor): Accuracy of top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(dataloader, model, topk=2):
    """
    Computes accuracy and top-k of the model.
    Parameters:
        dataloader(dict): Processed dataset.
        model(torchvision.models): Model to evaluate.
        topk(int): Size of k-argument.
    Returns:
        res(Tensor): Accuracy of top-k.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    top_k_acc = 0
    batch_count = 0

    model.eval()

    with torch.no_grad():
        for data in dataloader['test']:
            images = data['image'].to(device)
            labels = data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            top_k_acc += (accuracy_topk(outputs, torch.max(labels, 1)[1], topk=(topk,)))[0]
            total += labels.size(0)
            batch_count += 1
            correct += torch.sum(predicted == torch.max(labels, 1)[1])

    print('Accuracy of the network: {:.0f} %'.format(100 * correct / total))
    print('-' * 10)
    print('Top{} accuracy of the network: {:.0f} %'.format(
        topk, top_k_acc.cpu().numpy()[0] / batch_count))

def confusion_matrix(dataloader, model, labels):
    """
    Computes f1 score, per-class accuracy and confusion_matrix.
    Parameters:
        dataloader(dict): Processed dataset.
        model(torchvision.models): Model to evaluate.
        labels(list): List of classes names.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_classes = len(labels)

    model.eval()

    prediction_list = []
    labels_list = []
    matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data in dataloader['test']:
            images = data['image'].to(device)
            label = data['label'].to(device)
            labels_list.extend(torch.max(label, 1)[1].cpu().tolist())
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            prediction_list.extend(preds.cpu().tolist())
            for true, pred in zip(torch.max(label, 1)[1].view(-1), preds.view(-1)):
                matrix[true.long(), pred.long()] += 1

    class_acc = (matrix.diag() / matrix.sum(1)).cpu().tolist()

    print("F1 Weighted score: %.2f" % f1_score(labels_list, prediction_list, average='weighted'))
    print('-' * 10)
    print('Per class accuracy:')
    print()
    for idx, acc in enumerate(class_acc):
        print('{0}: {1:.2f} %'.format(labels[idx], acc * 100))
    print('-' * 10)


    df_cm = pd.DataFrame(matrix.numpy(), labels, labels)
    sn.set(font_scale=1.4)
    sn.heatmap(
        df_cm,
        annot=False,
        annot_kws={"size": 16},
        fmt='g',
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        cmap="Blues"
    )