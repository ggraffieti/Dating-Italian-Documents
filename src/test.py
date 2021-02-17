import torch
from sklearn.metrics import classification_report


def evaluate(model, test_loader, num_classes,  device):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for labels, text in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[_ for _ in range(num_classes)], digits=4))
