import torch.nn as nn
from transformers import CamembertForSequenceClassification


class Umberto(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(Umberto, self).__init__()
        self.encoder = CamembertForSequenceClassification.from_pretrained(bert_model, num_labels=num_classes)

    def forward(self, text, label):
        loss, logits = self.encoder(text, labels=label)[:2]
        return loss, logits
