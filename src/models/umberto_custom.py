import torch.nn as nn
import torch
from transformers import CamembertModel


class UmbertoCustom(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(UmbertoCustom, self).__init__()
        self.encoder = CamembertModel.from_pretrained(bert_model)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, num_classes)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, text, label):
        encodings = self.encoder(text, output_hidden_states=True)
        last4 = torch.stack(encodings[2][-1:], dim=0)
        sum_last4 = last4[..., 0, :].sum(dim=0)
        logits = self.classifier(sum_last4)
        loss = self.loss(logits, label)
        # secndo_to_last = encodings[2][10]
        # embedding = secndo_to_last[:, 0, :]
        # logits = self.classifier(embedding)
        # loss = self.loss(logits, label)
        return loss, logits
