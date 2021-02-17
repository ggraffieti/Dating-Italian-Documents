from torch.utils.data import Dataset
import torch


class LetterDataset(Dataset):
    def __init__(self, text_encodings, labels):
        super(LetterDataset, self).__init__()
        self.text_encodings = text_encodings
        max_l = 0
        for enc in text_encodings:
            if len(enc) > max_l:
                max_l = len(enc)
        print(max_l)
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        # first_part = self.text_encodings[idx][:256]
        # last_part = self.text_encodings[idx][len(self.text_encodings[idx]) - 256:]
        # full_enc = first_part + last_part
        # full_enc = self.text_encodings[idx][len(self.text_encodings[idx]) - 512:]
        # return self.labels[idx], torch.tensor(full_enc)

        return self.labels[idx], self.text_encodings[idx]

    def __len__(self):
        return len(self.labels)
