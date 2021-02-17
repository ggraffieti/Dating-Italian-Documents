from transformers import CamembertTokenizer
from data.letter_dataset import LetterDataset
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import  train_test_split


def _get_dataloader(data_frame, tokenizer, max_seq_lenght, drop_last=True):
    text = data_frame["text"].tolist()
    labels = data_frame["class"].tolist()

    text_enc = tokenizer(text, truncation=True, max_length=max_seq_lenght, padding='max_length', return_tensors="pt")
    article_dataset = LetterDataset(text_enc.input_ids, labels)

    return DataLoader(article_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)


def get_train_valid_test_coarse(bert_model, max_seq_lenght):
    tokenizer = CamembertTokenizer.from_pretrained(bert_model)
    csv_train = pd.read_csv("../data/coarse_train.csv")
    train, valid = train_test_split(csv_train, test_size=0.2)
    test = pd.read_csv("../data/coarse_test.csv")

    train_dl = _get_dataloader(train, tokenizer, max_seq_lenght)
    valid_dl = _get_dataloader(valid, tokenizer, max_seq_lenght, drop_last=False)
    test_dl = _get_dataloader(test, tokenizer, max_seq_lenght, drop_last=False)

    return train_dl, valid_dl, test_dl


def get_train_valid_test_fine(bert_model, max_seq_lenght):
    tokenizer = CamembertTokenizer.from_pretrained(bert_model)
    csv_train = pd.read_csv("../data/fine_train.csv")
    train, valid = train_test_split(csv_train, test_size=0.2)
    test = pd.read_csv("../data/fine_test.csv")

    train_dl = _get_dataloader(train, tokenizer, max_seq_lenght)
    valid_dl = _get_dataloader(valid, tokenizer, max_seq_lenght, drop_last=False)
    test_dl = _get_dataloader(test, tokenizer, max_seq_lenght, drop_last=False)

    return train_dl, valid_dl, test_dl


if __name__ == "__main__":
    _, _, dl = get_train_valid_test_fine("Musixmatch/umberto-commoncrawl-cased-v1", 512)
    for lab, t in dl:
        print(lab)
        print(t)
        print(t.shape)
        break
