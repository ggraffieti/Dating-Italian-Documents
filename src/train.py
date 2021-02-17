import torch
import torch.optim as optim
from models.umberto_custom import UmbertoCustom
from data.loaders import get_train_valid_test_coarse, get_train_valid_test_fine
from test import evaluate
from utils.training_utils import save_checkpoint, load_checkpoint
import time


def train(model, optimizer, train_loader, valid_loader, num_epochs, eval_every,
          file_path, best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    print("init train")
    for epoch in range(num_epochs):
        for labels, text in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for val_labels, val_text in valid_loader:
                        val_labels = val_labels.type(torch.LongTensor)
                        val_labels = val_labels.to(device)
                        val_text = val_text.type(torch.LongTensor)
                        val_text = val_text.to(device)
                        output = model(val_text, val_labels)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model2.pt', model, best_valid_loss)

    print('Finished Training!')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    bert_model = "Musixmatch/umberto-commoncrawl-cased-v1"
    # bert_model = "idb-ita/gilberto-uncased-from-camembert"
    num_classes = 11

    bert = UmbertoCustom(bert_model=bert_model, num_classes=num_classes).to(device)

    train_iter, valid_iter, test_iter = get_train_valid_test_fine(bert_model=bert_model, max_seq_lenght=512)

    opt = optim.Adam(bert.parameters(), lr=2e-5)
    init_time = time.time()

    train(model=bert, optimizer=opt, train_loader=train_iter, valid_loader=valid_iter, num_epochs=5,
          eval_every=len(train_iter) // 2, file_path="../data/models/")

    tot_time = time.time() - init_time
    print("time taken:", int(tot_time // 60), "minutes", int(tot_time % 60), "seconds")

    best_model = UmbertoCustom(bert_model=bert_model, num_classes=num_classes).to(device)

    load_checkpoint("../data/models" + '/model2.pt', best_model, device)

    evaluate(best_model, test_iter, num_classes, device)
