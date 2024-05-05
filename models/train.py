import data.data_representation as Loader
from bert_model import BERTClassifier
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(epoch_index, tb_writer, train_loader, optimizer, model, loss_fn, device):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def per_epoch_activity(epochs, model, train_loader, val_loader, loss_fn, optimizer, device):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/workspaces/Hate-Speech-Detection/runs/bertclassifier_{}'.format(timestamp))
    epoch_number = 0
    EPOCHS = epochs
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, train_loader, optimizer, model, loss_fn, device)
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def train_model():
    model = BERTClassifier(use_lstm=True)
    train_loader, val_loader = Loader.load()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCEWithLogitsLoss()
    epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_epoch_activity(epochs, model, train_loader, val_loader, loss, optimizer, device)

if __name__ =="__main__":
    train_model()