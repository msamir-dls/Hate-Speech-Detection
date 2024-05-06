import sys
sys.path.append("/workspaces/Hate-Speech-Detection/data")
sys.path.append("/workspaces/Hate-Speech-Detection/models")
import data_representation as Loader
from advancedbert import BERTClassifier
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def align_output(outputs):
    res = []
    for output in outputs:
        if output[0] > output[1]: res.append(0)
        else: res.append(1) 
    return torch.tensor(res, dtype=torch.long)

def train_one_epoch(epoch_index, tb_writer, train_loader, optimizer, model, loss_fn, device, batch_size=32):
    running_loss = 0.
    last_loss = 0.

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        label = batch['labels'].to(device).long()
        outputs = model(input_ids, mask)
        res = align_output(outputs)
        loss = loss_fn(res, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % batch_size == 0:
            last_loss = running_loss / batch_size # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('/workspaces/Hate-Speech-Detection/Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def per_epoch_activity(epochs, model, train_loader, val_loader, loss_fn, optimizer, device):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/workspaces/Hate-Speech-Detection/runs/bertclassifier_{}'.format(timestamp))
    epoch_number = 0
    EPOCHS = epochs
    best_vloss = 1_000_000.
    model.to(device)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, train_loader, optimizer, model, loss_fn, device)
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vbatch in enumerate(val_loader):
                vinputs = vbatch['input_ids'].to(device)
                vmask = vbatch['attention_mask'].to(device)
                vlabel = vbatch['labels'].to(device)
                voutputs = model(vinputs, vmask)
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