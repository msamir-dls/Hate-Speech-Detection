import sys
sys.path.append("/workspaces/Hate-Speech-Detection/data")
sys.path.append("/workspaces/Hate-Speech-Detection/models")
import data_representation as Loader
from bert import BERTClassifier
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def train():
    model = BERTClassifier()
    train_loader, val_loader = Loader.load()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    val_losses = []
    train_losses = []

    for epoch_i in range(0, epochs):
        
        model.train()
        print(f"Start training epoch {epoch_i}...")
        total_train_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
        
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            label = batch['labels'].to(device) 

            output = model(input_ids, masks)
            loss = criterion(output.squeeze(), label.float())

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        print("Start validation...")
        y_true_bert = list()
        y_pred_bert = list()
        
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                label = batch['labels'].to(device)
                
                output = model(input_ids, masks)
                max_output = (torch.sigmoid(output).cpu().numpy().reshape(-1)>= 0.5).astype(int)
                y_true_bert.extend(label.tolist())
                y_pred_bert.extend(max_output.tolist())
                
                loss_v = criterion(output.squeeze(), label.float())
                total_eval_loss += loss.item()
        avg_val_loss = total_eval_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Metrics after Epoch {epoch_i}")     
        print(f"Accuracy : {accuracy_score(y_true_bert, y_pred_bert)}")
        print(f"Presision: {np.round(precision_score(y_true_bert, y_pred_bert),3)}")
        print(f"Recall: {np.round(recall_score(y_true_bert, y_pred_bert),3)}") 
        print(f"F1: {np.round(f1_score(y_true_bert, y_pred_bert),3)}")
        print("   ")  
    
    return model, train_losses, val_losses

def viz_losses(loss1, loss2):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(loss2,label="val")
    plt.plot(loss1,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    model, train_loss, val_loss = train()
    # Save the trained model    
    torch.save(model.state_dict(), 'trained_models/BertForSequenceClassification.pth')  
    viz_losses(train_loss, val_loss)