import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):

    def __init__(self, freeze_bert=True, num_classes=2, dropout_prob=0.1, use_lstm=False, lstm_hidden_size=128):
        super(BERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.freeze_bert = freeze_bert  
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)

        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True)
        else:
            self.lstm = None

        self.linear = nn.Linear(self.bert.config.hidden_size if not self.use_lstm else lstm_hidden_size, num_classes)  # Adjust output size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, text, masks):
        output_bert = self.bert(text, attention_mask=masks).last_hidden_state
        pooled_output = self.pool(output_bert.transpose(1, 2)).squeeze(-1)

        if self.use_lstm:
            lstm_output, _ = self.lstm(self.dropout(pooled_output.unsqueeze(0))) 
            output = lstm_output.squeeze(0) 
        else:
            output = self.dropout(pooled_output)

        return self.linear(output)
