import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        for n, p in self.bert.named_parameters():
            p.requires_grad = False
            
    def forward(self, text, masks):
        output_bert = self.bert(text, attention_mask=masks).last_hidden_state
        output_bert = self.avg_pool(output_bert.transpose(1, 2)).squeeze(-1)
        
        return self.linear(self.dropout(output_bert))