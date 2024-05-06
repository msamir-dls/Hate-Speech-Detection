from data_preperation import HateSpeechDS
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch

class HateSpeechData(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :]

        text = data["tweet"]
        label = data["class"]
        text_encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        return {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"], ##.unsqueeze(0).int(),
            "label": label,
        }

def dataloader(dataset, batch_size = 32):
    return DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            #converting numpy arrays to torch arrays
            collate_fn=lambda x: {
                "input_ids": torch.stack([item["input_ids"] for item in x]),
                "attention_mask": torch.stack([item["attention_mask"] for item in x]),
                "labels": torch.tensor([item["label"] for item in x])
            },
            pin_memory=True,
)


def load(path = "/workspaces/Hate-Speech-Detection/data/labeled_data.csv"):
    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    dataset = HateSpeechDS(path)
    dataset.transform_df()
    data = dataset.df
    train, validation = train_test_split(data, test_size=0.3, stratify=data['class'], random_state=42)
    dataset_train = HateSpeechData(train)
    dataset_val = HateSpeechData(validation)
    train_dataloader = dataloader(dataset_train)
    val_dataloader = dataloader(dataset_val)
    return train_dataloader, val_dataloader


if __name__ == "__nain__":
    load()

