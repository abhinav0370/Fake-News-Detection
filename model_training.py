import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

def load_data(file_path):
    return pd.read_csv(file_path)

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(train_texts, train_labels, val_texts, val_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = FakeNewsDataset(train_encodings, train_labels)
    val_dataset = FakeNewsDataset(val_encodings, val_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    train_df = load_data('data/train_preprocessed.csv')
    val_df = load_data('data/valid_preprocessed.csv')

    train_texts, val_texts = train_df['statement'].tolist(), val_df['statement'].tolist()
    train_labels, val_labels = train_df['label'].tolist(), val_df['label'].tolist()

    train_model(train_texts, train_labels, val_texts, val_labels)