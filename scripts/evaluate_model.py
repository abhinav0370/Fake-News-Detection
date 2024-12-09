import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
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

def evaluate_model(test_texts, test_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    test_dataset = FakeNewsDataset(test_encodings, test_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    trainer = Trainer(model=model)

    results = trainer.evaluate(test_dataset)
    print(results)

if __name__ == "__main__":
    test_df = load_data('data/test_preprocessed.csv')
    test_texts, test_labels = test_df['statement'].tolist(), test_df['label'].tolist()

    evaluate_model(test_texts, test_labels)