import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load the LIAR dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 
                  'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 
                  'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    return df

# Preprocess the dataset
def preprocess_data(df):
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    # Keep only necessary columns
    df = df[['statement', 'label']]
    
    return df

if __name__ == "__main__":
    train_df = load_dataset(r'C:\fnd\data\train.tsv')
    test_df = load_dataset(r'C:\fnd\data\test.tsv')
    valid_df = load_dataset(r'C:\fnd\data\valid.tsv')

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    valid_df = preprocess_data(valid_df)
    
    # Save preprocessed data
    train_df.to_csv(r'C:\fnd\data\train_preprocessed.csv', index=False)
    test_df.to_csv(r'C:\fnd\data\test_preprocessed.csv', index=False)
    valid_df.to_csv(r'C:\fnd\data\valid_preprocessed.csv', index=False)