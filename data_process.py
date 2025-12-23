import pandas as pd
import numpy as np
from datetime import datetime
import pickle

def load_data():
    device = pd.read_csv('data/r1/device.csv')
    http = pd.read_csv('data/r1/http.csv', header=None, names=['id', 'date', 'user', 'pc', 'url'])
    logon = pd.read_csv('data/r1/logon.csv')
    
    # Add type
    device['type'] = device['activity'].map({'Connect': 0, 'Disconnect': 1})
    http['type'] = 2
    logon['type'] = logon['activity'].map({'Logon': 3, 'Logoff': 4})
    
    # Combine
    device = device[['date', 'user', 'type']]
    http = http[['date', 'user', 'type']]
    logon = logon[['date', 'user', 'type']]
    
    data = pd.concat([device, http, logon], ignore_index=True)
    
    # Parse date
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y %H:%M:%S')
    
    return data

def load_labels():
    # For r1, no insiders, so return empty set
    return set()

def create_sequences(data, max_len=100):
    users = data['user'].unique()
    sequences = {}
    labels = {}
    
    malicious_users = load_labels()
    
    for user in users:
        user_data = data[data['user'] == user].sort_values('date')
        seq = user_data['type'].tolist()
        if len(seq) > max_len:
            seq = seq[:max_len]
        elif len(seq) < max_len:
            seq += [5] * (max_len - len(seq))  # pad with 5 (unknown)
        
        sequences[user] = seq
        labels[user] = 1 if user in malicious_users else 0
    
    return sequences, labels

def one_hot_sequences(sequences, num_classes=6):
    X = []
    y = []
    for user, seq in sequences.items():
        seq_oh = np.zeros((len(seq), num_classes))
        for i, t in enumerate(seq):
            seq_oh[i, t] = 1
        X.append(seq_oh)
        y.append(labels[user])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    data = load_data()
    sequences, labels = create_sequences(data)
    
    X, y = one_hot_sequences(sequences)
    
    # Save
    with open('data/processed.pkl', 'wb') as f:
        pickle.dump((X, y), f)
    
    print(f"Data processed: {X.shape}, labels: {y.shape}")