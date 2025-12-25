import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the processed data
with open('data/processed.pkl', 'rb') as f:
    X, y = pickle.load(f)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of sequences: {len(X)}")
print(f"Sequence length: {X[0].shape[0] if len(X) > 0 else 0}")
print(f"Feature dimension per timestep: {X[0].shape[1] if len(X) > 0 else 0}")

# Label distribution
label_counts = Counter(y)
print(f"Label distribution: {label_counts}")

# Plot label distribution
plt.figure(figsize=(6, 4))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('Label (0: Normal, 1: Insider)')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.xticks([0, 1])
plt.show()

# Visualize a few sequences
num_samples = min(5, len(X))
fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
for i in range(num_samples):
    seq = X[i]
    label = y[i]
    # Activity types (argmax of first 6)
    activities = np.argmax(seq[:, :6], axis=1)
    # PC normalized
    pcs = seq[:, 6]
    axes[i, 0].plot(activities, marker='o')
    axes[i, 0].set_title(f'Sample {i+1} - Activities - Label: {label}')
    axes[i, 0].set_xlabel('Time Step')
    axes[i, 0].set_ylabel('Activity Type')
    axes[i, 0].set_yticks(range(6))
    axes[i, 0].set_yticklabels(['Connect', 'Disconnect', 'HTTP', 'Logon', 'Logoff', 'Pad'])
    
    axes[i, 1].plot(pcs, marker='o', color='orange')
    axes[i, 1].set_title(f'Sample {i+1} - PC Usage - Label: {label}')
    axes[i, 1].set_xlabel('Time Step')
    axes[i, 1].set_ylabel('PC ID (normalized)')
plt.tight_layout()
plt.show()

# Sequence lengths (before padding)
seq_lengths = [np.sum(seq[:, 5] == 0) for seq in X]  # Assuming pad has activity 5 (one-hot at index 5=1, but wait)
# Better: find first pad
seq_lengths = []
for seq in X:
    activities = np.argmax(seq[:, :6], axis=1)
    pad_start = np.where(activities == 5)[0]
    if len(pad_start) > 0:
        length = pad_start[0]
    else:
        length = len(seq)
    seq_lengths.append(length)
plt.figure(figsize=(8, 6))
plt.hist(seq_lengths, bins=50, alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Lengths')
plt.show()

print(f"Average sequence length: {np.mean(seq_lengths):.2f}")
print(f"Max sequence length: {np.max(seq_lengths)}")
print(f"Min sequence length: {np.min(seq_lengths)}")