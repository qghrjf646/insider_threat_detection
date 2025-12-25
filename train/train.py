import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch

from lib.reverse_auto_diff import Tensor, LSTM, Linear, sigmoid_d, log_d, sum_d
from lib.optim import SGD, Adam, RMSProp, Adagrad, AdamW, Momentum, LRSchedulerOnPlateau

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set device for lib
import lib.reverse_auto_diff
lib.reverse_auto_diff.device = device

def load_data():
    with open('../data/processed.pkl', 'rb') as f:
        X, y = pickle.load(f)
    return X, y

def bce_loss(pred, target):
    # pred is sigmoid output, target is 0 or 1
    loss = - (target * log_d(pred) + (1 - target) * log_d(1 - pred))
    return loss

class LSTMClassifier:
    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTM(input_size, hidden_size)
        self.linear = Linear(hidden_size, output_size)
    
    def parameters(self):
        return self.lstm.parameters() + self.linear.parameters()
    
    def forward(self, x_seq):
        h = self.lstm.forward([Tensor(x) for x in x_seq])
        out = self.linear.forward(h)
        return sigmoid_d(out)  # for binary classification
    
    def forward_batch(self, x_batch):
        # x_batch: numpy array or torch.Tensor shape (batch_size, seq_len, input_size)
        if isinstance(x_batch, np.ndarray):
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=device)
        if isinstance(x_batch, Tensor):
            x_batch = x_batch.data
        # x_batch is torch.Tensor (B, T, F)
        h = self.lstm.forward_batch(x_batch)
        out = self.linear.forward_batch(h)
        return sigmoid_d(out)  # Tensor (B, 1)

def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=10, scheduler=None, batch_size=32):
    # Batched training
    train_losses = []
    val_losses = []

    X_train_t = torch.tensor(X_train, device=device, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, device=device, dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.tensor(X_val, device=device, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, device=device, dtype=torch.float32).reshape(-1, 1)

    n_train = len(X_train)
    n_val = len(X_val)

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        epoch_loss_sum = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            pred = model.forward_batch(xb)               # lib.Tensor (B,1)
            loss_batch = bce_loss(pred, Tensor(yb))     # Tensor (B,1)
            loss = loss_batch.mean()                    # scalar Tensor

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_sum += float(loss.data.cpu().item()) * xb.shape[0]

        train_loss = epoch_loss_sum / n_train
        train_losses.append(train_loss)

        # Validation (batched)
        val_loss_sum = 0.0
        for start in range(0, n_val, batch_size):
            xb = X_val_t[start:start+batch_size]
            yb = y_val_t[start:start+batch_size]
            pred = model.forward_batch(xb)
            loss_batch = bce_loss(pred, Tensor(yb))
            val_loss_sum += float(loss_batch.mean().data.cpu().item()) * xb.shape[0]
        val_loss = val_loss_sum / n_val
        val_losses.append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        lr = optimizer.learning_rate
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {lr:.6f}")

    return train_losses, val_losses

def evaluate_model(model, X, y):
    # Batched evaluation
    X_t = torch.tensor(X, device=device, dtype=torch.float32)
    probs = []
    for i in range(0, len(X), 256):
        xb = X_t[i:i+256]
        pred = model.forward_batch(xb)             # lib.Tensor (B,1)
        probs.append(pred.data.cpu().numpy().ravel())
    probs = np.concatenate(probs)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0
    auprc = average_precision_score(y, probs) if len(np.unique(y)) > 1 else 0
    return acc, prec, rec, f1, auc, auprc

if __name__ == '__main__':
    X, y = load_data()
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Model hyperparameters
    input_size = 7
    hidden_size = 64
    output_size = 1
    
    model = LSTMClassifier(input_size, hidden_size, output_size)
    
    # Benchmark different optimizers
    optimizers = {
        'SGD': SGD(model.parameters(), learning_rate=0.01),
        'Adam': Adam(model.parameters(), learning_rate=0.001),
        'RMSProp': RMSProp(model.parameters(), learning_rate=0.001),
        'AdamW': AdamW(model.parameters(), learning_rate=0.001)
    }
    
    results = {}
    
    for name, opt in optimizers.items():
        print(f"\nTraining with {name}")
        # Reset model parameters for fair comparison
        for param in model.parameters():
            param.data = torch.randn(*param.data.shape, device=device) * 0.01
        
        train_losses, val_losses = train_model(model, opt, X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
        results[name] = (train_losses, val_losses)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    for name, (train_l, val_l) in results.items():
        plt.plot(train_l, label=f'{name} Train')
        plt.plot(val_l, label=f'{name} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Optimizer Convergence Comparison')
    plt.savefig('convergence.png')
    plt.show()
    
    # Select best optimizer (lowest final val loss)
    best_opt_name = min(results, key=lambda x: results[x][1][-1])
    print(f"Best optimizer: {best_opt_name}")
    
    # Retrain with best optimizer on full train + val, then test
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Reset model
    for param in model.parameters():
        param.data = torch.randn(*param.data.shape, device=device) * 0.01
    
    best_opt = optimizers[best_opt_name]
    scheduler = LRSchedulerOnPlateau(best_opt, initial_lr=best_opt.learning_rate, patience=5, factor=0.5, min_lr=1e-6)
    train_losses, _ = train_model(model, best_opt, X_train_full, y_train_full, X_test, y_test, epochs=50, scheduler=scheduler, batch_size=32)
    
    # Evaluate on test
    acc, prec, rec, f1, auc, auprc = evaluate_model(model, X_test, y_test)
    print(f"Test Metrics - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")
    
    # Save model
    model_dict = {
        'lstm_weights': [p.data.cpu().numpy() for p in model.lstm.parameters()],
        'linear_weights': [p.data.cpu().numpy() for p in model.linear.parameters()]
    }
    with open('lstm_model.pkl', 'wb') as f:
        pickle.dump(model_dict, f)
    
    print("Model saved as lstm_model.pkl")