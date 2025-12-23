import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=10, scheduler=None):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        total_loss = 0
        for x_seq, y in zip(X_train, y_train):
            pred = model.forward(x_seq)
            loss = bce_loss(pred, Tensor(y))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.data
        
        train_loss = total_loss / len(X_train)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = 0
        for x_seq, y in zip(X_val, y_val):
            pred = model.forward(x_seq)
            loss = bce_loss(pred, Tensor(y))
            val_loss += loss.data
        val_loss /= len(X_val)
        val_losses.append(val_loss)
        
        if scheduler:
            scheduler.step(val_loss)
        
        lr = optimizer.learning_rate
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, X, y):
    preds = []
    for x_seq in X:
        pred = model.forward(x_seq)
        preds.append((pred.data > 0.5).item())
    preds = np.array(preds)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, preds) if len(np.unique(y)) > 1 else 0
    return acc, prec, rec, f1, auc

if __name__ == '__main__':
    X, y = load_data()
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Model hyperparameters
    input_size = 6
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
        
        train_losses, val_losses = train_model(model, opt, X_train, y_train, X_val, y_val, epochs=20)
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
    train_losses, _ = train_model(model, best_opt, X_train_full, y_train_full, X_test, y_test, epochs=50, scheduler=scheduler)
    
    # Evaluate on test
    acc, prec, rec, f1, auc = evaluate_model(model, X_test, y_test)
    print(f"Test Metrics - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Save model
    model_dict = {
        'lstm_weights': [p.data.cpu().numpy() for p in model.lstm.parameters()],
        'linear_weights': [p.data.cpu().numpy() for p in model.linear.parameters()]
    }
    with open('lstm_model.pkl', 'wb') as f:
        pickle.dump(model_dict, f)
    
    print("Model saved as lstm_model.pkl")