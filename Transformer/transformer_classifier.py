import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head self attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                 dim_feedforward, num_classes, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = [TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                         for _ in range(num_encoder_layers)]
        self.transformer_encoder = nn.Sequential(*encoder_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Optional mask for attention
            src_key_padding_mask: Optional mask for padding
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder layers
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence length
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.classifier(x)
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            
            # Get probabilities for positive class
            probs = torch.softmax(output, dim=1)[:, 1]
            pred = output.argmax(dim=1, keepdim=True)
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Store labels and probabilities for AUC and AP calculation
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate AUC and Average Precision
    auc_score = roc_auc_score(all_labels, all_probs)
    ap_score = average_precision_score(all_labels, all_probs)
    
    print(f'\nEvaluation Metrics:')
    print(f'Loss: {total_loss / len(test_loader):.4f}')
    print(f'Accuracy: {100. * correct / total:.2f}%')
    print(f'AUC Score: {auc_score:.4f}')
    print(f'Average Precision Score: {ap_score:.4f}')
    
    return total_loss / len(test_loader), 100. * correct / total, auc_score, ap_score

class CreditCardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv("../../Data/creditcard.csv")
    
    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = CreditCardDataset(X_train, y_train)
    test_dataset = CreditCardDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X.shape[1]  # Number of features
    num_classes = 2  # Binary classification (fraud or not)
    
    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=128,  # Reduced model size for this task
        nhead=8,
        num_encoder_layers=4,  # Fewer layers for faster training
        dim_feedforward=512,
        num_classes=num_classes,
        dropout=0.1
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print("Starting training...")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, auc_score, ap_score = evaluate(model, test_loader, criterion, device)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Test AUC: {auc_score:.4f}, Test AP: {ap_score:.4f}')
        print('-' * 50)
    
    # Save the model
    torch.save(model.state_dict(), 'transformer_fraud_detector.pth')
    print("Model saved as 'transformer_fraud_detector.pth'")

if __name__ == "__main__":
    main()