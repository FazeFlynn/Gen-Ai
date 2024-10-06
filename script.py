# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from transformers import AdamW, get_linear_schedule_with_warmup  # For optimization
from nltk.tokenize import word_tokenize  # Or use SpaCy for tokenization
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')

# Define Hyperparameters
MAX_SEQ_LEN = 512
VOCAB_SIZE = 30000  # This depends on the tokenizer
EMBED_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
HIDDEN_DIM = 1024
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000

# Data Preprocessing Class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_SEQ_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text, max_length=self.max_len, padding='max_length', truncation=True)
        return torch.tensor(encoding)

# Define the Transformer-based Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, hidden_dim, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self.get_positional_encoding(max_seq_len, embed_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return self.softmax(x)
    
    def get_positional_encoding(self, max_seq_len, embed_dim):
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

# Load and Tokenize Data
texts = ["your dataset goes here"]  # You can load large datasets like The Pile
tokenizer = word_tokenize  # For simplicity. Consider using BPE or a better tokenizer.
train_texts, val_texts = train_test_split(texts, test_size=0.2)

train_dataset = TextDataset(train_texts, tokenizer)
val_dataset = TextDataset(val_texts, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Optimizer, and Scheduler Setup
model = TransformerModel(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
                         num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, max_seq_len=MAX_SEQ_LEN)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=EPOCHS*len(train_loader))

# Loss Function (Cross-Entropy)
criterion = nn.CrossEntropyLoss()

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch.long().to('cuda')
            outputs = model(input_ids)
            
            # Reshape and calculate loss
            outputs = outputs.view(-1, VOCAB_SIZE)
            input_ids = input_ids.view(-1)
            loss = criterion(outputs, input_ids)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

# Evaluate Model
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.long().to('cuda')
            outputs = model(input_ids)
            
            outputs = outputs.view(-1, VOCAB_SIZE)
            input_ids = input_ids.view(-1)
            loss = criterion(outputs, input_ids)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Train the Model
model.to('cuda')
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)
val_loss = evaluate_model(model, val_loader, criterion)
print(f'Validation Loss: {val_loss}')
