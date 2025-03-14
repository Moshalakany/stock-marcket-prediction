import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use the output from the last time step
        return self.decoder(x[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HybridCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        super(HybridCNNLSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layers for sequential processing
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        # Reshape for Conv1d: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        
        # Reshape for LSTM: (batch_size, seq_len/2, 128)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len/2, hidden_dim)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply output layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

class ModelBuilder:
    def __init__(self, models_dir: str = "models"):
        """Initialize the model builder."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Define model architectures for different sectors
        self.sector_architectures = {
            "Technology": "transformer",  # More dynamic, benefits from transformer's attention
            "Healthcare": "hybrid",       # Complex with long-term dependencies
            "Financial Services": "lstm",  # Traditional time-series patterns
            "Consumer Cyclical": "hybrid", # Seasonal with complex patterns
            "Communication Services": "transformer", # Rapidly changing, needs attention
            "Consumer Defensive": "lstm",  # Stable, traditional patterns
            "Industrials": "hybrid",      # Mixed patterns
            "Energy": "lstm",             # Cyclical patterns
            "Utilities": "lstm",          # Stable patterns
            "Basic Materials": "hybrid",  # Cyclical with external influences
            "Real Estate": "lstm"         # Slower moving trends
        }
        
        # Default architecture for any sector not explicitly defined
        self.default_architecture = "hybrid"
    
    def create_model(self, sector: str, input_dim: int) -> nn.Module:
        """Create a model based on the sector."""
        architecture = self.sector_architectures.get(sector, self.default_architecture)
        
        if architecture == "transformer":
            return TransformerModel(input_dim=input_dim)
        elif architecture == "lstm":
            return nn.LSTM(
                input_size=input_dim,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
        else:  # hybrid or default
            return HybridCNNLSTM(input_dim=input_dim)
            
    def save_model(self, model: nn.Module, scaler: StandardScaler, symbol: str, sector: str):
        """Save the trained model and scaler."""
        # Create directory for sector if it doesn't exist
        sector_dir = os.path.join(self.models_dir, sector.replace(" ", "_"))
        os.makedirs(sector_dir, exist_ok=True)
        
        # Create directory for symbol if it doesn't exist
        symbol_dir = os.path.join(sector_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(symbol_dir, f"model_{timestamp}.pth")
        scaler_path = os.path.join(symbol_dir, f"scaler_{timestamp}.pkl")
        
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save latest model reference
        latest_path = os.path.join(symbol_dir, "latest_model.txt")
        with open(latest_path, 'w') as f:
            f.write(f"model_{timestamp}.pth\nscaler_{timestamp}.pkl")
            
        return model_path, scaler_path
        
    def load_model(self, symbol: str, sector: str, input_dim: Optional[int] = None) -> Tuple[nn.Module, StandardScaler]:
        """Load the latest trained model for a symbol."""
        sector_dir = os.path.join(self.models_dir, sector.replace(" ", "_"))
        symbol_dir = os.path.join(sector_dir, symbol)
        latest_path = os.path.join(symbol_dir, "latest_model.txt")
        
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"No trained model found for {symbol} in sector {sector}")
            
        with open(latest_path, 'r') as f:
            model_file, scaler_file = f.read().strip().split('\n')
            
        model_path = os.path.join(symbol_dir, model_file)
        scaler_path = os.path.join(symbol_dir, scaler_file)
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        # Create and load model
        architecture = self.sector_architectures.get(sector, self.default_architecture)
        if input_dim is None:
            # Try to infer input_dim from scaler
            if hasattr(scaler, 'n_features_in_'):
                input_dim = scaler.n_features_in_
            else:
                raise ValueError("input_dim must be provided when it cannot be inferred from scaler")
                
        model = self.create_model(sector, input_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model, scaler
