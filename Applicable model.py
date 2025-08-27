import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import random
from typing import List, Tuple, Dict
import warnings
import os
import json
from datetime import datetime
import math
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class EnzymeDataGenerator:
    """Generate synthetic enzyme variant data for training with controlled mutation patterns"""
    
    def __init__(self, wt_sequence: str, substrate_sequence: str):
        self.wt_sequence = wt_sequence
        self.substrate_sequence = substrate_sequence
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
    def generate_mutations_by_sites(self, n_mutations: int, n_variants: int) -> List[str]:
        """Generate variants with exactly n_mutations point mutations"""
        if n_mutations == 0:
            return [self.wt_sequence] * n_variants
            
        variants = []
        max_attempts = n_variants * 10  # Prevent infinite loops
        attempts = 0
        
        while len(variants) < n_variants and attempts < max_attempts:
            attempts += 1
            
            # Select random positions for mutation
            positions = random.sample(range(len(self.wt_sequence)), n_mutations)
            variant = list(self.wt_sequence)
            
            # Mutate selected positions
            valid_variant = True
            for pos in positions:
                new_aa = random.choice(self.amino_acids)
                # Ensure mutation is different from wild type
                while new_aa == self.wt_sequence[pos]:
                    new_aa = random.choice(self.amino_acids)
                variant[pos] = new_aa
            
            variant_str = ''.join(variant)
            if variant_str not in variants:  # Avoid duplicates
                variants.append(variant_str)
        
        # Fill remaining slots if needed
        while len(variants) < n_variants:
            variants.append(variants[0] if variants else self.wt_sequence)
        
        return variants[:n_variants]
    
    def generate_comprehensive_library(self, variants_per_mutation_level: List[int]) -> Tuple[List[str], List[int]]:
        """Generate comprehensive mutation library with different mutation levels
        
        Args:
            variants_per_mutation_level: List where index i represents number of variants 
                                       with i mutations (0-site, 1-site, ..., 6-site)
        """
        all_variants = []
        mutation_levels = []
        
        print("Generating comprehensive mutation library:")
        for n_mutations, n_variants in enumerate(variants_per_mutation_level):
            if n_variants > 0:
                variants = self.generate_mutations_by_sites(n_mutations, n_variants)
                all_variants.extend(variants)
                mutation_levels.extend([n_mutations] * len(variants))
                print(f"  {n_mutations}-site mutations: {len(variants)} variants")
        
        return all_variants, mutation_levels
    
    def simulate_activity(self, sequence: str, mutation_level: int = None) -> float:
        """Enhanced activity simulation with mutation-level awareness"""
        # Calculate amino acid composition features
        hydrophobic_aas = set('AILMFPWV')
        charged_aas = set('DEKR')
        polar_aas = set('STNQCY')
        aromatic_aas = set('FWY')
        
        hydrophobicity = sum(1 for aa in sequence if aa in hydrophobic_aas) / len(sequence)
        charge_ratio = sum(1 for aa in sequence if aa in charged_aas) / len(sequence)
        polarity = sum(1 for aa in sequence if aa in polar_aas) / len(sequence)
        aromaticity = sum(1 for aa in sequence if aa in aromatic_aas) / len(sequence)
        
        # Calculate actual mutations from wild type
        actual_mutations = sum(1 for i, aa in enumerate(sequence) if aa != self.wt_sequence[i])
        
        # More sophisticated activity model
        base_activity = 1.0
        
        # Mutation penalty - non-linear relationship
        if actual_mutations == 0:
            mut_penalty = 1.0
        else:
            mut_penalty = np.exp(-actual_mutations * 0.15) * (1 + 0.1 * np.random.normal())
        
        # Amino acid composition effects
        composition_effect = (
            1.0 + 0.3 * hydrophobicity - 0.2 * charge_ratio + 
            0.15 * polarity + 0.25 * aromaticity
        )
        
        # Position-specific effects (some positions more critical)
        critical_positions = list(range(50, 100))  # Example critical region
        critical_mutations = sum(1 for i in range(len(sequence)) 
                               if i in critical_positions and sequence[i] != self.wt_sequence[i])
        critical_penalty = np.exp(-critical_mutations * 0.3)
        
        # Calculate final activity
        activity = base_activity * composition_effect * mut_penalty * critical_penalty
        
        # Add experimental noise (increases with mutation level)
        noise_level = 0.05 + 0.02 * actual_mutations
        activity += np.random.normal(0, noise_level)
        
        # Ensure non-negative activity
        activity = max(0, activity)
        
        return activity

class EnzymeDataset(Dataset):
    """Enhanced PyTorch dataset for enzyme sequences and activities"""
    
    def __init__(self, sequences: List[str], activities: List[float], substrate_seq: str, 
                 mutation_levels: List[int] = None):
        self.sequences = sequences
        self.activities = activities
        self.substrate_seq = substrate_seq
        self.mutation_levels = mutation_levels if mutation_levels else [0] * len(sequences)
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # One-hot encode enzyme sequence
        enzyme_encoded = self.encode_sequence(self.sequences[idx])
        
        # One-hot encode substrate sequence
        substrate_encoded = self.encode_sequence(self.substrate_seq)
        
        # Add mutation level as additional feature
        mutation_level = torch.tensor([self.mutation_levels[idx]], dtype=torch.float32)
        
        # Combine all features
        combined_features = torch.cat([enzyme_encoded, substrate_encoded, mutation_level], dim=0)
        
        activity = torch.tensor(self.activities[idx], dtype=torch.float32)
        
        return combined_features, activity
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """One-hot encode amino acid sequence"""
        encoded = torch.zeros(len(sequence), len(self.amino_acids))
        for i, aa in enumerate(sequence):
            if aa in self.aa_to_idx:
                encoded[i, self.aa_to_idx[aa]] = 1
        return encoded.flatten()

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EnzymeTransformer(nn.Module):
    """Transformer-based model for enzyme activity prediction"""
    
    def __init__(self, enzyme_len: int, substrate_len: int, vocab_size: int = 20, 
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(EnzymeTransformer, self).__init__()
        
        self.d_model = d_model
        self.enzyme_len = enzyme_len
        self.substrate_len = substrate_len
        
        # Embedding layers
        self.enzyme_embedding = nn.Linear(vocab_size, d_model)
        self.substrate_embedding = nn.Linear(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.ReLU()  # Ensure non-negative output
        )
    
    def forward(self, x):
        # Split combined input back into components
        enzyme_flat = x[:, :self.enzyme_len * 20]
        substrate_flat = x[:, self.enzyme_len * 20:self.enzyme_len * 20 + self.substrate_len * 20]
        mutation_level = x[:, -1:]
        
        # Reshape to sequence format
        enzyme_seq = enzyme_flat.view(-1, self.enzyme_len, 20)
        substrate_seq = substrate_flat.view(-1, self.substrate_len, 20)
        
        # Embed sequences
        enzyme_embedded = self.enzyme_embedding(enzyme_seq)
        substrate_embedded = self.substrate_embedding(substrate_seq)
        
        # Concatenate enzyme and substrate sequences
        combined_seq = torch.cat([enzyme_embedded, substrate_embedded], dim=1)
        
        # Add positional encoding
        combined_seq = self.pos_encoding(combined_seq.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(combined_seq)
        
        # Global pooling
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        
        # Add mutation level information
        pooled = torch.cat([pooled, mutation_level], dim=1)
        
        # Final prediction
        output = self.classifier(pooled).squeeze(-1)
        
        return output

class EnzymeActivityPredictor(nn.Module):
    """Enhanced deep learning model for predicting enzyme activity"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [512, 256, 128, 64]):
        super(EnzymeActivityPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3 if i < len(hidden_sizes) - 1 else 0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.ReLU()  # Ensure non-negative output
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

class ModelManager:
    """Handle model saving and loading"""
    
    @staticmethod
    def save_model(model, model_type: str, metrics: dict, save_dir: str = "saved_models"):
        """Save model with metadata"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_enzyme_model_{timestamp}"
        
        # Save model state
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            "model_type": model_type,
            "timestamp": timestamp,
            "metrics": metrics,
            "model_path": model_path
        }
        
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return model_path, metadata_path
    
    @staticmethod
    def load_model(model_path: str, model_class, *args, **kwargs):
        """Load saved model"""
        model = model_class(*args, **kwargs)
        model.load_state_dict(torch.load(model_path))
        return model

class EnzymeEvolutionPipeline:
    """Enhanced main pipeline for enzyme evolution using deep learning"""
    
    def __init__(self, wt_sequence: str, substrate_sequence: str):
        self.wt_sequence = wt_sequence
        self.substrate_sequence = substrate_sequence
        self.data_generator = EnzymeDataGenerator(wt_sequence, substrate_sequence)
        self.model = None
        self.transformer_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = ModelManager()
        print(f"Using device: {self.device}")
    
    def generate_training_data(self, variants_per_level: List[int] = None) -> Tuple[List[str], List[float], List[int]]:
        """Generate comprehensive training dataset with controlled mutation levels"""
        
        if variants_per_level is None:
            # Default: 0-site to 6-site mutations
            variants_per_level = [50, 300, 400, 350, 250, 150, 100]  # Total: 1600 variants
        
        print("Generating comprehensive training data...")
        
        # Generate variants with different mutation levels
        all_variants, mutation_levels = self.data_generator.generate_comprehensive_library(variants_per_level)
        
        # Simulate activities
        activities = []
        for i, seq in enumerate(all_variants):
            activity = self.data_generator.simulate_activity(seq, mutation_levels[i])
            activities.append(activity)
        
        print(f"Generated {len(all_variants)} variants total")
        print(f"Activity range: {min(activities):.3f} - {max(activities):.3f}")
        print(f"Mutation levels: {min(mutation_levels)} - {max(mutation_levels)}")
        
        return all_variants, activities, mutation_levels
    
    def prepare_datasets(self, sequences: List[str], activities: List[float], 
                        mutation_levels: List[int], test_size: float = 0.2, val_size: float = 0.1):
        """Prepare train/validation/test datasets"""
        
        # Split data
        X_temp, X_test, y_temp, y_test, m_temp, m_test = train_test_split(
            sequences, activities, mutation_levels, test_size=test_size, random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
            X_temp, y_temp, m_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # Create datasets
        train_dataset = EnzymeDataset(X_train, y_train, self.substrate_sequence, m_train)
        val_dataset = EnzymeDataset(X_val, y_val, self.substrate_sequence, m_val)
        test_dataset = EnzymeDataset(X_test, y_test, self.substrate_sequence, m_test)
        
        return train_dataset, val_dataset, test_dataset
    
    def train_mlp_model(self, train_dataset, val_dataset, batch_size: int = 32, 
                       epochs: int = 100, learning_rate: float = 0.001):
        """Train the MLP deep learning model"""
        
        print("Training MLP model...")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = len(train_dataset[0][0])
        self.model = EnzymeActivityPredictor(input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_mlp_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 20:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= 20:
                print("Early stopping triggered")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_mlp_model.pth'))
        
        return train_losses, val_losses
    
    def train_transformer_model(self, train_dataset, val_dataset, batch_size: int = 16, 
                               epochs: int = 100, learning_rate: float = 0.0005):
        """Train the transformer model"""
        
        print("Training Transformer model...")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize transformer model
        enzyme_len = len(self.wt_sequence)
        substrate_len = len(self.substrate_sequence)
        
        self.transformer_model = EnzymeTransformer(
            enzyme_len=enzyme_len,
            substrate_len=substrate_len,
            vocab_size=20,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.transformer_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        
        # Training loop
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.transformer_model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.transformer_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.transformer_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.transformer_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.transformer_model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 15:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= 15:
                print("Early stopping triggered")
                break
        
        # Load best model
        self.transformer_model.load_state_dict(torch.load('best_transformer_model.pth'))
        
        return train_losses, val_losses
    
    def evaluate_model(self, model, test_dataset, model_name: str):
        """Evaluate model performance"""
        
        print(f"Evaluating {model_name} model...")
        
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        mse = mean_squared_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        pearson_r, pearson_p = pearsonr(true_values, predictions)
        spearman_r, spearman_p = spearmanr(true_values, predictions)
        
        metrics = {
            'MSE': mse,
            'R²': r2,
            'Pearson r': pearson_r,
            'Pearson p-value': pearson_p,
            'Spearman r': spearman_r,
            'Spearman p-value': spearman_p
        }
        
        print(f"\n{model_name} Model Performance:")
        for metric, value in metrics.items():
            if 'p-value' in metric:
                print(f"{metric}: {value:.2e}")
            else:
                print(f"{metric}: {value:.4f}")
        
        # Save model with metadata
        self.model_manager.save_model(model, model_name.lower(), metrics)
        
        return predictions, true_values, metrics
    
    def predict_activity(self, sequences: List[str], model_type: str = 'mlp') -> np.ndarray:
        """Predict activity for new sequences"""
        
        model = self.model if model_type == 'mlp' else self.transformer_model
        
        if model is None:
            raise ValueError(f"{model_type} model not trained yet")
        
        # Create dummy mutation levels (will be calculated automatically)
        mutation_levels = [sum(1 for i, aa in enumerate(seq) if aa != self.wt_sequence[i]) 
                          for seq in sequences]
        
        dataset = EnzymeDataset(sequences, [0] * len(sequences), self.substrate_sequence, mutation_levels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def design_variants(self, n_variants: int = 100, max_mutations: int = 6, 
                       strategy: str = 'random') -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
        """Design new enzyme variants using both models"""
        
        print(f"Designing {n_variants} new variants with up to {max_mutations} mutations...")
        
        if strategy == 'random':
            # Generate variants with different mutation levels
            variants = []
            for n_muts in range(1, max_mutations + 1):
                n_vars_this_level = n_variants // max_mutations
                vars_this_level = self.data_generator.generate_mutations_by_sites(n_muts, n_vars_this_level)
                variants.extend(vars_this_level)
            
            # Fill to exact number
            while len(variants) < n_variants:
                extra_vars = self.data_generator.generate_mutations_by_sites(
                    random.randint(1, max_mutations), 1)
                variants.extend(extra_vars)
            variants = variants[:n_variants]
        
        # Predict activities with both models
        mlp_predictions = self.predict_activity(variants, 'mlp')
        transformer_predictions = self.predict_activity(variants, 'transformer')
        
        # Create variant-activity tuples
        mlp_results = list(zip(variants, mlp_predictions))
        transformer_results = list(zip(variants, transformer_predictions))
        
        # Sort by predicted activity
        mlp_results.sort(key=lambda x: x[1], reverse=True)
        transformer_results.sort(key=lambda x: x[1], reverse=True)
        
        # Add mutation counts
        mlp_results_with_muts = [(var, act, sum(1 for i, aa in enumerate(var) if aa != self.wt_sequence[i])) 
                                for var, act in mlp_results]
        transformer_results_with_muts = [(var, act, sum(1 for i, aa in enumerate(var) if aa != self.wt_sequence[i])) 
                                        for var, act in transformer_results]
        
        wt_mlp_activity = self.predict_activity([self.wt_sequence], 'mlp')[0]
        wt_transformer_activity = self.predict_activity([self.wt_sequence], 'transformer')[0]
        
        print(f"Wild type predicted activity - MLP: {wt_mlp_activity:.4f}, Transformer: {wt_transformer_activity:.4f}")
        print(f"Top MLP prediction: {mlp_results_with_muts[0][1]:.4f}")
        print(f"Top Transformer prediction: {transformer_results_with_muts[0][1]:.4f}")
        
        return mlp_results_with_muts, transformer_results_with_muts
    
    def plot_results(self, mlp_losses, transformer_losses, mlp_predictions, mlp_true, 
                    transformer_predictions, transformer_true):
        """Plot comprehensive results comparison"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # Training curves comparison
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(mlp_losses[0], label='MLP Train', alpha=0.8)
        plt.plot(mlp_losses[1], label='MLP Val', alpha=0.8)
        plt.plot(transformer_losses[0], label='Transformer Train', alpha=0.8)
        plt.plot(transformer_losses[1], label='Transformer Val', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        # MLP Prediction vs True
        ax2 = plt.subplot(3, 3, 2)
        plt.scatter(mlp_true, mlp_predictions, alpha=0.6, color='blue')
        min_val, max_val = min(mlp_true), max(mlp_true)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('True Activity')
        plt.ylabel('Predicted Activity')
        plt.title('MLP: Predicted vs True')
        plt.grid(True)
        
        # Transformer Prediction vs True
        ax3 = plt.subplot(3, 3, 3)
        plt.scatter(transformer_true, transformer_predictions, alpha=0.6, color='green')
        min_val, max_val = min(transformer_true), max(transformer_true)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('True Activity')
        plt.ylabel('Predicted Activity')
        plt.title('Transformer: Predicted vs True')
        plt.grid(True)
        
        # MLP Residuals
        ax4 = plt.subplot(3, 3, 4)
        mlp_residuals = np.array(mlp_predictions) - np.array(mlp_true)
        plt.scatter(mlp_true, mlp_residuals, alpha=0.6, color='blue')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Activity')
        plt.ylabel('Residuals')
        plt.title('MLP Residuals')
        plt.grid(True)
        
        # Transformer Residuals
        ax5 = plt.subplot(3, 3, 5)
        transformer_residuals = np.array(transformer_predictions) - np.array(transformer_true)
        plt.scatter(transformer_true, transformer_residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Activity')
        plt.ylabel('Residuals')
        plt.title('Transformer Residuals')
        plt.grid(True)
        
        # Activity distributions comparison
        ax6 = plt.subplot(3, 3, 6)
        plt.hist(mlp_true, bins=30, alpha=0.5, label='True', density=True, color='gray')
        plt.hist(mlp_predictions, bins=30, alpha=0.7, label='MLP', density=True, color='blue')
        plt.hist(transformer_predictions, bins=30, alpha=0.7, label='Transformer', density=True, color='green')
        plt.xlabel('Activity')
        plt.ylabel('Density')
        plt.title('Activity Distribution Comparison')
        plt.legend()
        plt.grid(True)
        
        # Model comparison scatter
        ax7 = plt.subplot(3, 3, 7)
        plt.scatter(mlp_predictions, transformer_predictions, alpha=0.6)
        min_pred = min(min(mlp_predictions), min(transformer_predictions))
        max_pred = max(max(mlp_predictions), max(transformer_predictions))
        plt.plot([min_pred, max_pred], [min_pred, max_pred], 'r--', lw=2)
        plt.xlabel('MLP Predictions')
        plt.ylabel('Transformer Predictions')
        plt.title('Model Predictions Comparison')
        plt.grid(True)
        
        # Error comparison
        ax8 = plt.subplot(3, 3, 8)
        mlp_errors = np.abs(mlp_residuals)
        transformer_errors = np.abs(transformer_residuals)
        plt.boxplot([mlp_errors, transformer_errors], labels=['MLP', 'Transformer'])
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution Comparison')
        plt.grid(True)
        
        # Performance metrics comparison
        ax9 = plt.subplot(3, 3, 9)
        mlp_r2 = r2_score(mlp_true, mlp_predictions)
        transformer_r2 = r2_score(transformer_true, transformer_predictions)
        mlp_mse = mean_squared_error(mlp_true, mlp_predictions)
        transformer_mse = mean_squared_error(transformer_true, transformer_predictions)
        
        metrics = ['R²', 'MSE']
        mlp_scores = [mlp_r2, mlp_mse]
        transformer_scores = [transformer_r2, transformer_mse]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, mlp_scores, width, label='MLP', color='blue', alpha=0.7)
        plt.bar(x + width/2, transformer_scores, width, label='Transformer', color='green', alpha=0.7)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_mutation_patterns(self, sequences: List[str], activities: List[float], 
                                mutation_levels: List[int]):
        """Analyze activity patterns by mutation level"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Activity by mutation level
        mutation_activities = {}
        for i, level in enumerate(mutation_levels):
            if level not in mutation_activities:
                mutation_activities[level] = []
            mutation_activities[level].append(activities[i])
        
        levels = sorted(mutation_activities.keys())
        mean_activities = [np.mean(mutation_activities[level]) for level in levels]
        std_activities = [np.std(mutation_activities[level]) for level in levels]
        
        axes[0, 0].errorbar(levels, mean_activities, yerr=std_activities, 
                           marker='o', capsize=5, capthick=2)
        axes[0, 0].set_xlabel('Number of Mutations')
        axes[0, 0].set_ylabel('Mean Activity')
        axes[0, 0].set_title('Activity vs Mutation Level')
        axes[0, 0].grid(True)
        
        # Activity distribution by mutation level
        activity_data = [mutation_activities[level] for level in levels]
        axes[0, 1].boxplot(activity_data, labels=levels)
        axes[0, 1].set_xlabel('Number of Mutations')
        axes[0, 1].set_ylabel('Activity')
        axes[0, 1].set_title('Activity Distribution by Mutation Level')
        axes[0, 1].grid(True)
        
        # Best activity at each mutation level
        best_activities = [max(mutation_activities[level]) for level in levels]
        axes[1, 0].bar(levels, best_activities, alpha=0.7)
        axes[1, 0].set_xlabel('Number of Mutations')
        axes[1, 0].set_ylabel('Best Activity')
        axes[1, 0].set_title('Best Activity by Mutation Level')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample count by mutation level
        sample_counts = [len(mutation_activities[level]) for level in levels]
        axes[1, 1].bar(levels, sample_counts, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Number of Mutations')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Dataset Composition')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return mutation_activities

def main():
    """Enhanced main function with comprehensive mutation analysis"""
    
    # Define wild type enzyme and substrate sequences
    wt_enzyme = "MTKIIEAQPGTWLDWQTRTGVRLNVGKDYEVRMTFKAGLPGDLIGGQKYDGKQATQVFLFSTANVDNKPVMLSYDSFLQSEIIIYPKIPYHFQNLNPQTLSRGGAHFKNSASYLSDTHKPLASQFEVLQVYSAISDAKGFSYTAYNEKIDNRLQFNVVMQGLILGNGGVFNKWGGLTLYQIVVQGYNFTDVIAMAYDLSDMTEDIEAAKAKRDGFVFIDAYGGHTNELGQYAAFKKRTREWLKGQLLSYFRAAFGTPIYTGTYQLVADKNNDHVWKWKMTRFGKFNGQNTVADTYQMYLVVPPRNWFVMDSFYVAQHSRKFGQLTKNMAEGFVQSVFNSASVEKMRELDIDLGMIGKTAAGKYEVDEYKGYVLVHEERRLLYLDDNGTLDPPSEQAGAVLSDVFWEEAETDLPIDMGYTFQVVGKTAIVPTTFPDATWFFLKYETRKAKLYLYILGVPKKDGTMAYYWGNLNVDIDFNPLLYIVDDKGGTILGKGGQALLLSEGQGDVVVAMYDGVVMNWRPFNPKEILSRFAAFLLPYKGEELQFEVREQTFHIAYELVTEDRDQSQAAFKSDYKGIDLVHKWIRPLQAQTLTDMLLQYISYVLQHDKDRDMGIPQSLRTAWMSATLKVVNALSTQSVEAQIYGMEYAAYDGEFYRRAGSSVTDDLLKLGYILPEAFNALGEKDLQHFPIGIWQLARDLNKMADPMQMQLNVDFPGDNNHLMALKIQQGLKEQQCWIMHKLDTTTRYSEADIYAEQGTIRVNRDHDFKPIFRFFETMCASWASYSRKYLSSLMTHPLVVQHMPMFFGVLAPIQLQTFNLLKLDTMDNRYLNVMEKVAANPNALVAQFGQRVNLLTTAELGFGLVQKFASAQKLGDSATVNNLTKYIHMDQVVSMGRVDDKVDQMRKAYTQFFQHGGGLQPDLNKIPVFIEPGGDEKYRHVILETDLGEPIVGSPFWATQWHMLDVLSLRDAAGFMFVKKAEKQFASLNTLLVDGLNKGLYQGFVNRYNAFEDGYLLQPGQWFDGLVLEALAFRQVQNLVEVPSHDLYMSSLYLDYLRQVNIWTGDRSANMGGSKKDLYYLTEEYMSWLDIAFAKFPWSLNKQNIMDQWHLQVSDHLSPTLHTLGFTFMWGDGYNAMDLNTPQDKAALDKIKWVKGDLRQYAHEYGLNHLEAAFQNVNSQVQAVVEHAAAQNDKGDSDRYKSYGVGAEGLTVGTLLGTGLLVDPHDMFTKFVTGDPATYEKDVNGLGCGVKQAQASKLPSHRVLPQYFLLLAKVQLSEIVKSQPDKAFGASSDTILAQYRLQVDDGGAKLIESVYQNNHSGYNLLKQHWGPPLHGDQYQVMYVPFATLFPAAELLQDVEPVGRGGLAGRYLFMTTNYNYQTQAAKGQYHWYLHQHPMDGQASMGFQEIFSGDKKMAKGVSVEQFGQLNGEQLAWRLHKYYGDAIIEDASRTLAYAEVAARALRHSQVLGDFNAEGFRGVSNKPYQYFDLQAGFAADWRIQNVAVQLDGRFAGALVSPLDPQFMGGVLQSRGGAAGHHAFMGLHLDIDKDVDMAFVFTVLRVRVRQGDISQYSQIGMQVVNLRQQSQEIMTLFAKQAFKMAKLNPSLAEKLLHGEPAAYKDQHAGFVFQFIHDPGAQMQYGAKVATLEKELLAYEPEDVQMMTGQEDLGKELVHQLQKPFKMAGLQKMVVQLLMKQEEFKQFHDMFNSLELYYGAAKQLLPHGLLNRDIKGQKDYQLYKTDFATYYGLGKYQMLEAAQASYLYNMVLFGLQRQLRPFPPNVQAYGVYVYSKPLGGQGWIQSGQALHTQVKQALTANVTYLMALKIAAYYMRLVDWVYTVYSGGDSGMMGPQDKLLMYYKSMQAIHLVDYAVSYQINTVLFYLQRDYAGANDALLAGTLSEYLLYQDGFTRMPEYQEIKLHQYFFQFLQGEKFAADKKGVVDMASVDCVQRLLGDFHVFQKGNLPNQQDLVGLLWGQHIFATLIQIVDGLGLGQFGDPRSQQRVQLLQYAQGQRVFHTLYGGAYQAQYGSYLQGDLSIDSHPYAYDDKYQQLLKYIQSYDPRPVAAWYIGDLNLTYLQGSGGGFGHQREWYLNQQQTHLYLAEQAATGLQTLAYDGNATYFFADGMGQIAADRFSLHEPLFAIQYYQQLVNLPLPQYQGMLLGHDLLYDNSQVQAAFGYFSVQSQGMFGRFMQGHGLSIYQQNKKQLNVLGQGYDQAEHQLLEKLHDLLDYEAFMPNWFQQPVVMPPQGQNPQAAAGWLSLLFGQVPASLPQQQQVLLRQQLVQQSLVHALATVQKQVQQHGGMPILLGGLRWGAAFQHVQQFKVQGQYHHQYQGGAGALFSGGLQDQFGDDDHDYQVGGLNQFVPFFGQATYQNSTLQGQPFQQVMGGLDNYFQRDGQMQAIAKLGDMAFYQNFPHFAKGLDYAAQGGHQNFMVGTLNQVAQIFAQFYTLQQVQGFKQRQKLPGQFGDQQVLGVLGTYQMAHGIPQGFMPMAQFHQVDTVQMNGHFFQQQPLTVAIFGQGQMLNQQNAGYHQAMQFHLYEGSQLQGFMQIQNQVSLPFGFAMQYIQNQVFHQAFFMQFYQGALFQFPLQALYALFQNQLHQYLWQGQSLQFVQGQMYIAQDYQGQLQFGDNQLLMAFGSQYQQVYHFQGQFQAQQFAMGQSYDYQEYQNFAFQGPYQHFLLSQQFPGQAQLQFQFGQNQFAAGQMLQGGLQFQGQYQGQNAGQFQGQSYQNQQQVQFQGGQYQGFQFQGGQPQQVYAFGQQHYIQQQGQMGQAAYQFVQGQHYQAQQVQFQGGQYQGFQFQGGQPQQVYAFGQQHYIQQQGQMGQAAYQFVQGQHYQAQQVQFQGGQYQGFQFQGGQPQQA"
    substrate_seq = "MGSSHHHHHHSSGLVPRGSHMGTGSFLVRESETNPAVTTAR"
    
    print("Enhanced Enzyme Evolution Deep Learning Pipeline")
    print("=" * 60)
    print("Features:")
    print("- Controlled mutation levels (0-6 site mutations)")
    print("- MLP and Transformer model comparison")
    print("- Comprehensive model saving and evaluation")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnzymeEvolutionPipeline(wt_enzyme, substrate_seq)
    
    # Generate comprehensive training data
    variants_per_level = [100, 300, 400, 350, 250, 200, 150]  # 0-6 site mutations
    sequences, activities, mutation_levels = pipeline.generate_training_data(variants_per_level)
    
    # Analyze mutation patterns
    print("\nAnalyzing mutation patterns...")
    mutation_activities = pipeline.analyze_mutation_patterns(sequences, activities, mutation_levels)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = pipeline.prepare_datasets(
        sequences, activities, mutation_levels)
    
    print(f"\nDataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Train MLP model
    print("\n" + "="*40)
    print("TRAINING MLP MODEL")
    print("="*40)
    mlp_train_losses, mlp_val_losses = pipeline.train_mlp_model(
        train_dataset, val_dataset, batch_size=32, epochs=80, learning_rate=0.001
    )
    
    # Train Transformer model
    print("\n" + "="*40)
    print("TRAINING TRANSFORMER MODEL")
    print("="*40)
    transformer_train_losses, transformer_val_losses = pipeline.train_transformer_model(
        train_dataset, val_dataset, batch_size=16, epochs=60, learning_rate=0.0005
    )
    
    # Evaluate both models
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    
    mlp_predictions, mlp_true_values, mlp_metrics = pipeline.evaluate_model(
        pipeline.model, test_dataset, "MLP")
    
    transformer_predictions, transformer_true_values, transformer_metrics = pipeline.evaluate_model(
        pipeline.transformer_model, test_dataset, "Transformer")
    
    # Design new variants using both models
    print("\n" + "="*40)
    print("VARIANT DESIGN")
    print("="*40)
    
    mlp_designed, transformer_designed = pipeline.design_variants(
        n_variants=50, max_mutations=6, strategy='random')
    
    print("\nTop 5 MLP-Designed Variants:")
    for i, (variant, activity, n_muts) in enumerate(mlp_designed[:5]):
        print(f"{i+1}. Activity: {activity:.4f}, Mutations: {n_muts}")
    
    print("\nTop 5 Transformer-Designed Variants:")
    for i, (variant, activity, n_muts) in enumerate(transformer_designed[:5]):
        print(f"{i+1}. Activity: {activity:.4f}, Mutations: {n_muts}")
    
    # Plot comprehensive comparison
    print("\n" + "="*40)
    print("GENERATING PLOTS")
    print("="*40)
    
    pipeline.plot_results(
        (mlp_train_losses, mlp_val_losses),
        (transformer_train_losses, transformer_val_losses),
        mlp_predictions, mlp_true_values,
        transformer_predictions, transformer_true_values
    )
    
    # Model comparison summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<15} {'MLP':<12} {'Transformer':<12} {'Better':<10}")
    print("-" * 50)
    print(f"{'R²':<15} {mlp_metrics['R²']:<12.4f} {transformer_metrics['R²']:<12.4f} {'Transformer' if transformer_metrics['R²'] > mlp_metrics['R²'] else 'MLP':<10}")
    print(f"{'MSE':<15} {mlp_metrics['MSE']:<12.4f} {transformer_metrics['MSE']:<12.4f} {'Transformer' if transformer_metrics['MSE'] < mlp_metrics['MSE'] else 'MLP':<10}")
    print(f"{'Pearson r':<15} {mlp_metrics['Pearson r']:<12.4f} {transformer_metrics['Pearson r']:<12.4f} {'Transformer' if transformer_metrics['Pearson r'] > mlp_metrics['Pearson r'] else 'MLP':<10}")
    print(f"{'Spearman r':<15} {mlp_metrics['Spearman r']:<12.4f} {transformer_metrics['Spearman r']:<12.4f} {'Transformer' if transformer_metrics['Spearman r'] > mlp_metrics['Spearman r'] else 'MLP':<10}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Saved models:")
    print("- MLP model: best_mlp_model.pth")
    print("- Transformer model: best_transformer_model.pth")
    print("- Model metadata and performance metrics in saved_models/ directory")
    print("- Both models ready for variant prediction and enzyme evolution!")

if __name__ == "__main__":
    main()
