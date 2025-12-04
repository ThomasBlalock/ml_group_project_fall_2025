#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os

# --- Configuration ---
DATA_FILE = "model_artifacts/data_with_clusters.csv"
ARTIFACTS_DIR = "model_artifacts"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
TEST_SIZE = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create artifacts directory
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 1. Data Loading & Preprocessing ---

def load_and_prep_data(filepath):
    print("\n--- Loading and Preprocessing Data ---")
    df = pd.read_csv(filepath)
    
    numeric_cols = [
        'MEDIAN_HOUSEHOLD_INCOME', 'POVERTY_RATE', 'UNEMPLOYMENT_RATE', 
        'SNAP_RECEIPT_RATE', 'POP_16_PLUS', 'HOUSEHOLDS_TOTAL'
    ]

    categorical_cols = [
        'State', 'YEAR', 'Low_Threshold_Type', 'Cluster'
    ]
    
    target_col = 'Food_Insecurity_Rate'
    df_clean = df.dropna(subset=numeric_cols + categorical_cols + [target_col]).copy()
    df_clean = df_clean[numeric_cols + categorical_cols + [target_col]].copy()
    unique_values = {col: sorted(list(df_clean[col].unique().astype(str))) for col in categorical_cols}
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str)
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    feature_columns = [c for c in df_encoded.columns if c != target_col]
    encoded_cat_cols = [c for c in feature_columns if c not in numeric_cols]
    
    print(f"Total Features after Encoding: {len(feature_columns)}")

    # Split Data
    X_numeric = df_encoded[numeric_cols].values
    X_categorical = df_encoded[encoded_cat_cols].values.astype(float)
    y = df_encoded[target_col].values

    # Stratification
    stratify_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_numeric, X_categorical, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=stratify_bins
    )

    # Scaling
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train = np.hstack([X_num_train_scaled, X_cat_train])
    X_test = np.hstack([X_num_test_scaled, X_cat_test])
    
    metadata = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cat_cols': encoded_cat_cols,
        'feature_columns': feature_columns, 
        'unique_values': unique_values 
    }

    return X_train, X_test, y_train, y_test, scaler, metadata

# --- 2. Dataset & Model ---

class FoodSecurityDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).view(-1, 1)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

class FoodSecurityFFNN(nn.Module):
    def __init__(self, input_dim):
        super(FoodSecurityFFNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()
        
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.act3(self.bn3(self.layer3(x)))
        return self.layer4(x)

# --- 3. Training & Saving ---

def train_and_save():
    X_train, X_test, y_train, y_test, scaler, metadata = load_and_prep_data(DATA_FILE)
    
    train_loader = DataLoader(FoodSecurityDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FoodSecurityDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[1]
    model = FoodSecurityFFNN(input_dim).to(DEVICE)

    # --- Generate Architecture Diagram ---
    try:
        print("\n--- Generating Network Architecture Diagram ---")
        dummy_input = torch.randn(1, input_dim).to(DEVICE)
        y_hat = model(dummy_input)
        dot = make_dot(y_hat, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        dot.format = 'png'
        dot.render(os.path.join(ARTIFACTS_DIR, "network_architecture"))
        print(f"Diagram saved to {ARTIFACTS_DIR}/network_architecture.png")
    except Exception as e:
        print(f"Could not generate diagram (ensure torchviz and graphviz are installed): {e}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train Loop
    history = {'train_loss': [], 'val_loss': []}
    print("\n--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                val_loss += criterion(model(inputs), targets).item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(test_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten()
    r2 = r2_score(y_test, preds)
    print(f"\nFinal R^2 Score on Test Set: {r2:.4f}")

    # Save
    print(f"\n--- Saving Artifacts to '{ARTIFACTS_DIR}' ---")
    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "food_security_model.pth"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.save"))
    joblib.dump(metadata, os.path.join(ARTIFACTS_DIR, "model_metadata.save"))
    with open(os.path.join(ARTIFACTS_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f)
    print("Save complete.")

if __name__ == "__main__":
    train_and_save()