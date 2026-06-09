import os
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class DiseaseRiskNet(nn.Module):

    def __init__(self, input_size):
        super(DiseaseRiskNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


def train_model(df):

    FEATURE_COLS = ["age", "bmi", "glucose", "blood_pressure",
                    "insulin", "hba1c", "family_history",
                    "physical_activity", "smoking", "cholesterol"]

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["disease"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.from_numpy(X_train)
    X_test_t  = torch.from_numpy(X_test)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    y_test_t  = torch.from_numpy(y_test).unsqueeze(1)

    dataset    = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model     = DiseaseRiskNet(input_size=len(FEATURE_COLS))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS       = 30
    train_losses = []
    test_losses  = []

    for epoch in range(EPOCHS):

        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train_t), y_train_t).item()
            test_loss  = criterion(model(X_test_t),  y_test_t).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Train: {train_loss:.4f} | "
                  f"Test: {test_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy().flatten()

    test_preds = (test_probs > 0.5).astype(int)
    accuracy   = accuracy_score(y_test, test_preds)
    print(f"\n  Final Accuracy: {accuracy:.4f}")

    gc.collect()

    return model, scaler, X_test, y_test, test_probs, train_losses, test_losses


if __name__ == "__main__":
    from data import generate_patient_data
    df = generate_patient_data()
    train_model(df)