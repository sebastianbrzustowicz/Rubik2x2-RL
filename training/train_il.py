import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from envs.rubik2x2_env import Rubik2x2Env

# ===== Dataset =====
class ILDataset(Dataset):
    def __init__(self, algo_file):
        with open(algo_file, "r") as f:
            self.algorithms = json.load(f)
        self.env = Rubik2x2Env()
        self.data = []
        self.labels = []
        self._generate_samples()

    def _generate_samples(self):
        MOVE_MAP = {
            "U": (0, 0), "U'": (0, 1), "U2": (0, 2),
            "D": (1, 0), "D'": (1, 1), "D2": (1, 2),
            "F": (2, 0), "F'": (2, 1), "F2": (2, 2),
            "B": (3, 0), "B'": (3, 1), "B2": (3, 2),
            "L": (4, 0), "L'": (4, 1), "L2": (4, 2),
            "R": (5, 0), "R'": (5, 1), "R2": (5, 2),
        }

        for algo_id, (name, moves) in enumerate(self.algorithms.items()):
            self.env.cube.reset()
            # Apply reversed algorithm to get unique pattern
            for move in reversed(moves):
                if move.endswith("2"):
                    inv = move
                elif move.endswith("'"):
                    inv = move[:-1]
                else:
                    inv = move + "'"
                face, direction = MOVE_MAP[inv]
                if direction == 0:
                    self.env.cube.rotate_cw(face)
                elif direction == 1:
                    self.env.cube.rotate_ccw(face)
                else:
                    self.env.cube.rotate_180(face)

            obs = np.array(self.env.cube.state).flatten()
            self.data.append(obs)
            self.labels.append(algo_id)

        self.data = np.array(self.data, dtype=np.int64)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.eye(6, dtype=np.float32)[self.data[idx]].flatten()
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)


# ===== Model =====
class ILClassifier(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=256, num_classes=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===== Training =====
def train(model, dataloader, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for x, y in dataloader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
        acc = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.3f} | Acc: {acc:.3f}")


if __name__ == "__main__":
    DATASET_PATH = "datasets/upper_layer_algorithms.json"
    dataset = ILDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ILClassifier(input_dim=144, num_classes=len(dataset.algorithms))
    train(model, loader, epochs=50, lr=1e-3)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/il_classifier.pth")
    print("Model saved to models/il_classifier.pth")
