from methods.BaseMethod import BaseMethod
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import cv2
import numpy as np

class DepthDataset(Dataset):
    def __init__(self, csv_file, img_folder):
        self.df = pd.read_csv(csv_file)
        self.images = []
        self.intra_depths = []
        self.inter_depths = []

        for _, row in self.df.iterrows():
            img_path = os.path.join(img_folder, row["image_id"])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1)  # (C,H,W) for PyTorch
            image = torch.tensor(image, dtype=torch.float32) / 255.0

            self.images.append(image)
            self.intra_depths.append(torch.tensor(row["intra_depth"], dtype=torch.float32).unsqueeze(-1))
            self.inter_depths.append(torch.tensor(row["inter_depth"], dtype=torch.float32).unsqueeze(-1))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.intra_depths[idx], self.inter_depths[idx]
    

class DepthOrderingModel(nn.Module):
    def __init__(self, input_channels=3):
        super(DepthOrderingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (N, 64, 64, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: (N, 128, 32, 32)
        )
        
        # Calculate the flattened size: 128 channels * 32 height * 32 width
        self.flattened_size = 128 * 32 * 32 

        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU()
        )
        self.inter_depth_output = nn.Linear(256, 1)
        self.intra_depth_output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.regressor(x)
        inter_depth = self.inter_depth_output(x)
        intra_depth = self.intra_depth_output(x)
        return inter_depth, intra_depth
    

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model
        self.model = DepthOrderingModel().to(self.device)

    def train(self):

        # Load in train data
        folder = "./data/train"
        img_folder = os.path.join(folder, "images")
        csv = os.path.join(folder, "annotations.csv")

        dataset = DepthDataset(csv, img_folder)

        # Train/validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create trianing variables, loss, optimizer
        optim = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.MSELoss()
        patience = 10
        epochs_without_improvement = 0
        best_validation_loss = torch.inf
        best_model_state = None
        
        for epoch in range(50):

            self.model.train()

            for img, y_intra, y_inter in train_loader:

                # Move to device
                img = img.to(self.device)
                y_intra = y_intra.to(self.device)
                y_inter = y_inter.to(self.device)

                pred_intra, pred_inter = self.model(img)

                loss_intra = criterion(pred_intra, y_intra)
                loss_inter = criterion(pred_inter, y_inter)

                loss = loss_intra + loss_inter

                optim.zero_grad()

                loss.backward()
                optim.step()

            # Validation round

            self.model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for val_img, val_y_intra, val_y_inter in val_loader:

                    val_img = val_img.to(self.device)
                    val_y_intra = val_y_intra.to(self.device)
                    val_y_inter = val_y_inter.to(self.device)

                    pred_intra, pred_inter = self.model(val_img)

                    val_loss_intra = criterion(pred_intra, val_y_intra)
                    val_loss_inter = criterion(pred_inter, val_y_inter)

                    validation_loss += (val_loss_intra + val_loss_inter).item()

            # Early stopping logic
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = self.model.state_dict()  # save best state
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > patience:
                self.model.load_state_dict(best_model_state)  # rollback
                break

    def run(self, data_directory):
        self.model.eval()

        img_folder = os.path.join(data_directory, "images")
        csv = os.path.join(data_directory, "annotations.csv")
        output_path = os.path.join("./output", f"predictions.csv")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load dataset
        dataset = DepthDataset(csv, img_folder)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []
        with torch.no_grad():
            for img, _, _ in loader:
                img = img.to(self.device)
                pred_intra, pred_inter = self.model(img)
                # Convert predictions to list of tuples
                preds = list(zip(pred_intra.cpu().numpy().flatten(),
                                pred_inter.cpu().numpy().flatten()))
                predictions.extend(preds)

        # Save predictions with corresponding image IDs
        df = pd.read_csv(csv)
        df["pred_intra"] = [p[0] for p in predictions]
        df["pred_inter"] = [p[1] for p in predictions]
        df.to_csv(output_path, index=False)