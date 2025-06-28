from methods.BaseMethod import BaseMethod
import typing
import os
import pickle

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import networkx as nx

class CausalDataset(Dataset):
    """
    A PyTorch Dataset for causal discovery data.
    """

    def __init__(
        self,
        X: typing.List[pd.DataFrame],
        y: typing.List[pd.DataFrame]
    ) -> None:
        self.X = np.zeros([len(X), 1000, 10], dtype=np.float32)
        self.y = np.zeros([len(X), 10, 10], dtype=np.float32)
        self.target_mask = np.zeros([len(X), 10, 10], dtype=bool)

        for i in range(len(X)):
            self.X[i, : X[i].shape[0], : X[i].shape[1]] = X[i].values
            self.y[i, : y[i].shape[0], : y[i].shape[1]] = y[i].values
            self.target_mask[i, : y[i].shape[0], : y[i].shape[1]] = True

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "X": self.X[idx],
            "y": self.y[idx],
            "target_mask": self.target_mask[idx],
        }

def preprocessing(X: pd.DataFrame) -> torch.Tensor:
    x = torch.Tensor(X.values).unsqueeze(0)
    return x

class CausalModel(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.final = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k = self.input_layer(x.unsqueeze(-1)).chunk(2, dim=-1)
        x = torch.einsum("b s i d, b s j d -> b i j d", q, k) * (x.shape[1] ** -0.5)
        y = self.final(x).squeeze(-1)
        return y

class ModelWrapper(pl.LightningModule):
    def __init__(self, d_model=64):
        super().__init__()
        self.model = CausalModel(d_model)
        self.train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx: int):
        x = batch["X"]
        y = batch["y"]
        mask = batch["target_mask"]
        preds = self(x)
        loss = self.train_criterion(preds[mask], y[mask])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

def transform_proba_to_DAG(
    nodes: typing.List[str],
    pred: np.ndarray
) -> np.ndarray:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edge("X", "Y")

    x_idx, y_idx = np.unravel_index(np.argsort(pred.ravel())[::-1], pred.shape)
    for i, j in zip(x_idx, y_idx):
        n1, n2 = nodes[i], nodes[j]
        if i == j or {n1, n2} == {"X", "Y"}:
            continue
        if pred[i, j] > 0.5:
            G.add_edge(n1, n2)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(n1, n2)

    return nx.to_numpy_array(G)

class MyMethod(BaseMethod):
    """
    Baseline method that trains on X_train/y_train when phase=='dev'
    and infers DAGs on X_test.
    """

    def __init__(self, name: str, d_model: int = 64, batch_size: int = 64, max_epochs: int = 10):
        super().__init__(name)
        self.d_model = d_model
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def run(
        self,
        phase: str,
        X_train: typing.Dict[str, pd.DataFrame] = None,
        y_train: typing.Dict[str, pd.DataFrame] = None,
        X_test: typing.Dict[str, pd.DataFrame] = None,
        model_directory_path: str = "model_dir_NN",
        id_column_name: str = "id",
        prediction_column_name: str = "pred"
    ) -> pd.DataFrame:
        # Train on dev if needed
        if phase == "dev" and X_train is not None and y_train is not None:
            self._train(X_train, y_train, model_directory_path)

        # Always infer on test/dev set
        df_pred = self._infer(
            X_test,
            model_directory_path,
            id_column_name,
            prediction_column_name
        )
        return df_pred

    def _train(
        self,
        X_train: typing.Dict[str, pd.DataFrame],
        y_train: typing.Dict[str, pd.DataFrame],
        model_directory_path: str
    ) -> None:
        X_list = list(X_train.values())
        y_list = list(y_train.values())
        dataset = CausalDataset(X_list, y_list)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        model = ModelWrapper(d_model=self.d_model)
        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=self.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        trainer.fit(model, loader)

        os.makedirs(model_directory_path, exist_ok=True)
        torch.save(model.model.state_dict(), os.path.join(model_directory_path, "model.pt"))

    def _infer(
        self,
        X_test: typing.Dict[str, pd.DataFrame],
        model_directory_path: str,
        id_column_name: str,
        prediction_column_name: str
    ) -> pd.DataFrame:
        model = CausalModel(d_model=self.d_model)
        state = torch.load(os.path.join(model_directory_path, "model.pt"), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        results = {}
        for name, df in X_test.items():
            x = preprocessing(df)
            with torch.no_grad():
                pred = model(x)[0]
                pred = torch.sigmoid(pred).cpu().numpy()

            nodes = list(df.columns)
            dag = transform_proba_to_DAG(nodes, pred).astype(int)
            G = pd.DataFrame(dag, columns=nodes, index=nodes)

            for i in nodes:
                for j in nodes:
                    results[f"{name}_{i}_{j}"] = int(G.loc[i, j])

        series = pd.Series(results)
        df_out = series.reset_index()
        df_out.columns = [id_column_name, prediction_column_name]
        return df_out
