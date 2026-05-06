from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "PyTorch ist für LSTM/CNN erforderlich. Bitte `pip install -r requirements.txt` ausführen."
        ) from exc
    return torch, nn


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def build_model(model_type: str, n_features: int, hidden_size: int = 32):
    torch, nn = _torch()
    model_type = model_type.upper()

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=1,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, max(8, hidden_size // 2)),
                nn.ReLU(),
                nn.Linear(max(8, hidden_size // 2), 1),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    class CNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            channels = max(16, hidden_size)
            self.net = nn.Sequential(
                nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(channels, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.net(x).squeeze(-1)
            return self.head(x).squeeze(-1)

    if model_type == "LSTM":
        return LSTMRegressor()
    if model_type == "CNN":
        return CNNRegressor()
    raise ValueError(f"Unsupported ML model type: {model_type}")


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    epochs: int = 60,
    learning_rate: float = 0.001,
    hidden_size: int = 32,
    seed: int = 42,
):
    torch, nn = _torch()
    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_type, n_features=x_train.shape[-1], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    losses = []
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = loss_fn(prediction, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return model, {"loss": losses, "device": str(device)}


def predict_model(model, x_values: np.ndarray) -> np.ndarray:
    torch, _ = _torch()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_values, dtype=torch.float32, device=device)
        prediction = model(x_tensor).detach().cpu().numpy()
    return prediction.astype(float)
