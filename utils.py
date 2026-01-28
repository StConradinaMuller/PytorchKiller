import torch
from tqdm import tqdm
import os

def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(out, y) * bs
        n += bs
    return running_loss / n, running_acc / n

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy_from_logits(out, y) * bs
            n += bs
    return running_loss / n, running_acc / n

def save_checkpoint(model, optimizer, epoch, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(model, optimizer, path, device="cpu"):
    import torch
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optim_state" in state:
        optimizer.load_state_dict(state["optim_state"])
    return state.get("epoch", 0)

def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
