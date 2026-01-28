#!/usr/bin/env python3
"""
Main training script for the PyTorch demo project.

Usage example:
python src/train.py --dataset mnist --model cnn --epochs 5 --batch-size 128
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloaders
from models.simple_cnn import SimpleCNN
from models.simple_mlp import SimpleMLP
from utils import train_one_epoch, evaluate, save_checkpoint, set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--save-dir", default="checkpoints")
    return p.parse_args()

def build_model(model_name, dataset):
    if model_name == "mlp":
        # input size depends on dataset image size/channels
        if dataset == "mnist":
            input_shape = (1, 28, 28)
        else:
            input_shape = (3, 32, 32)
        return SimpleMLP(input_shape, num_classes=10)
    else:
        # cnn
        in_channels = 1 if dataset == "mnist" else 3
        return SimpleCNN(in_channels=in_channels, num_classes=10)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    train_loader, val_loader = get_dataloaders(args.dataset, args.batch_size)

    model = build_model(args.model, args.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}  Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}  "
              f"Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f"best_{args.dataset}_{args.model}.pth")
            save_checkpoint(model, optimizer, epoch, save_path)

    writer.close()

if __name__ == "__main__":
    main()
