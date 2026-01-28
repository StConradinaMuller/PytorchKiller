#!/usr/bin/env bash
# Example run script (make executable: chmod +x scripts/run.sh)
python src/train.py --dataset mnist --model cnn --epochs 5 --batch-size 128 --lr 1e-3
# CIFAR example:
# python src/train.py --dataset cifar10 --model cnn --epochs 10 --batch-size 256 --lr 1e-3
