# PyTorch Demo Project — Learn ML by Doing

This demo project is a compact, runnable PyTorch example to learn the full cycle: dataset, model, training loop, evaluation, checkpointing, and logging.

Features
- Train on MNIST or CIFAR-10 (torchvision datasets)
- Two simple models: `SimpleMLP` and `SimpleCNN`
- Training loop with GPU support, checkpointing, and TensorBoard logging
- Utilities for evaluation and reproducibility

Quick start (local)
1. Create a project directory and save the files from this repo (preserving the paths).
2. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows (PowerShell)
   pip install -r requirements.txt
   ```

3. Train on MNIST (default):
   ```bash
   python src/train.py --dataset mnist --model cnn --epochs 5 --batch-size 128
   ```

4. Use TensorBoard to inspect training:
   ```bash
   tensorboard --logdir runs
   ```

Initialize a git repository and make the first commit
```bash
git init
git add .
git commit -m "Initial commit: PyTorch demo project"
# If you want the branch named main:
git branch -M main
# To push to a new GitHub repo:
# git remote add origin git@github.com:<owner>/<repo>.git
# git push -u origin main
```

Files overview
- `src/train.py` — main training script
- `src/data.py` — dataloaders for MNIST and CIFAR-10
- `src/models/simple_mlp.py` — small fully-connected model
- `src/models/simple_cnn.py` — small CNN for image classification
- `src/utils.py` — training/evaluation helpers and checkpointing
- `requirements.txt` — Python dependencies
- `scripts/run.sh` — example run script
- `.gitignore` — ignore common files
- `LICENSE` — MIT License

Learning tips
- Start with MNIST + MLP, then try the CNN and compare results.
- Add augmentation, schedulers, or experiment with optimizers.
- Add more instrumentation (per-class metrics, confusion matrix, gradient visualizations).

If you want, I can:
- Create the GitHub repo and push the initial commit (I’ll need owner/repo name),
- Add a Jupyter notebook walkthrough that visualizes training curves and predictions.
- Add GitHub Actions CI to run linting/tests on pushes.

License
This project is provided under the MIT License (see LICENSE file).