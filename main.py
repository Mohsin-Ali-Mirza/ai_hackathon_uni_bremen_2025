import os
import os.path
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.models import densenet121
from torcheval.metrics import BinaryAUROC
from alert import alert

# Set up CUDA benchmarks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# --- 1. Setup Functions ---

def setup_logging(root_dir: Path):
    """Configures logging to file and console."""
    log_path = root_dir / "cxr_pneu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    # Capture uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler

def setup_wandb(run_name: str, wandb_dir: Path, config: Dict[str, Any]):
    """Initializes a new Weights & Biases run."""
    wandb_dir.mkdir(exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.abspath(wandb_dir)
    wandb.init(
        project="cxr_small_data_pneu",
        name=run_name,
        dir=wandb_dir,
        config=config
    )

# --- 2. Model & Data Classes ---

class CXP_Model(nn.Module):
    """DenseNet-121 model for CheXpert classification."""
    def __init__(self):
        super().__init__()
        self.encoder = densenet121(weights='IMAGENET1K_V1')
        self.clf = nn.Linear(1000, 1) # Output logits

    def forward(self, x):
        z = self.encode(x)
        return self.clf(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def predict_proba(self, x):
        return torch.sigmoid(self(x))

def get_transforms(augment: bool = False) -> transforms.Compose:
    """Returns the image transformation pipeline."""
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    base_transforms = [
        transforms.Resize(
            (224, 224), 
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
        normalize_transform
    ]
    
    if augment:
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3))
        ]
        return transforms.Compose(base_transforms + augmentation_transforms)
    else:
        return transforms.Compose(base_transforms)

class CXP_dataset(torchvision.datasets.VisionDataset):
    """Custom CheXpert dataset."""
    def __init__(self, root_dir: str, csv_file: str, transform: transforms.Compose) -> None:
        super().__init__(root_dir, transform=transform)
        
        df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.path = df.Path.str.replace('CheXpert-v1.0/', 'CheXpert-v1.0-small/', regex=False)
        self.labels = df.Pneumothorax.astype(int)
        self.drain = df.Drain.astype(int)
        self.transform = transform

    def __getitem__(self, index: int):
        try:
            img_path = os.path.join(self.root_dir, self.path[index])
            img = torchvision.io.read_image(img_path)
            img = self.transform(img)
            return img, self.labels[index], self.drain[index]
        except RuntimeError as e:
            logging.error(f"Error loading image at index {index}: {self.path[index]}")
            logging.error(f"Error message: {e}")
            # Return the next valid image
            return self.__getitem__((index + 1) % len(self))
    
    def __len__(self) -> int:
        return len(self.path)

def get_dataloaders(data_dir: Path, csv_dir: Path, batch_size: int, num_workers: int, balance_train_set: bool) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Creates and returns all necessary dataloaders."""
    
    train_transform = get_transforms(augment=True)
    eval_transform = get_transforms(augment=False)

    train_data = CXP_dataset(data_dir, csv_dir / 'train_drain_shortcut.csv', train_transform)
    val_data = CXP_dataset(data_dir, csv_dir / 'val_drain_shortcut.csv', eval_transform)
    test_data_aligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_aligned.csv', eval_transform)
    test_data_misaligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_misaligned.csv', eval_transform)

    sampler = None
    shuffle_train = True

    if balance_train_set:
        logging.info("Balancing training set by drain status.")
        pneu_msk = train_data.labels == 1
        
        drain_counts_pneu = torch.bincount(torch.from_numpy(train_data.drain[pneu_msk].values))
        drain_weights_pneu = 1.0 / drain_counts_pneu.float()
        
        drain_counts_nopneu = torch.bincount(torch.from_numpy(train_data.drain[~pneu_msk].values))
        drain_weights_nopneu = 1.0 / drain_counts_nopneu.float()
        
        sample_weights = torch.zeros_like(torch.from_numpy(train_data.labels.values), dtype=torch.float32)
        sample_weights[pneu_msk] = drain_weights_pneu[train_data.drain[pneu_msk].values]
        sample_weights[~pneu_msk] = drain_weights_nopneu[train_data.drain[~pneu_msk].values]

        logging.info(f'Pneu weights: {drain_weights_pneu}')
        logging.info(f'No Pneu weights: {drain_weights_nopneu}')

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False # Sampler and shuffle are mutually exclusive

    common_loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': 2
    }

    train_loader = DataLoader(train_data, sampler=sampler, shuffle=shuffle_train, **common_loader_params)
    val_loader = DataLoader(val_data, shuffle=True, **common_loader_params) # Original code used shuffle=True for val
    test_loader_aligned = DataLoader(test_data_aligned, shuffle=False, **common_loader_params)
    test_loader_misaligned = DataLoader(test_data_misaligned, shuffle=False, **common_loader_params)
    
    return train_loader, val_loader, test_loader_aligned, test_loader_misaligned

# --- 3. Core Training & Evaluation Functions ---

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer, device: torch.device, 
                    train_auroc: BinaryAUROC) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    train_auroc.reset()
    
    total_loss = 0.0
    total_brier = 0.0
    num_samples = 0
    
    for inputs, labels, _ in tqdm(loader, desc="Training"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(inputs).reshape(-1)
        loss = criterion(outputs, labels.to(torch.float32))
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            total_loss += loss.item() * inputs.size(0)
            train_auroc.update(outputs, labels)
            probs = torch.sigmoid(outputs)
            brier = ((probs - labels.float()) ** 2).sum().item()
            total_brier += brier
        
        num_samples += inputs.size(0)

    avg_loss = total_loss / num_samples
    avg_brier = total_brier / num_samples
    auroc = train_auroc.compute()
    
    return avg_loss, auroc, avg_brier

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
             device: torch.device, eval_auroc: BinaryAUROC, 
             desc: str = "Evaluating") -> Tuple[float, float, float]:
    """Runs a single evaluation loop."""
    model.eval()
    eval_auroc.reset()
    
    total_loss = 0.0
    total_brier = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, labels, _ in tqdm(loader, desc=desc):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs).reshape(-1)
            loss = criterion(outputs, labels.to(torch.float32))
            
            total_loss += loss.item() * inputs.size(0)
            eval_auroc.update(outputs, labels)
            
            probs = torch.sigmoid(outputs)
            brier = ((probs - labels.float()) ** 2).sum().item()
            total_brier += brier
            
            num_samples += inputs.size(0)

    avg_loss = total_loss / num_samples
    avg_brier = total_brier / num_samples
    auroc = eval_auroc.compute()
    
    return avg_loss, auroc, avg_brier

def get_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Gets model predictions for a given dataset."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for inputs, labels, drain in tqdm(loader, desc="Getting Predictions"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs).reshape(-1)
            probs = torch.sigmoid(outputs.cpu())
            
            results.append(pd.DataFrame({
                'label': labels.cpu().numpy(),
                'y_prob': probs.numpy(),
                'drain': drain.numpy()
            }))
            
    results_df = pd.concat(results, ignore_index=True)
    results_df.label = results_df.label.astype(bool)
    results_df.y_prob = results_df.y_prob.astype(np.float64)
    return results_df

# --- 4. Main Experiment Orchestrator ---

def run_experiment(config: Dict[str, Any], data_dir: Path, csv_dir: Path, 
                   out_dir: Path, device: torch.device) -> Dict[str, float]:
    """
    Runs a single, complete experiment, including training, validation,
    and final testing.
    """
    # --- Setup ---
    run_name = config['run_name']
    checkpoint_path = out_dir / f"{run_name}.chkpt"
    
    setup_wandb(run_name, out_dir / "wandb", config)
    
    train_loader, val_loader, test_loader_aligned, test_loader_misaligned = \
        get_dataloaders(data_dir, csv_dir, config['batch_size'], 
                        config['num_workers'], config['balance_train'])

    model = CXP_Model().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.005)
    
    # Initialize metrics
    metrics = {
        'train_auroc': BinaryAUROC().to(device),
        'val_auroc': BinaryAUROC().to(device),
        'test_auroc_aligned_epoch': BinaryAUROC().to(device),
        'test_auroc_misaligned_epoch': BinaryAUROC().to(device),
        'test_auroc_aligned_final': BinaryAUROC().to(device),
        'test_auroc_misaligned_final': BinaryAUROC().to(device),
    }

    best_val_auroc = 0.0

    # --- Training Loop ---
    for epoch in range(config['epochs']):
        logging.info(f'======= EPOCH {epoch+1} / {config["epochs"]} =======')
        
        # Train
        train_loss, train_auroc, train_brier = train_one_epoch(
            model, train_loader, criterion, optimizer, device, metrics['train_auroc']
        )
        
        # Update and evaluate EMA model
        ema_model.update_parameters(model)
        
        # Validate
        val_loss, val_auroc, val_brier = evaluate(
            ema_model, val_loader, criterion, device, metrics['val_auroc'], "Validating"
        )
        
        logging.info(f"Epoch [{epoch + 1}/{config['epochs']}] Train Loss: {train_loss:.4f} AUROC: {train_auroc:.4f} Brier: {train_brier:.4f}")
        logging.info(f"                   Val   Loss: {val_loss:.4f} AUROC: {val_auroc:.4f} Brier: {val_brier:.4f}")

        # Per-epoch test evaluation
        test_loss_aligned, test_auroc_aligned, test_brier_aligned = evaluate(
            ema_model, test_loader_aligned, criterion, device, metrics['test_auroc_aligned_epoch'], "Testing Aligned (Epoch)"
        )
        test_loss_misaligned, test_auroc_misaligned, test_brier_misaligned = evaluate(
            ema_model, test_loader_misaligned, criterion, device, metrics['test_auroc_misaligned_epoch'], "Testing Misaligned (Epoch)"
        )
        
        # Log to W&B
        wandb.log({
            "Loss/train": train_loss, "auroc/train": train_auroc, "brier/train": train_brier,
            "Loss/val": val_loss, "auroc/val": val_auroc, "brier/val": val_brier,
            "Test_epoch/Loss_Aligned": test_loss_aligned, "Test_epoch/AUROC_Aligned": test_auroc_aligned, "Test_epoch/Brier_Aligned": test_brier_aligned,
            "Test_epoch/Loss_Misaligned": test_loss_misaligned, "Test_epoch/AUROC_Misaligned": test_auroc_misaligned, "Test_epoch/Brier_Misaligned": test_brier_misaligned
        })
        
        # Save best checkpoint based on validation AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            logging.info(f"Saving new best chkpt at epoch {epoch+1} (Val AUROC: {best_val_auroc:.4f}).")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_loss': val_loss,
            }, checkpoint_path)

    # --- Final Evaluation ---
    logging.info("===== STARTING FINAL EVALUATION =====")
    
    # Load best checkpoint
    best_chkpt = torch.load(checkpoint_path)
    # Re-initialize model and EMA model to load state
    model = CXP_Model().to(device) 
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    ema_model.load_state_dict(best_chkpt['ema_model_state_dict'])
    ema_model.to(device)
    
    logging.info(f"Loaded best model from epoch {best_chkpt['epoch']} with Val AUROC: {best_chkpt['val_auroc']:.4f}")
    
    # Sanity check: re-run on val set
    val_loss_final, val_auroc_final, _ = evaluate(
        ema_model, val_loader, criterion, device, metrics['val_auroc'], "Validating (Final)"
    )
    logging.info(f"Reloaded Val AUROC (sanity check): {val_auroc_final:.4f}")
    
    # Final Test: Aligned
    final_loss_aligned, final_auroc_aligned, final_brier_aligned = evaluate(
        ema_model, test_loader_aligned, criterion, device, metrics['test_auroc_aligned_final'], "Testing Aligned (Final)"
    )
    logging.info(f"Test Loss ALIGNED: {final_loss_aligned:.4f} AUROC: {final_auroc_aligned:.4f} Brier: {final_brier_aligned:.4f}")

    # Final Test: Misaligned
    final_loss_misaligned, final_auroc_misaligned, final_brier_misaligned = evaluate(
        ema_model, test_loader_misaligned, criterion, device, metrics['test_auroc_misaligned_final'], "Testing Misaligned (Final)"
    )
    logging.info(f"Test Loss MISALIGNED: {final_loss_misaligned:.4f} AUROC: {final_auroc_misaligned:.4f} Brier: {final_brier_misaligned:.4f}")

    # Log final metrics to W&B
    wandb.log({
        "Test_final/Loss_Aligned": final_loss_aligned, "Test_final/AUROC_Aligned": final_auroc_aligned, "Test_final/Brier_Aligned": final_brier_aligned,
        "Test_final/Loss_Misaligned": final_loss_misaligned, "Test_final/AUROC_Misaligned": final_auroc_misaligned, "Test_final/Brier_Misaligned": final_brier_misaligned
    })

    # Save predictions to CSV
    test_results_aligned_df = get_predictions(ema_model, test_loader_aligned, device)
    test_results_aligned_df.to_csv(out_dir / f'cxp_pneu_densenet_test_results_aligned_{run_name}.csv', index=False)
    
    test_results_misaligned_df = get_predictions(ema_model, test_loader_misaligned, device)
    test_results_misaligned_df.to_csv(out_dir / f'cxp_pneu_densenet_test_results_misaligned_{run_name}.csv', index=False)

    wandb.finish()
    
    return {
        'test_auroc_aligned': final_auroc_aligned.item(),
        'test_auroc_misaligned': final_auroc_misaligned.item(),
        'test_brier_aligned': final_brier_aligned,
        'test_brier_misaligned': final_brier_misaligned,
        'test_loss_aligned': final_loss_aligned,
        'test_loss_misaligned': final_loss_misaligned
    }

# --- 5. Argument Parser and Main Entrypoint ---

def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="CheXpert Pneumothorax Shortcut Training")
    
    # --- Paths ---
    parser.add_argument('--data_dir', type=str, default='/data',
                        help='Directory above /CheXpert-v1.0-small')
    parser.add_argument('--csv_dir', type=str, default='.',
                        help='Directory containing train/val/test CSV files')
    parser.add_argument('--out_dir', type=str, default='~/cxp_shortcut_out',
                        help='Directory for outputs (logs, checkpoints, etc.)')
    
    # --- Experiment Config ---
    parser.add_argument('--base_name', type=str, default='mohsin_run_baseline',
                        help='Base name for W&B runs and checkpoints')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of times to repeat the experiment')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs per run')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training and evaluation batch size')
    parser.add_argument('--balance_train', action='store_true',
                        help='Balance the train set by drain presence')
    parser.add_argument('--no_balance_train', dest='balance_train', action='store_false',
                        help='Do not balance the train set (default)')
    parser.set_defaults(balance_train=True) # Default set to True as in original
    
    # --- System ---
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of dataloader workers')

    return parser.parse_args()

def main():
    """Main execution function."""
    args = get_args()
    
    data_dir = Path(args.data_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir).expanduser()
    
    setup_logging(out_dir)

    if torch.cuda.is_available():
        logging.info("Using GPU")
        device = torch.device("cuda:0")
    else:
        logging.info("Using CPU")
        device = torch.device("cpu")
    
    # Create the master config dictionary
    config = {
        "learning_rate": args.lr,
        "architecture": "densenet121",
        "dataset": "CheXpert",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "balance_train": args.balance_train,
        "base_name": args.base_name
    }
    
    all_results: List[Dict[str, float]] = []
    
    for i in range(args.num_runs):
        run_num = i + 1
        logging.info(f"===== STARTING RUN {run_num} / {args.num_runs} =====")
        
        # Create a unique name for this specific run
        run_name = f"{args.base_name}_{args.epochs}ep_run_{run_num}"
        config['run_name'] = run_name
        
        try:
            run_results = run_experiment(config, data_dir, csv_dir, out_dir, device)
            all_results.append(run_results)
            logging.info(f"===== FINISHED RUN {run_num} / {args.num_runs} =====")
        except Exception as e:
            logging.critical(f"Run {run_num} failed with error: {e}", exc_info=True)
            # Ensure wandb is finished even if the run fails
            if wandb.run is not None:
                wandb.finish(exit_code=1)

    # --- Aggregate and log final results across all runs ---
    if all_results:
        logging.info("===== FINAL RESULTS ACROSS ALL RUNS =====")
        
        # Get all metric keys from the first run
        metric_keys = all_results[0].keys()
        
        for metric in metric_keys:
            vals = [result[metric] for result in all_results]
            logging.info(f'{metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
    else:
        logging.warning("No runs completed successfully. No results to aggregate.")

    try:
        alert("cxp_pneu_mohsin_baseline_COMPLETE", "cxp_pneu_mohsin_baseline.py")
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")

if __name__ == '__main__':
    main()
