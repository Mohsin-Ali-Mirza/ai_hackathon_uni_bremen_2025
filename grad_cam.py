import os
import os.path
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.models import densenet121
from torcheval.metrics import BinaryAUROC
import torch.nn.functional as F

from captum.attr import LayerGradCam
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

NAME = f"mohsin_run_GradCAM_Model_Comparison"

def setup_logging(root_dir):
    log_path = root_dir / "cxr_pneu_gradcam_compare.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler 

    wandb_dir = root_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.abspath(wandb_dir)
    wandb.init(
        project="cxr_small_data_pneu",
        dir=wandb_dir,
        name=NAME,
        config={
            "analysis_type": "GradCAM_model_comparison"
        }
    )
    


class CXP_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = densenet121(weights='IMAGENET1K_V1')
        self.clf = nn.Linear(1000, 1)

    def forward(self, x):
        z = self.encode(x)
        return self.clf(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def predict_proba(self, x):
        logits = self(x)
        return torch.sigmoid(logits)

class CXP_dataset(torchvision.datasets.VisionDataset):

    def __init__(self, root_dir, csv_file=None, df=None, augment=True, inference_only=False) -> None:
        if augment:
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        super().__init__(root_dir, transform)

        if df is None:
            if csv_file is None:
                raise ValueError("Either 'csv_file' or 'df' must be provided.")
            df = pd.read_csv(csv_file)
        
        df = df.reset_index(drop=True)

        self.root_dir = root_dir
        self.path = df.Path.str.replace('CheXpert-v1.0/', 'CheXpert-v1.0-small/', regex=False)
        self.idx = df.index
        self.transform = transform

        self.labels = df.Pneumothorax.astype(int)
        self.drain = df.Drain.astype(int)

    def __getitem__(self, index: int):
        try:
            img = torchvision.io.read_image(os.path.join(self.root_dir, self.path[index]))
            img = self.transform(img)
            return img, self.labels[index], self.drain[index]
        except RuntimeError as e:
            logging.error(f"Error loading image at index {index}: {self.path[index]}")
            logging.error(f"Error message: {e}")
            return self.__getitem__((index + 1) % len(self)) 
    
    def __len__(self) -> int:
        return len(self.path)

# --- NEW HELPER FUNCTION ---
def get_model_outputs(model_path, test_images, device):
    """Loads a model and computes its predictions and Grad-CAM attributions."""
    
    logging.info(f"Loading model from: {model_path}")
    model = CXP_Model()
    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)

    checkpoint = torch.load(model_path, map_location=device)
    
    if 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        logging.info(f"Loaded 'ema_model_state_dict' for {model_path.name}.")
    elif 'model_state_dict' in checkpoint:
        ema_model.module.load_state_dict(checkpoint['model_state_dict'])
        logging.warning(f"Loaded 'model_state_dict' into EMA model for {model_path.name}.")
    else:
        logging.error(f"Could not find valid state_dict in {model_path.name}.")
        return None, None

    ema_model.eval()

    # Get model predictions
    with torch.no_grad():
        pred_logits = ema_model(test_images).reshape(-1).cpu().numpy()
        pred_probs = 1 / (1 + np.exp(-pred_logits)) # Sigmoid

    # Initialize Grad-CAM
    target_layer = ema_model.module.encoder.features.denseblock4
    lgc = LayerGradCam(ema_model, target_layer)

    logging.info(f"Generating Grad-CAM attributions for {model_path.name}...")
    attribution_maps = lgc.attribute(
        test_images,
        target=0,
        relu_attributions=True
    )
    attribution_maps = attribution_maps.detach().cpu()
    
    return pred_probs, attribution_maps

# --- MODIFIED FUNCTION ---
def run_gradcam_comparison(data_dir, csv_dir, out_dir, model_path_1, model_path_2, device):
    
    logging.info("Setting up data loaders for Grad-CAM comparison...")
    
    try:
        df_aligned = pd.read_csv(csv_dir / 'test_drain_shortcut_aligned.csv')
        df_misaligned = pd.read_csv(csv_dir / 'test_drain_shortcut_misaligned.csv')
    except FileNotFoundError as e:
        logging.error(f"Could not find test CSV files: {e}")
        wandb.finish()
        return

    RANDOM_STATE = 42

    # Sample 1 from each of the 4 groups
    sample_1 = df_aligned[df_aligned.Pneumothorax == 1].sample(1, random_state=RANDOM_STATE)
    sample_2 = df_aligned[df_aligned.Pneumothorax == 0].sample(1, random_state=RANDOM_STATE)
    sample_3 = df_misaligned[df_misaligned.Pneumothorax == 1].sample(1, random_state=RANDOM_STATE)
    sample_4 = df_misaligned[df_misaligned.Pneumothorax == 0].sample(1, random_state=RANDOM_STATE)

    df_samples = pd.concat([sample_1, sample_2, sample_3, sample_4])
    logging.info("Created dataset with 4 specific samples (all variations).")

    test_data = CXP_dataset(data_dir, df=df_samples, augment=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    
    # Get the one and only batch (containing our 4 samples)
    explain_batch = next(iter(test_loader))
    images_to_explain, labels_to_explain, drains_to_explain = explain_batch
    
    num_to_explain = 4
    test_images = images_to_explain.to(device)
    test_labels = labels_to_explain
    test_drains = drains_to_explain

    logging.info(f"Explaining {num_to_explain} images for Grad-CAM.")
    logging.info(f"  Labels: {test_labels.numpy()}")
    logging.info(f"  Drains: {test_drains.numpy()}")

    # --- Get outputs for both models ---
    pred_probs_1, attr_maps_1 = get_model_outputs(model_path_1, test_images, device)
    pred_probs_2, attr_maps_2 = get_model_outputs(model_path_2, test_images, device)
    
    if pred_probs_1 is None or pred_probs_2 is None:
        logging.error("Failed to get model outputs, aborting plot.")
        wandb.finish()
        return

    # Denormalize images for plotting
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    logging.info("Plotting Grad-CAM comparison...")

    # --- Create a 3-column subplot ---
    fig, axes = plt.subplots(
        nrows=num_to_explain,
        ncols=3, # Original, Model 1, Model 2
        figsize=(15, num_to_explain * 5) # Wider figure
    )
    
    if num_to_explain == 1:
        axes = np.array([axes])

    for i in range(num_to_explain):
        # Denormalize the image
        img_cpu = test_images[i].cpu().numpy()
        img_denorm = img_cpu * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1)
        img_denorm = np.clip(img_denorm, 0, 1)
        img_denorm_for_plot = img_denorm.transpose(1, 2, 0)

        # Get the i-th attribution map for both models
        attr_map_1 = attr_maps_1[i].squeeze().numpy()
        attr_map_2 = attr_maps_2[i].squeeze().numpy()

        # --- Column 0: Original Image ---
        title = f"Drain: {test_drains[i].item()}\nLabel: {test_labels[i].item()}"
        axes[i, 0].imshow(img_denorm_for_plot)
        axes[i, 0].set_title(f"Original Image\n{title}")
        axes[i, 0].axis('off')
        
        # --- Column 1: Grad-CAM for Model 1 ---
        title_1 = f"Model 1 ({model_path_1.name[:10]}...)\nProb: {pred_probs_1[i]:.2f}"
        axes[i, 1].imshow(img_denorm_for_plot)
        axes[i, 1].imshow(
            attr_map_1, 
            cmap='Reds', 
            alpha=0.6,
            interpolation='bilinear'
        )
        axes[i, 1].set_title(title_1)
        axes[i, 1].axis('off')
        
        # --- Column 2: Grad-CAM for Model 2 ---
        title_2 = f"Model 2 ({model_path_2.name[:10]}...)\nProb: {pred_probs_2[i]:.2f}"
        axes[i, 2].imshow(img_denorm_for_plot)
        axes[i, 2].imshow(
            attr_map_2, 
            cmap='Reds', 
            alpha=0.6,
            interpolation='bilinear'
        )
        axes[i, 2].set_title(title_2)
        axes[i, 2].axis('off')

    plot_filename = out_dir / f"{NAME}_gradcam_comparison.png"
    plt.suptitle("Grad-CAM Model Comparison (Red = Contributes to Pneumothorax)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
    plt.close()

    logging.info(f"Saved Grad-CAM comparison plot to {plot_filename}")
    
    wandb.log({"Grad-CAM Comparison": wandb.Image(str(plot_filename))})
    
    wandb.finish()
    logging.info("Grad-CAM comparison complete. Run finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Directory above /CheXpert-v1.0-small',
                        default='/data')
    parser.add_argument('--csv_dir', type=str, required=False,
                        help='Directory above that contains train_drain_shortcut.csv, etc.',
                        default='.')
    # --- FIX: Corrected typo 'add_pre_argument' to 'add_argument' ---
    parser.add_argument('--out_dir', type=str, required=False,
                        help='Directory where outputs (logs, plots, etc.) will be placed',
                        default='~/cxp_shortcut_out')
    
    # --- NEW: Accepting two model paths ---
    parser.add_argument('--model_path_1', type=str, required=True,
                        help='Path to the first pre-trained .chkpt model file')
    parser.add_argument('--model_path_2', type=str, required=True,
                        help='Path to the second pre-trained .chkpt model file')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir).expanduser()
    model_path_1 = Path(args.model_path_1).expanduser()
    model_path_2 = Path(args.model_path_2).expanduser()
    
    # Check if model paths are valid
    if not model_path_1.exists():
        logging.error(f"Model 1 file not found at: {model_path_1}")
        sys.exit(1)
    if not model_path_2.exists():
        logging.error(f"Model 2 file not found at: {model_path_2}")
        sys.exit(1)
    
    setup_logging(out_dir) 
    
    if torch.cuda.is_available():
        logging.info("Using GPU")
    else:
        logging.info("Using CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    run_gradcam_comparison(data_dir, csv_dir, out_dir, model_path_1, model_path_2, device)
