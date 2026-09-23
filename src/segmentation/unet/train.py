# train_unet.py
import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    MapTransform,
)
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='monai.transforms')

# Import UNet
from monai.networks.nets import UNet

# ============================================================================
# Configuration - UNet for BraTS 2023
# ============================================================================

class Config:
    # Data
    train_json = "../train_list.json"
    val_json = "../val_list.json"

    # Model - UNet parameters
    roi_size = (128, 128, 128)  # Patch size
    in_channels = 4              # T1n, T1c, T2w, T2f
    out_channels = 3             # ET, TC, WT
    
    # UNet architecture
    spatial_dims = 3
    channels = (16, 32, 64, 128, 256)  # Feature channels per layer
    strides = (2, 2, 2, 2)              # Downsampling strides
    num_res_units = 2                   # Residual units per block
    kernel_size = 3
    up_kernel_size = 3
    dropout = 0.0
    
    # Training
    max_epochs = 300
    batch_size = 1
    learning_rate = 1e-4
    weight_decay = 1e-5

    # Data loading
    num_workers = 2
    cache_rate = 0.0

    # Validation
    val_interval = 10

    # Inference
    sw_batch_size = 2
    overlap = 0.5

    # Logging
    log_dir = "./logs_unet"
    checkpoint_dir = "./checkpoints_unet"
    log_interval = 20

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True

config = Config()

# ============================================================================
# Logger Class (Same as SwinUNETR/SegResNet)
# ============================================================================

class TrainingLogger:
    """Comprehensive training logger"""

    def __init__(self, log_dir, checkpoint_dir):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

        self.train_losses = []
        self.val_dices = []
        self.best_dice = 0
        self.start_time = time.time()

        self.log("="*80)
        self.log("Training Session Started - UNet")
        self.log(f"Timestamp: {timestamp}")
        self.log("="*80)

    def log(self, message, print_msg=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
        if print_msg:
            print(message)

    def log_config(self, config):
        self.log("\nConfiguration:")
        self.log("-" * 80)
        for key, value in vars(config).items():
            self.log(f"  {key}: {value}")
        self.log("-" * 80)

    def log_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log("\nModel Information:")
        self.log("-" * 80)
        self.log(f"  Architecture: UNet")
        self.log(f"  Total Parameters: {total_params:,}")
        self.log(f"  Trainable Parameters: {trainable_params:,}")
        self.log(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (float32)")
        self.log("-" * 80)

    def log_epoch_start(self, epoch, max_epochs):
        self.log(f"\n{'='*80}")
        self.log(f"Epoch {epoch}/{max_epochs}")
        self.log(f"{'='*80}")
        self.epoch_start_time = time.time()

    def log_batch(self, epoch, batch_idx, total_batches, loss, lr):
        self.writer.add_scalar("Batch/Loss", loss, epoch * total_batches + batch_idx)
        self.writer.add_scalar("Batch/LearningRate", lr, epoch * total_batches + batch_idx)

    def log_epoch_end(self, epoch, train_loss, lr):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        self.train_losses.append(train_loss)

        self.log(f"\nEpoch {epoch} Summary:")
        self.log(f"  Training Loss: {train_loss:.6f}")
        self.log(f"  Learning Rate: {lr:.6e}")
        self.log(f"  Epoch Time: {epoch_time:.2f}s")
        self.log(f"  Total Time: {total_time/3600:.2f}h")

        self.writer.add_scalar("Epoch/TrainLoss", train_loss, epoch)
        self.writer.add_scalar("Epoch/LearningRate", lr, epoch)
        self.writer.add_scalar("Epoch/EpochTime", epoch_time, epoch)

    def log_validation(self, epoch, dice_scores):
        et_dice = dice_scores[0]
        tc_dice = dice_scores[1]
        wt_dice = dice_scores[2]
        avg_dice = dice_scores.mean()

        self.val_dices.append({
            'epoch': epoch,
            'et': float(et_dice),
            'tc': float(tc_dice),
            'wt': float(wt_dice),
            'avg': float(avg_dice)
        })

        self.log(f"\n{'─'*80}")
        self.log("Validation Results:")
        self.log(f"  ET (Enhancing Tumor): {et_dice:.4f}")
        self.log(f"  TC (Tumor Core):      {tc_dice:.4f}")
        self.log(f"  WT (Whole Tumor):     {wt_dice:.4f}")
        self.log(f"  Average Dice:         {avg_dice:.4f}")

        is_best = avg_dice > self.best_dice
        if is_best:
            improvement = avg_dice - self.best_dice
            self.best_dice = avg_dice
            self.log(f"  ✓ New best model! (↑ {improvement:.4f})")
        else:
            self.log(f"  Best remains: {self.best_dice:.4f}")
        self.log(f"{'─'*80}")

        self.writer.add_scalar("Validation/ET_Dice", et_dice, epoch)
        self.writer.add_scalar("Validation/TC_Dice", tc_dice, epoch)
        self.writer.add_scalar("Validation/WT_Dice", wt_dice, epoch)
        self.writer.add_scalar("Validation/Avg_Dice", avg_dice, epoch)

        return is_best

    def log_checkpoint_saved(self, epoch, checkpoint_type="regular"):
        self.log(f"  ✓ Saved {checkpoint_type} checkpoint (Epoch {epoch})")

    def log_training_complete(self):
        total_time = time.time() - self.start_time
        self.log(f"\n{'='*80}")
        self.log("Training Complete!")
        self.log(f"{'='*80}")
        self.log(f"Total Training Time: {total_time/3600:.2f} hours")
        self.log(f"Best Average Dice: {self.best_dice:.4f}")
        self.log(f"Total Epochs: {len(self.train_losses)}")
        self.log(f"Best Model: {self.checkpoint_dir}/best_model.pth")
        self.log(f"Logs: {self.log_file}")
        self.log(f"TensorBoard: {self.log_dir}/tensorboard")
        self.log("="*80)
        self.save_training_summary()

    def save_training_summary(self):
        summary = {
            'best_dice': float(self.best_dice),
            'total_epochs': len(self.train_losses),
            'final_train_loss': float(self.train_losses[-1]) if self.train_losses else 0,
            'validation_history': self.val_dices,
            'total_time_hours': (time.time() - self.start_time) / 3600
        }
        summary_path = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.log(f"Training summary saved: {summary_path}")

    def close(self):
        self.writer.close()

# ============================================================================
# Data Transforms (Same as SwinUNETR/SegResNet)
# ============================================================================

class ConvertBRaTSLabelsd(MapTransform):
    """Convert BraTS labels to binary format for ET, TC, WT"""
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            result = torch.zeros((3, *label.shape[1:]), dtype=label.dtype, device=label.device)
            
            # Channel 0: ET (Enhancing Tumor) - label 3
            result[0] = (label[0] == 3)
            
            # Channel 1: TC (Tumor Core) - labels 1 and 3
            result[1] = torch.logical_or(label[0] == 1, label[0] == 3)
            
            # Channel 2: WT (Whole Tumor) - labels 1, 2, and 3
            result[2] = torch.logical_or(
                torch.logical_or(label[0] == 1, label[0] == 2),
                label[0] == 3
            )
            
            d[key] = result.float()
        
        return d

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ConvertBRaTSLabelsd(keys=["label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=config.roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ConvertBRaTSLabelsd(keys=["label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

# ============================================================================
# Create Model - UNet
# ============================================================================

def create_model():
    """
    Create UNet model
    
    Based on Ronneberger et al. 2015:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    
    Enhanced with residual units as described in:
    https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40
    """
    model = UNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        kernel_size=config.kernel_size,
        up_kernel_size=config.up_kernel_size,
        num_res_units=config.num_res_units,
        act="PRELU",
        norm="INSTANCE",
        dropout=config.dropout,
        bias=True,
        adn_ordering="NDA",
    )
    return model

# ============================================================================
# Training Functions (Same as SwinUNETR/SegResNet)
# ============================================================================

def train_epoch(model, loader, optimizer, loss_fn, scaler, epoch, logger):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(loader, desc=f"Training Epoch {epoch}")

    for batch_idx, batch_data in enumerate(progress_bar):
        inputs = batch_data["image"].to(config.device)
        labels = batch_data["label"].to(config.device)

        optimizer.zero_grad()

        with autocast(enabled=config.use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Periodic cache clearing
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        epoch_loss += loss.item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{epoch_loss/(batch_idx+1):.4f}"
        })

        if batch_idx % config.log_interval == 0:
            logger.log_batch(
                epoch, batch_idx, len(loader),
                loss.item(),
                optimizer.param_groups[0]['lr']
            )

    return epoch_loss / len(loader)

def validate(model, loader, logger):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    progress_bar = tqdm(loader, desc="Validation")

    with torch.no_grad():
        for batch_data in progress_bar:
            inputs = batch_data["image"].to(config.device)
            labels = batch_data["label"].to(config.device)

            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=config.roi_size,
                sw_batch_size=config.sw_batch_size,
                predictor=model,
                overlap=config.overlap,
            )

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            dice_metric(y_pred=outputs, y=labels)
            
            torch.cuda.empty_cache()

    mean_dice = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()
    return mean_dice

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    torch.cuda.empty_cache()

    logger = TrainingLogger(config.log_dir, config.checkpoint_dir)

    logger.log("="*80)
    logger.log("UNet Training - BraTS 2023")
    logger.log("="*80)
    logger.log(f"Device: {config.device}")

    if config.device == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    logger.log_config(config)

    # Load data lists
    logger.log("\nLoading data lists...")
    with open(config.train_json) as f:
        train_files = json.load(f)
    with open(config.val_json) as f:
        val_files = json.load(f)

    logger.log(f"  Training samples: {len(train_files)}")
    logger.log(f"  Validation samples: {len(val_files)}")

    # Create datasets
    logger.log("\nCreating datasets...")
    train_ds = Dataset(data=train_files, transform=get_train_transforms())
    val_ds = Dataset(data=val_files, transform=get_val_transforms())
    logger.log("  ✓ Datasets created")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create model
    logger.log("\nCreating model...")
    model = create_model()
    model = model.to(config.device)
    logger.log_model_info(model)

    # Create optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
    )

    loss_fn = DiceLoss(
        to_onehot_y=False,
        sigmoid=True,
        squared_pred=True,
    )

    scaler = GradScaler(enabled=config.use_amp)

    logger.log("\n✓ Training setup complete")
    logger.log("\nStarting training...")

    # Training loop
    try:
        for epoch in range(1, config.max_epochs + 1):
            logger.log_epoch_start(epoch, config.max_epochs)

            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, scaler, epoch, logger
            )

            current_lr = optimizer.param_groups[0]['lr']
            logger.log_epoch_end(epoch, train_loss, current_lr)

            # Validate
            if epoch % config.val_interval == 0:
                dice_scores = validate(model, val_loader, logger)
                is_best = logger.log_validation(epoch, dice_scores)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'dice_scores': {
                        'et': float(dice_scores[0]),
                        'tc': float(dice_scores[1]),
                        'wt': float(dice_scores[2]),
                        'avg': float(dice_scores.mean()),
                    },
                }

                if is_best:
                    save_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
                    torch.save(checkpoint, save_path)
                    logger.log_checkpoint_saved(epoch, "best")

            scheduler.step()

            # Save checkpoint every 50 epochs
            if epoch % 50 == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                torch.save(checkpoint, checkpoint_path)
                logger.log_checkpoint_saved(epoch, f"epoch_{epoch}")
            
            torch.cuda.empty_cache()

        logger.log_training_complete()

    except KeyboardInterrupt:
        logger.log("\n\nTraining interrupted by user!")
        logger.log_training_complete()

    except Exception as e:
        logger.log(f"\n\nError during training: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise

    finally:
        logger.close()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
