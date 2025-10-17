import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

# keep your original FOD imports
from FOD.Trainer import Trainer
# now import the new dataset class
from FOD.dataset_focalstack import AutoFocusStackDataset  # <-- new dataset

# load configuration
with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])
torch.manual_seed(config['General']['seed'])

# list of datasets to use (from config)
list_data = config['Dataset']['paths']['list_datasets']

########################################
# TRAIN SET
########################################
autofocus_datasets_train = []
for dataset_name in list_data:
    dataset = AutoFocusStackDataset(config, dataset_name, 'train')
    autofocus_datasets_train.append(dataset)

train_data = ConcatDataset(autofocus_datasets_train)

train_dataloader = DataLoader(
    train_data,
    batch_size=config['General']['batch_size'],
    shuffle=True,
    num_workers=config['General'].get('num_workers', 4),
    pin_memory=True
)

########################################
# VALIDATION SET
########################################
autofocus_datasets_val = []
for dataset_name in list_data:
    dataset = AutoFocusStackDataset(config, dataset_name, 'val')
    autofocus_datasets_val.append(dataset)

val_data = ConcatDataset(autofocus_datasets_val)

val_dataloader = DataLoader(
    val_data,
    batch_size=config['General']['batch_size'],
    shuffle=False,
    num_workers=config['General'].get('num_workers', 4),
    pin_memory=True
)

########################################
# TRAINING LOOP
########################################
trainer = Trainer(config)

# Optional: sanity check one batch before training
if config['General'].get('debug_batch_shape', False):
    batch = next(iter(train_dataloader))
    img_stack, depth = batch
    print(f"Batch shapes -> img_stack: {img_stack.shape}, depth: {depth.shape}")
    # expected: [B, N_focus, 3, H, W], [B, 1, H, W]

# Start training
trainer.train(train_dataloader, val_dataloader)
