

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_EPOCH = 10
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCH = 500
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.0
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC_Z = "disc_z.pth.tar"
CHECKPOINT_DISC_H = "disc_h.pth.tar"
CHECKPOINT_GEN_Z = "gen_z.pth.tar"
CHECKPOINT_GEN_H = "gen_h.pth.tar"


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ],
    additional_targets={'image0': 'image'},
)
