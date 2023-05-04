import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ROOT_DIR = "/Users/eduardpauliuc/PycharmProjects/federated"
ROOT_DIR = "/content/drive/MyDrive/Licenta"
TRAIN_DIR = ROOT_DIR + "/maps/train"
VAL_DIR = ROOT_DIR + "/maps/val"
MODELS_DIR = ROOT_DIR + "/saved_models"
EVALUATION_DIR = ROOT_DIR + "/maps/evaluation"
RESULTS_DIR = ROOT_DIR + "/results"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
LOAD_MODEL = False
SAVE_MODEL = True

both_transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

transform_only_input_no_jitter = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)
