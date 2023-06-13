import os

import torch
import torch.optim as optim
import config
from pix2pix.custom.dataset import MapDataset
from pix2pix.custom.generator_model import Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def load_checkpoint(filename, model):
    device = config.DEVICE
    data_gen = torch.load(filename, map_location=device)
    # print(data_gen)
    # print(data_gen.keys())
    print("=> Loading checkpoint")
    # checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(data_gen["model"])
    # model.load_state_dict(data_gen["state_dict"])


def run_predictions(gen, folder):
    val_dataset = MapDataset(root_dir=config.VAL_DIR_ALL, AtoB=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    loop = tqdm(val_loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake, folder + f"/{idx+1}.png")

        # break


def run(model_path, folder):
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)

    load_checkpoint(model_path, gen)

    if not os.path.exists(folder):
        os.makedirs(folder)

    run_predictions(gen, folder)


if __name__ == "__main__":
    run('models/gen_fed_10.pt', 'generated/gen_fed_10')
