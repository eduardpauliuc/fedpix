import os

import torch
import config
from torchvision.utils import save_image


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 0:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_results(results, folder):
    filename = f"results.txt"
    path = os.path.join(folder, filename)

    with open(path, "a+") as f:
        # Writing data to a file
        print(results)
        for result in results:
            f.write(" ".join([str(x) for x in result]) + '\n')
        f.close()
