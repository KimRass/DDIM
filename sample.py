import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/DDIM")

import torch
import argparse

from utils import get_device, save_image, image_to_grid
from model import DDIM
from ddpm import DDPM

torch.set_printoptions(linewidth=70)


if __name__ == "__main__":
    DEVICE = get_device()

    # model = DDIM(n_ddim_diffusion_steps=50, device=DEVICE)
    model = DDIM(n_ddim_diffusion_steps=20, device=DEVICE)
    # model = DDPM().to(DEVICE)
    model_params_path = "/Users/jongbeomkim/Downloads/ddpm_celeba_32×32.pth"
    state_dict = torch.load(str(model_params_path), map_location=DEVICE)
    model.load_state_dict(state_dict["model"])

    # model.ddim_alpha_bar[-5:]
    # model.ddim_prev_alpha_bar[-5:]
    gen_image = model.sample(
        batch_size=1,
        n_channels=3,
        img_size=32,
        # device=DEVICE,
    )

    gen_grid = image_to_grid(gen_image, n_cols=1)
    gen_grid.show()
