# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import random
import numpy as np
import os
import re


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_grad_scaler(device):
    return GradScaler() if device.type == "cuda" else None


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(str(save_path), quality=100)


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def denorm(tensor):
    tensor /= 2
    tensor += 0.5
    return tensor


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def modify_state_dict(state_dict, pattern=r"^module.|^_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key, value in state_dict.items():
        new_key = re.sub(pattern=pattern, repl="", string=old_key)
        new_state_dict[new_key] = value
    return new_state_dict


def print_n_prams(model):
    n_params = 0
    n_trainable_params = 0
    for p in model.parameters():
        n = p.numel()
        n_params += n
        if p.requires_grad:
            n_trainable_params += n
    print(f"[ # OF PARAMS: {n_params:,} ][ # OF TRAINABLE PARAMS: {n_trainable_params:,} ]")




def get_linear_beta_schdule(init_beta, fin_beta, n_timesteps):
    # "We set the forward process variances to constants increasing linearly."
    # return torch.linspace(init_beta, fin_beta, n_timesteps) # "$\beta_{t}$"
    return torch.linspace(init_beta, fin_beta, n_timesteps + 1) # "$\beta_{t}$"


def index(x, t):
    return x[t].view(-1, 1, 1, 1)


def sample_noise(batch_size, n_channels, img_size, device):
    return torch.randn(batch_size, n_channels, img_size, img_size, device=device)


def sample_t(n_timesteps, batch_size, device):
    return torch.randint(low=0, high=n_timesteps, size=(batch_size,), device=device)


def modify_state_dict(state_dict, keyword="_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(keyword):
            new_key = old_key[len(keyword):]
        else:
            new_key = old_key
        new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict