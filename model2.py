# References:
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
import math
from tqdm import tqdm
from pathlib import Path

from utils import get_device, save_image, image_to_grid


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_diffusion_steps, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(n_diffusion_steps).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [n_diffusion_steps, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [n_diffusion_steps, d_model // 2, 2]
        emb = emb.view(n_diffusion_steps, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
    #     self.initialize()

    # def initialize(self):
    #     for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
    #         nn.init.xavier_uniform_(module.weight)
    #         nn.init.zeros_(module.bias)
    #     nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, n_diffusion_steps=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1):
        super().__init__()

        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(
            n_diffusion_steps=n_diffusion_steps, d_model=ch, dim=tdim,
        )

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        cxs = [ch]  # record output channel when dowmsample for upsample
        cur_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=cur_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn)
                    )
                )
                cur_ch = out_ch
                cxs.append(cur_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(cur_ch))
                cxs.append(cur_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=True),
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=cxs.pop() + cur_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                cur_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(cur_ch))
        assert len(cxs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            Swish(),
            nn.Conv2d(cur_ch, 3, kernel_size=3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, noisy_image, diffusion_step):
        temb = self.time_embedding(diffusion_step)
        x = self.head(noisy_image)
        xs = [x]
        for layer in self.downblocks:
            x = layer(x, temb)
            xs.append(x)

        for layer in self.middleblocks:
            x = layer(x, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, xs.pop()], dim=1)

            if isinstance(layer, UpSample):
                x = layer(x)
            else:
                x = layer(x, temb)
        x = self.tail(x)
        assert len(xs) == 0
        return x


class DDIM(nn.Module):
    def get_linear_beta_schdule(self):
        return torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )

    def __init__(
        self,
        img_size,
        device,
        image_channels=3,
        n_ddim_diffusion_steps=50,
        ddim_eta=0,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.ddim_eta = ddim_eta
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.ddim_diffusion_step_size = n_diffusion_steps // n_ddim_diffusion_steps

        self.beta = self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.model = UNet(n_diffusion_steps=n_diffusion_steps).to(device)

    @staticmethod
    def index(x, diffusion_step):
        return torch.index_select(
            x,
            dim=0,
            index=torch.maximum(diffusion_step, torch.zeros_like(diffusion_step)),
        )[:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def forward(self, noisy_image, diffusion_step):
        return self.model(noisy_image=noisy_image, diffusion_step=diffusion_step)

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )

        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        prev_alpha_bar_t = self.index(
            self.alpha_bar, diffusion_step=diffusion_step - self.ddim_diffusion_step_size,
        )
        
        pred_noise = self(noisy_image=noisy_image, diffusion_step=diffusion_step)
        pred_ori_image = (
            noisy_image - (1 - alpha_bar_t) ** 0.5 * pred_noise
        ) / (alpha_bar_t ** 0.5)

        ddim_sigma_t = self.ddim_eta * (
            (1 - prev_alpha_bar_t) / (1 - alpha_bar_t) * (1 - alpha_bar_t / prev_alpha_bar_t)
        ) ** 0.5
        dir_xt = ((1 - prev_alpha_bar_t - ddim_sigma_t ** 2) ** 0.5) * pred_noise

        random_noise = self.sample_noise(batch_size=noisy_image.size(0))
        denoised_image = (prev_alpha_bar_t ** 0.5) * pred_ori_image + dir_xt + ddim_sigma_t * random_noise
        return denoised_image

    def perform_denoising_process(self, noisy_image):
        x = noisy_image
        for diffusion_step_idx in  reversed(
            range(0, self.n_diffusion_steps, self.ddim_diffusion_step_size),
        ):
            x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)
        return x

    def sample(self, batch_size):
        random_noise = self.sample_noise(batch_size=batch_size)
        return self.perform_denoising_process(noisy_image=random_noise)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()

    model = DDIM(img_size=32, n_ddim_diffusion_steps=20, ddim_eta=0.5, device=DEVICE)
    model_params_path = "/Users/jongbeomkim/Downloads/ddpm_celeba_32×32.pth"
    state_dict = torch.load(str(model_params_path), map_location=DEVICE)
    model.load_state_dict(state_dict["model"])
    # model.ddim_diffusion_step
    # for ddim_diffusion_step_idx in reversed(range(0, model.n_diffusion_steps, model.n_diffusion_steps // model.n_ddim_diffusion_steps))
    # list(range(0, model.n_diffusion_steps, model.n_diffusion_steps // model.n_ddim_diffusion_steps))

    gen_image = model.sample(batch_size=36)
    gen_grid = image_to_grid(gen_image, n_cols=6)
    gen_grid.show()
