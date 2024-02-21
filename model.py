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
    # "We set T = 1000 without a sweep."
    # "We chose a linear schedule from $\beta_{1} = 10^{-4}$ to  $\beta_{T} = 0:02$."
    def __init__(self, device, n_diffusion_steps=1000, n_ddim_diffusion_steps=50, ddim_eta=0, init_beta=0.0001, fin_beta=0.02):
        super().__init__()

        self.device = device
        self.n_diffusion_steps = n_diffusion_steps
        self.n_ddim_diffusion_steps = n_ddim_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.ddim_diffusion_step = torch.arange(
            0, n_diffusion_steps, n_diffusion_steps // n_ddim_diffusion_steps,
        )
        # print(self.ddim_diffusion_step)
        self.beta = self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.ddim_alpha_bar = self.alpha_bar[self.ddim_diffusion_step]
        # print(self.ddim_alpha_bar)
        self.ddim_prev_alpha_bar = torch.cat([self.alpha_bar[: 1], self.ddim_alpha_bar[: -1]], dim=0)
        self.ddim_sigma = ddim_eta * ((1 - self.ddim_prev_alpha_bar) / (1 - self.ddim_alpha_bar) * (1 - self.ddim_alpha_bar / self.ddim_prev_alpha_bar)) ** 0.5
        # print(self.ddim_sigma)

        self.model = UNet(n_diffusion_steps=n_diffusion_steps).to(device)

    def get_linear_beta_schdule(self):
        return torch.linspace(self.init_beta, self.fin_beta, self.n_diffusion_steps, device=self.device)

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step].view(-1, 1, 1, 1)

    def sample_noise(self, batch_size, n_channels, img_size):
        return torch.randn(size=(batch_size, n_channels, img_size, img_size), device=self.device)

    def forward(self, noisy_image, diffusion_step):
        return self.model(noisy_image=noisy_image, diffusion_step=diffusion_step)

    def get_prev_noisy_image(self, noisy_image, diffusion_step, ddim_diffusion_step_idx):
        # print(ddim_diffusion_step_idx, diffusion_step)
        pred_noise = self(noisy_image=noisy_image, diffusion_step=diffusion_step)

        # alpha_bar_t = self.ddim_alpha_bar[ddim_diffusion_step_idx]
        alpha_bar_t = self.index(
            self.ddim_alpha_bar, diffusion_step=ddim_diffusion_step_idx,
        )
        # prev_alpha_bar_t = self.ddim_prev_alpha_bar[ddim_diffusion_step_idx]
        prev_alpha_bar_t = self.index(
            self.ddim_alpha_bar, diffusion_step=ddim_diffusion_step_idx - 1,
        )
        print(prev_alpha_bar_t.item(), self.ddim_prev_alpha_bar[ddim_diffusion_step_idx].item())
        # print(f"{ddim_diffusion_step_idx}, {alpha_bar_t.item():.5f}, {prev_alpha_bar_t.item():.5f}")
        sigma_t = self.ddim_sigma[ddim_diffusion_step_idx]
        pred_ori_image = (noisy_image - (1 - alpha_bar_t) ** 0.5 * pred_noise) / (alpha_bar_t ** 0.5)
        # print("C")
        dir_xt = ((1 - prev_alpha_bar_t - sigma_t ** 2) ** 0.5) * pred_noise

        if sigma_t == 0:
            random_noise = 0
        else:
            random_noise = torch.randn(noisy_image.shape, device=noisy_image.device)
        prev_noisy_image = (prev_alpha_bar_t ** 0.5) * pred_ori_image + dir_xt + sigma_t * random_noise
        return prev_noisy_image

    def sample(self, batch_size, n_channels, img_size): # Reverse (denoising) process
        x = self.sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size)
        for idx, cur_ddim_diffusion_step in tqdm(enumerate(torch.flip(self.ddim_diffusion_step, dims=(0,))), total=self.n_ddim_diffusion_steps):
            # print(cur_ddim_diffusion_step, idx)
            batch_cur_ddim_diffusion_step = torch.full(
                size=(batch_size,), fill_value=cur_ddim_diffusion_step.item(), dtype=torch.long, device=self.device,
            )
            x = self.get_prev_noisy_image(x, batch_cur_ddim_diffusion_step, self.ddim_diffusion_step.size(0) - idx - 1)
        return x


if __name__ == "__main__":
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()

    model = DDIM(n_ddim_diffusion_steps=20, ddim_eta=0.5, device=DEVICE)
    model_params_path = "/Users/jongbeomkim/Downloads/ddpm_celeba_32×32.pth"
    state_dict = torch.load(str(model_params_path), map_location=DEVICE)
    model.load_state_dict(state_dict["model"])

    # model.ddim_alpha_bar[-5:]
    # model.ddim_prev_alpha_bar[-5:]
    gen_image = model.sample(
        batch_size=2,
        n_channels=3,
        img_size=32,
        # device=DEVICE,
    )
    gen_grid = image_to_grid(gen_image, n_cols=1)
    gen_grid.show()
