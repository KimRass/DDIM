# References:
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16

import torch
from torch import nn
from tqdm import tqdm
from scipy.stats import truncnorm

from data import CelebADS


# "We consider two types of selection procedure for given the desired dim(  ) < T: • Linear: we select the timesteps such that  i = bcic for some c; • Quadratic: we select the timesteps such that  i = bci2c for some c. The constant value c is selected such that  􀀀1 is close to T. We used quadratic for CIFAR10 and linear for the remaining datasets. These choices achieve slightly better FID than their alternatives in the respective datasets."
# "현재 Linear만 구현되어 있습니다."
class DDIM(nn.Module):
    def get_linear_beta_schdule(self):
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )

    def __init__(
        self,
        model,
        img_size,
        device,
        image_channels=3,
        n_ddim_steps=50,
        eta=0,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_ddim_steps = n_ddim_steps
        self.eta = eta
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.model = model.to(device)

        self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.ddim_step_size = self.n_diffusion_steps // n_ddim_steps

    @staticmethod
    def index(x, diffusion_step):
        return torch.index_select(
            x,
            dim=0,
            index=torch.maximum(diffusion_step, torch.zeros_like(diffusion_step)),
        )[:, None, None, None]

    def sample_from_trunc_normal(self, size, thresh=1):
        x = torch.randn(size=size, device=self.device)
        while True:
            mask = (-thresh <= x) & (x <= thresh)
            if torch.all(mask).item():
                break
            x[~mask] = torch.randn_like(x, device=self.device)[~mask]
        return x

    def sample_from_trunc_normal2(self, batch_size, thresh=1):
        return torch.tensor(
            truncnorm.rvs(
                -thresh,
                thresh,
                size=(batch_size, self.image_channels, self.img_size, self.img_size)
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def sample_noise(self, batch_size, thresh=None):
        if thresh is None:
            return torch.randn(
                size=(batch_size, self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        else:
            return self.sample_from_trunc_normal(
                size=(batch_size, self.image_channels, self.img_size, self.img_size),
                thresh=thresh,
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
    def predict_ori_image(self, noisy_image, noise, alpha_bar_t):
        # "$\frac{x_{t} - \sqrt{1 - \alpha_{t}}\epsilon_{\theta}^{(t)}(x_{t})}{\sqrt{\alpha_{t}}}$"
        return (noisy_image - ((1 - alpha_bar_t) ** 0.5) * noise) / (alpha_bar_t ** 0.5)

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )

        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        prev_alpha_bar_t = self.index(
            self.alpha_bar, diffusion_step=diffusion_step - self.ddim_step_size,
        )

        pred_noise = self(noisy_image=noisy_image, diffusion_step=diffusion_step)
        pred_ori_image  = self.predict_ori_image(
            noisy_image=noisy_image, noise=pred_noise, alpha_bar_t=alpha_bar_t,
        )

        sigma_t = self.eta * (
            (1 - prev_alpha_bar_t) / (1 - alpha_bar_t) * (1 - alpha_bar_t / prev_alpha_bar_t)
        ) ** 0.5
        # "$\sqrt{1 - \alpha_{t - 1} - \sigma_{t}^{2}} \cdot \epsilon_{\theta}^{(t)}(x_{t})$"
        dir_xt = ((1 - prev_alpha_bar_t - sigma_t ** 2) ** 0.5) * pred_noise
        model_mean = (prev_alpha_bar_t ** 0.5) * pred_ori_image + dir_xt

        model_var = sigma_t ** 2
        # "We define $\alpha_{0} := 1$."
        if diffusion_step_idx > 0:
            rand_noise = self.sample_noise(batch_size=noisy_image.size(0))
        else:
            rand_noise = torch.zeros(
                size=(noisy_image.size(0), self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        return model_mean + (model_var ** 0.5) * rand_noise

    def perform_denoising_process(self, noisy_image):
        x = noisy_image
        pbar = tqdm(
            reversed(range(0, self.n_diffusion_steps, self.ddim_step_size)),
            total=self.n_ddim_steps,
            leave=False,
        )
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)
        return x

    def sample(self, batch_size, thresh=None):
        rand_noise = self.sample_noise(batch_size=batch_size, thresh=thresh)
        return self.perform_denoising_process(noisy_image=rand_noise)

    def get_ori_images(self, data_dir, image_idx1, image_idx2):
        test_ds = CelebADS(
            data_dir=data_dir, split="test", img_size=self.img_size, hflip=False,
        )
        ori_image1 = test_ds[image_idx1][None, ...].to(self.device)
        ori_image2 = test_ds[image_idx2][None, ...].to(self.device)
        return ori_image1, ori_image2

    @staticmethod
    def inner_product(x, y):
        return torch.sum(x * y, dim=(1, 2, 3))

    def get_angle(self, x, y):
        return torch.arccos(
            self.inner_product(x, y) / (
                (self.inner_product(x, x) ** 0.5) * (self.inner_product(y, y) ** 0.5)
            )
        )[:, None, None, None]

    def get_interpolation_weight(self, n_points):
        return torch.linspace(
            start=0, end=1, steps=n_points, device=self.device,
        )[:, None, None, None]

    def _get_spherically_interpolated_rand_noise(self, n_points, thresh=None):
        rand_noise1 = self.sample_noise(batch_size=1, thresh=thresh)
        rand_noise2 = self.sample_noise(batch_size=1, thresh=thresh)
        ang = self.get_angle(rand_noise1, rand_noise2)
        weight = self.get_interpolation_weight(n_points)
        x_weight = torch.sin((1 - weight) * ang) / torch.sin(ang)
        y_weight = torch.sin(weight * ang) / torch.sin(ang)
        return x_weight * rand_noise1 + y_weight * rand_noise2

    def interpolate_in_latent_space(self, n_points=10, thresh=None):
        rand_noise = self._get_spherically_interpolated_rand_noise(
            n_points=n_points, thresh=thresh,
        )
        return self.perform_denoising_process(rand_noise)

    def interpolate_on_grid(self, n_rows=5, n_cols=10, thresh=None):
        rand_noise1 = self._get_spherically_interpolated_rand_noise(
            n_points=n_rows, thresh=thresh,
        )
        rand_noise2 = self._get_spherically_interpolated_rand_noise(
            n_points=n_rows, thresh=thresh,
        )
        ang = self.get_angle(rand_noise1, rand_noise2)
        weight = self.get_interpolation_weight(n_cols)
        x_weight = torch.sin((1 - weight) * ang.unsqueeze(1)) / torch.sin(ang.unsqueeze(1))
        y_weight = torch.sin(weight * ang.unsqueeze(1)) / torch.sin(ang.unsqueeze(1))
        rand_noise = x_weight * rand_noise1.unsqueeze(1) + y_weight * rand_noise2.unsqueeze(1)
        image = self.perform_denoising_process(torch.flatten(rand_noise, start_dim=0, end_dim=1))
        return image
