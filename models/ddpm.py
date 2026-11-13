import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=device
            ) * -scale
        )
        embeddings = t.float()[:, None] * frequencies[None, :]
        embeddings = torch.cat(
            [torch.sin(embeddings), torch.cos(embeddings)],
            dim=-1
        )
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        residual = self.shortcut(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        time_features = self.time_projection(time_embedding)
        time_features = time_features[:, :, None, None]
        h = h + time_features
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + residual)
 
class SimpleUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.stem = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(128, 128, time_dim)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualBlock(256, 256, time_dim)
        self.mid = ResidualBlock(256, 256, time_dim)
        self.up1_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up1 = ResidualBlock(256, 128, time_dim)
        self.up2_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up2 = ResidualBlock(128, 64, time_dim)
        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        time_embedding = self.time_embedding(t)
        # 28 x 28
        h0 = self.stem(x)
        # 14 x 14
        h1 = self.down1(h0)
        h1 = self.res1(h1, time_embedding)
        # 7 x 7
        h2 = self.down2(h1)
        h2 = self.res2(h2, time_embedding)
        h = self.mid(h2, time_embedding)
        # 7 x 7 -> 14 x 14
        h = F.interpolate(h, size=h1.shape[-2:], mode="nearest")
        h = self.up1_conv(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, time_embedding)
        # 14 x 14 -> 28 x 28
        h = F.interpolate(h, size=h0.shape[-2:], mode="nearest")
        h = self.up2_conv(h)
        h = torch.cat([h, h0], dim=1)
        h = self.up2(h, time_embedding)

        return self.output(h)

class DDPM(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.model = SimpleUNet()
        #linear noise
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def predict_noise(self, x, t):
        return self.model(x, t)

    #forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_bar = self.alpha_bars[t]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        noisy_image = (torch.sqrt(alpha_bar) * x_start + torch.sqrt(1.0 - alpha_bar) * noise)

        return noisy_image, noise

    #reverse diffusion
    @torch.no_grad()
    def p_sample(self, x, t):
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        predicted_noise = self.predict_noise(x, t)
        mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise) / torch.sqrt(alpha_t)

        if t[0].item() == 0:
            return mean

        noise = torch.randn_like(x)
        return mean + torch.sqrt(beta_t) * noise

    #generate
    @torch.no_grad()
    def sample(self, num_samples, device):
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), timestep, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1, 1)