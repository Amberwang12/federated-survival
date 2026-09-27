import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        return torch.relu(self.linear2(x))


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.linear1(x))
        return self.linear2(x)


class DecoderTime(nn.Module):
    """时间解码器"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super(DecoderTime, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return torch.relu(self.linear3(x))


class MVAE(nn.Module):
    """变分自编码器"""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        decoder_time: DecoderTime,
        latent_dim: int,
        encoder_out: int,
    ):
        super(MVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_time = decoder_time
        self.latent_dim = latent_dim

        # 两个全连接层用于生成均值和方差
        self._enc_mu = nn.Linear(encoder_out, latent_dim)
        self._enc_log_sigma = nn.Linear(encoder_out, latent_dim)

    def _sample_latent(self, h_enc: torch.Tensor) -> torch.Tensor:
        """从潜在空间采样"""
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.randn_like(sigma)

        self.z_mean = mu
        self.z_sigma = sigma
        self.z = self.z_mean + self.z_sigma * std_z

        return self.z

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z), self.decoder_time(z)

    def sample(self, sample_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """从潜在空间采样生成新样本"""
        new_z = torch.randn(sample_num, self.latent_dim)
        vae_pre = (self.decoder(new_z), self.decoder_time(new_z))
        X_pre = vae_pre[0].detach().numpy()
        y_pre = vae_pre[1].detach().numpy()
        y_pre = np.hstack((y_pre, np.ones_like(y_pre)))  # 添加status=1
        return X_pre, y_pre

    def condition_sample(
        self, index: np.ndarray, sample_num: int, gamma: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在指定训练样本的潜变量附近生成精确数量的新样本。"""
        if sample_num < 0:
            raise ValueError("sample_num must be non-negative")
        index = np.asarray(index, dtype=int).reshape(-1)
        if sample_num == 0:
            feature_dim = self.decoder.linear2.out_features
            return (
                np.empty((0, feature_dim), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
        if index.size == 0:
            raise ValueError("condition_sample requires at least one source index")
        if np.any(index < 0) or np.any(index >= self.z.shape[0]):
            raise IndexError("condition_sample index is outside the trained latent codes")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        source = torch.as_tensor(index, dtype=torch.long, device=self.z.device)
        chosen = source[torch.randint(source.numel(), (sample_num,), device=self.z.device)]
        target_z = self.z.detach()[chosen, :]
        new_z = target_z + gamma * torch.randn_like(target_z)
        new_X = self.decoder(new_z).detach().cpu().numpy()
        new_y = self.decoder_time(new_z).detach().cpu().numpy()
        new_y = np.hstack((new_y, np.ones_like(new_y)))  # 添加status=1
        return new_X, new_y

    def generate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成重构样本"""
        _x = torch.from_numpy(x).float()
        return self.forward(_x)


def recon_mse(dec: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """重构损失"""
    diff_sq = torch.pow(dec - X, 2)
    return torch.mean(torch.sum(diff_sq, dim=1))


def cmse(pre: torch.Tensor, time: torch.Tensor, status: torch.Tensor) -> torch.Tensor:
    """条件均方误差"""
    compare = (status == 1) | ((pre < time) & (status == 0))
    mse = torch.pow(pre - time, 2)
    return torch.mean(mse * compare.float())


def latent_loss(z_mean: torch.Tensor, z_stddev: torch.Tensor) -> torch.Tensor:
    """潜在空间损失"""
    mean_sq = torch.pow(z_mean, 2)
    var = torch.pow(z_stddev, 2)
    return 0.5 * torch.sum(torch.mean(mean_sq + var - torch.log(var) - 1, dim=0))


def vae_train(
    train_X: np.ndarray,
    train_y: np.ndarray,
    latent_num: int = 10,
    hidden_num: int = 30,
    alpha: float = 1.0,
    beta: float = 1.0,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    step_size: int = 100,
    gamma: float = 0.5,
) -> MVAE:
    """
    训练MVAE模型

    Args:
        train_X: 训练特征数据
        train_y: 训练标签数据
        latent_num: 潜在空间维度
        hidden_num: 隐藏层维度
        alpha: KL散度权重
        beta: 条件损失权重
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        step_size: 学习率调整步长
        gamma: 学习率衰减因子

    Returns:
        MVAE: 训练好的MVAE模型
    """
    input_dim = train_X.shape[1]
    encoder = Encoder(input_dim, hidden_num, hidden_num)
    decoder = Decoder(latent_num, hidden_num, input_dim)
    decoder_time = DecoderTime(latent_num, hidden_num)
    vae = MVAE(encoder, decoder, decoder_time, latent_num, hidden_num)

    criterion = nn.MSELoss()
    optimizer = Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    _X = torch.from_numpy(train_X).float()
    _y = torch.from_numpy(train_y).float().reshape(-1, 1)

    for epoch in range(epochs):
        vae.train()
        optimizer.zero_grad()

        dec, pre = vae(_X)
        kl_ll = latent_loss(vae.z_mean, vae.z_sigma)
        dec_ll = recon_mse(dec, _X)
        cmse_ll = criterion(pre, _y)

        loss = dec_ll + alpha * kl_ll + beta * cmse_ll

        loss.backward()
        optimizer.step()
        scheduler.step()

    vae.eval()
    return vae
