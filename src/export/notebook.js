import { MODEL_PRESETS } from '../model/unet.js';

// Builds a Jupyter notebook (.ipynb) containing a PyTorch port of this playground's
// exact DDPM pipeline (schedule, U-Net, training loop, DDPM/DDIM sampling),
// parameterized with the current UI settings, so it can be run with a real GPU
// in Colab/Jupyter instead of the browser.

function markdownCell(text) {
  return { cell_type: 'markdown', metadata: {}, source: text.split('\n').map((l, i, a) => (i < a.length - 1 ? l + '\n' : l)) };
}

function codeCell(text) {
  return {
    cell_type: 'code',
    metadata: {},
    execution_count: null,
    outputs: [],
    source: text.split('\n').map((l, i, a) => (i < a.length - 1 ? l + '\n' : l)),
  };
}

function pyBool(v) {
  return v ? 'True' : 'False';
}

// Builds the ipynb JSON object for the given playground configuration.
export function buildNotebook(config) {
  const {
    datasetId = 'mnist',
    T = 300,
    scheduleType = 'linear',
    betaStart = 1e-4,
    betaEnd = 0.02,
    modelSize = 'medium',
    learningRate = 2e-4,
    batchSize = 64,
    totalSteps = 1500,
    samplingMethod = 'ddim',
    ddimSteps = 50,
  } = config;

  const channels = (MODEL_PRESETS[modelSize] || MODEL_PRESETS.medium).channels;
  const torchDataset = datasetId === 'fashion-mnist' ? 'FashionMNIST' : 'MNIST';

  const cells = [];

  cells.push(markdownCell(
    `# DDPM Playground: exported PyTorch notebook\n\n` +
    `Generated from the browser DDPM Playground with the settings you had selected there. ` +
    `This is a PyTorch port of the exact same pipeline (noise schedule, U-Net, training loop, and DDPM/DDIM sampling), ` +
    `so you can train it here with a real GPU (e.g. in Google Colab: **Runtime > Change runtime type > GPU**), ` +
    `which will be much faster than the browser.\n\n` +
    `**Exported settings:** dataset=\`${datasetId}\`, T=\`${T}\`, schedule=\`${scheduleType}\`, ` +
    `model size=\`${modelSize}\` (channels ${JSON.stringify(channels)}), learning rate=\`${learningRate}\`, ` +
    `batch size=\`${batchSize}\`, training steps=\`${totalSteps}\`, sampling=\`${samplingMethod}\`` +
    (samplingMethod === 'ddim' ? ` (${ddimSteps} steps)` : '') + `.`
  ));

  cells.push(markdownCell('## 1. Install requirements\n\nAlready installed in Colab; safe to run anywhere else too.'));
  cells.push(codeCell('!pip install torch torchvision matplotlib numpy --quiet'));

  cells.push(codeCell(
`import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset`
  ));

  cells.push(markdownCell('## 2. Configuration\n\nThese match the settings selected in the DDPM Playground when this notebook was generated.'));
  cells.push(codeCell(
`DATASET = ${JSON.stringify(datasetId)}
IMAGE_SIZE = 28
SUBSET_SIZE = 6000
T = ${T}
SCHEDULE_TYPE = ${JSON.stringify(scheduleType)}
BETA_START = ${betaStart}
BETA_END = ${betaEnd}
MODEL_CHANNELS = ${JSON.stringify(channels)}
TIME_EMBED_DIM = 64
LEARNING_RATE = ${learningRate}
BATCH_SIZE = ${batchSize}
TOTAL_STEPS = ${totalSteps}
SAMPLING_METHOD = ${JSON.stringify(samplingMethod)}
DDIM_STEPS = ${ddimSteps}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)`
  ));

  cells.push(markdownCell('## 3. Dataset\n\nDownloads via torchvision and takes the same size subset used in the playground.'));
  cells.push(codeCell(
`transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1], same range the model was designed for
])

dataset_cls = datasets.FashionMNIST if DATASET == "fashion-mnist" else datasets.MNIST
full_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)

subset_idx = torch.randperm(len(full_dataset))[:SUBSET_SIZE]
train_dataset = Subset(full_dataset, subset_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"Training on {len(train_dataset)} images from {DATASET}")`
  ));

  cells.push(markdownCell('## 4. Noise schedule\n\nSame linear/cosine schedule math as the playground.'));
  cells.push(codeCell(
`def make_schedule(T, schedule_type="linear", beta_start=1e-4, beta_end=0.02):
    if schedule_type == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bars_full = f / f[0]
        betas = 1 - (alpha_bars_full[1:] / alpha_bars_full[:-1])
        betas = torch.clamp(betas, 0, 0.999).float()
    else:
        betas = torch.linspace(beta_start, beta_end, T)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])
    denom = 1 - alpha_bars
    posterior_variance = betas * (1 - alpha_bars_prev) / denom.clamp(min=1e-8)
    posterior_variance[0] = 0.0

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt((1 - alpha_bars).clamp(min=0)),
        "posterior_variance": posterior_variance,
    }

schedule = make_schedule(T, SCHEDULE_TYPE, BETA_START, BETA_END)
schedule = {k: v.to(DEVICE) for k, v in schedule.items()}`
  ));

  cells.push(markdownCell('## 5. Sinusoidal timestep embedding'));
  cells.push(codeCell(
`def sinusoidal_embedding(t, dim=TIME_EMBED_DIM):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)`
  ));

  cells.push(markdownCell(
    '## 6. U-Net noise predictor\n\n' +
    'Same topology as the playground: stem, 2 down blocks with skip connections, a bottleneck, ' +
    '2 up blocks, and a time embedding injected at every block (project, tile to spatial size, ' +
    'concatenate with the feature map, fuse back down with a conv). No normalization layers ' +
    'and no activation on the output head, matching the playground model.'
  ));
  cells.push(codeCell(
`class TimeMLP(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, t_emb):
        return self.net(t_emb)


class TimeInject(nn.Module):
    def __init__(self, time_dim, channels):
        super().__init__()
        self.proj = nn.Linear(time_dim, channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x, t_feat):
        b, c, h, w = x.shape
        t_map = self.proj(t_feat)[:, :, None, None].expand(-1, -1, h, w)
        return F.relu(self.fuse(torch.cat([x, t_map], dim=1)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, channels=(16, 32, 64), time_embed_dim=TIME_EMBED_DIM, time_hidden=128):
        super().__init__()
        c0, c1, c2 = channels
        self.time_mlp = TimeMLP(time_embed_dim, time_hidden)

        self.stem = nn.Conv2d(1, c0, 3, padding=1)

        self.down1 = ConvBlock(c0, c0)
        self.inject1 = TimeInject(time_hidden, c0)
        self.downsample1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)

        self.down2 = ConvBlock(c1, c1)
        self.inject2 = TimeInject(time_hidden, c1)
        self.downsample2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        self.bottleneck = ConvBlock(c2, c2)
        self.inject_mid = TimeInject(time_hidden, c2)

        self.up_t2 = nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1)
        self.up2 = ConvBlock(c1 * 2, c1)
        self.inject3 = TimeInject(time_hidden, c1)

        self.up_t1 = nn.ConvTranspose2d(c1, c0, 3, stride=2, padding=1, output_padding=1)
        self.up1 = ConvBlock(c0 * 2, c0)
        self.inject4 = TimeInject(time_hidden, c0)

        self.head = nn.Conv2d(c0, 1, 3, padding=1)  # no activation: predicts unbounded epsilon

    def forward(self, x, t_emb):
        t_feat = self.time_mlp(t_emb)

        x0 = F.relu(self.stem(x))
        d1 = self.inject1(self.down1(x0), t_feat)                 # skip1, at full resolution

        x1 = F.relu(self.downsample1(d1))
        d2 = self.inject2(self.down2(x1), t_feat)                 # skip2, at half resolution

        x2 = F.relu(self.downsample2(d2))
        bn = self.inject_mid(self.bottleneck(x2), t_feat)

        u1 = F.relu(self.up_t2(bn))
        u1 = self.inject3(self.up2(torch.cat([u1, d2], dim=1)), t_feat)

        u2 = F.relu(self.up_t1(u1))
        u2 = self.inject4(self.up1(torch.cat([u2, d1], dim=1)), t_feat)

        return self.head(u2)


model = UNet(channels=MODEL_CHANNELS).to(DEVICE)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")`
  ));

  cells.push(markdownCell('## 7. Training loop\n\nSame simplified DDPM loss (MSE between predicted and true noise) as the playground.'));
  cells.push(codeCell(
`optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def make_noisy_batch(x0, t, schedule):
    sqrt_ab = schedule["sqrt_alpha_bars"][t].view(-1, 1, 1, 1)
    sqrt_omab = schedule["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = sqrt_ab * x0 + sqrt_omab * eps
    return xt, eps


losses = []
step = 0
data_iter = iter(train_loader)
model.train()

while step < TOTAL_STEPS:
    try:
        x0, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x0, _ = next(data_iter)

    x0 = x0.to(DEVICE)
    t = torch.randint(0, T, (x0.shape[0],), device=DEVICE)
    xt, eps = make_noisy_batch(x0, t, schedule)
    t_emb = sinusoidal_embedding(t)

    eps_pred = model(xt, t_emb)
    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if step % 50 == 0 or step == TOTAL_STEPS - 1:
        print(f"step {step + 1}/{TOTAL_STEPS}  loss {loss.item():.4f}")
    step += 1

plt.figure(figsize=(6, 3))
plt.plot(losses)
plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss")
plt.show()`
  ));

  cells.push(markdownCell('## 8. Sampling: DDPM (full ancestral) and DDIM (fast)\n\nSame formulas as the playground\'s sampler, sharing the trained model and schedule.'));
  cells.push(codeCell(
`@torch.no_grad()
def sample_ddpm(model, schedule, num_samples=16, image_size=IMAGE_SIZE):
    model.eval()
    x = torch.randn(num_samples, 1, image_size, image_size, device=DEVICE)

    for t in reversed(range(T)):
        beta = schedule["betas"][t]
        alpha = schedule["alphas"][t]
        alpha_bar = schedule["alpha_bars"][t]
        post_var = schedule["posterior_variance"][t]

        t_batch = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_batch)
        eps_pred = model(x, t_emb)

        mean = (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_pred) / torch.sqrt(alpha)
        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(post_var) * z
        else:
            x = mean
    return x


@torch.no_grad()
def sample_ddim(model, schedule, num_samples=16, image_size=IMAGE_SIZE, ddim_steps=DDIM_STEPS, eta=0.0):
    model.eval()
    seq = sorted(set(int(i * T / ddim_steps) for i in range(ddim_steps)))
    x = torch.randn(num_samples, 1, image_size, image_size, device=DEVICE)

    for i in reversed(range(len(seq))):
        t = seq[i]
        t_prev = seq[i - 1] if i > 0 else -1

        alpha_bar = schedule["alpha_bars"][t]
        alpha_bar_prev = schedule["alpha_bars"][t_prev] if t_prev >= 0 else torch.tensor(1.0, device=DEVICE)

        t_batch = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_batch)
        eps_pred = model(x, t_emb)

        x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        sigma = torch.tensor(0.0, device=DEVICE)
        if eta > 0 and t_prev >= 0:
            term1 = (1 - alpha_bar_prev) / (1 - alpha_bar).clamp(min=1e-8)
            term2 = 1 - alpha_bar / alpha_bar_prev.clamp(min=1e-8)
            sigma = eta * torch.sqrt(term1.clamp(min=0) * term2.clamp(min=0))

        dir_coef = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0))
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_coef * eps_pred
        if sigma.item() > 0:
            x = x + sigma * torch.randn_like(x)
    return x`
  ));

  cells.push(markdownCell('## 9. Generate and view samples'));
  cells.push(codeCell(
`if SAMPLING_METHOD == "ddim":
    samples = sample_ddim(model, schedule, num_samples=16, ddim_steps=DDIM_STEPS)
else:
    samples = sample_ddpm(model, schedule, num_samples=16)

samples = samples.detach().cpu().clamp(-1, 1)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img = (samples[i, 0] + 1) / 2  # back to [0, 1] for display
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.show()`
  ));

  return {
    cells,
    metadata: {
      kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
      language_info: { name: 'python', pygments_lexer: 'ipython3' },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

// Triggers a browser download of the generated notebook as a .ipynb file.
export function downloadNotebook(config, filename = 'ddpm_playground.ipynb') {
  const notebook = buildNotebook(config);
  const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/x-ipynb+json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
