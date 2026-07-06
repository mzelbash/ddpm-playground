# DDPM Playground

A hands-on, in-browser playground for teaching Denoising Diffusion Probabilistic Models (DDPM), built with [TensorFlow.js](https://www.tensorflow.org/js). Train a small diffusion model entirely client-side, tweak the noise schedule and model hyperparameters, and watch both the forward noising process and reverse sampling (DDPM/DDIM) happen live in the browser.

**Live demo: https://mzelbash.github.io/ddpm-playground/**

![DDPM Playground screenshot](docs/screenshot.png)

## What it does

- Trains a tiny U-Net noise-prediction model on MNIST or Fashion-MNIST, entirely in the browser (no backend, no GPU server required).
- Visualizes the forward noising process (a training image progressively corrupted per the noise schedule) and live training loss.
- Generates new images by running the trained model in reverse, with a choice of DDPM (full ancestral) or DDIM (fast) sampling.
- Exports the current settings as a runnable Jupyter notebook (PyTorch), so the same setup can be trained with a real GPU in Colab or a local Jupyter install.

## Getting started locally

```bash
npm install
npm run dev
```

Then open the printed local URL in your browser.

### Regenerating dataset assets

The MNIST / Fashion-MNIST sprite sheets under `public/data/` are pre-built and already checked into this repo. To regenerate them from scratch:

```bash
node scripts/prepare-dataset.mjs --dataset=mnist --count=6000
node scripts/prepare-dataset.mjs --dataset=fashion-mnist --count=6000
```

## Deployment

Pushes to `main` automatically build and deploy to GitHub Pages via `.github/workflows/deploy.yml`.
