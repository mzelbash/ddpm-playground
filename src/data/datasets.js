// Registry of available datasets. URLs are relative to the site root (Vite serves
// everything under /public at /). BASE_URL reflects the configured `base` in
// vite.config.js ('/' in dev, '/ddpm-playground/' when built for GitHub Pages),
// since Vite does not rewrite plain runtime fetch() strings on its own.
const base = import.meta.env.BASE_URL;

export const DATASETS = {
  mnist: {
    id: 'mnist',
    label: 'MNIST',
    disabled: false,
    manifestUrl: `${base}data/mnist/manifest.json`,
    imagesUrl: `${base}data/mnist/train-images.png`,
    labelsUrl: `${base}data/mnist/train-labels.bin`,
  },
  'fashion-mnist': {
    id: 'fashion-mnist',
    label: 'Fashion-MNIST',
    disabled: false,
    manifestUrl: `${base}data/fashion-mnist/manifest.json`,
    imagesUrl: `${base}data/fashion-mnist/train-images.png`,
    labelsUrl: `${base}data/fashion-mnist/train-labels.bin`,
  },
  cifar10: {
    id: 'cifar10',
    label: 'CIFAR-10 (coming soon)',
    disabled: true,
  },
};
