import * as tf from '@tensorflow/tfjs';
import { tensorToCanvas } from './canvasUtils.js';

// A grid of small canvases showing generated samples.
export function createSampleGrid(containerEl, { rows = 4, cols = 4, tileSize = 64 } = {}) {
  containerEl.innerHTML = '';
  containerEl.classList.add('sample-grid');
  containerEl.style.gridTemplateColumns = `repeat(${cols}, ${tileSize}px)`;

  const total = rows * cols;
  const canvases = [];
  for (let i = 0; i < total; i++) {
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.width = tileSize + 'px';
    canvas.style.height = tileSize + 'px';
    canvas.className = 'sample-cell';
    containerEl.appendChild(canvas);
    canvases.push(canvas);
  }

  function showNoise() {
    for (const canvas of canvases) {
      const noise = tf.randomUniform([28, 28, 1], -1, 1);
      tensorToCanvas(noise, canvas).then(() => noise.dispose());
    }
  }

  async function updateFromTensor(batchTensor) {
    const count = batchTensor.shape[0];
    const n = Math.min(count, canvases.length);
    // Split into per-image tensors.
    const imgs = tf.tidy(() => tf.unstack(batchTensor, 0)); // array of [28,28,1]
    try {
      for (let i = 0; i < n; i++) {
        await tensorToCanvas(imgs[i], canvases[i]);
      }
    } finally {
      for (const t of imgs) t.dispose();
    }
  }

  return { showNoise, updateFromTensor, canvases };
}
