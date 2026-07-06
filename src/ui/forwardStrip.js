import * as tf from '@tensorflow/tfjs';
import { tensorToCanvas } from './canvasUtils.js';

// Render the forward noising process for a single image across evenly-spaced timesteps.
// image: tf.Tensor of shape [28,28,1] in [-1,1].
export function renderForwardStrip(containerEl, { image, schedule, numFrames = 10, tileSize = 56 }) {
  containerEl.innerHTML = '';
  const T = schedule.T;
  const imageSize = image.shape[0];

  // Deterministic fixed noise for this render.
  const epsilon = tf.randomNormal([imageSize, imageSize, 1]);

  const timesteps = [];
  for (let k = 0; k < numFrames; k++) {
    const t = numFrames === 1 ? 0 : Math.round((k * (T - 1)) / (numFrames - 1));
    timesteps.push(t);
  }

  const renderPromises = [];
  for (const t of timesteps) {
    const frame = document.createElement('div');
    frame.className = 'forward-frame';

    const canvas = document.createElement('canvas');
    canvas.width = imageSize;
    canvas.height = imageSize;
    canvas.style.width = tileSize + 'px';
    canvas.style.height = tileSize + 'px';
    frame.appendChild(canvas);

    const label = document.createElement('div');
    label.className = 'forward-label';
    label.textContent = `t=${t}`;
    frame.appendChild(label);

    containerEl.appendChild(frame);

    const sa = schedule.sqrtAlphaBars[t];
    const soma = schedule.sqrtOneMinusAlphaBars[t];
    const xt = tf.tidy(() => image.mul(sa).add(epsilon.mul(soma)));
    renderPromises.push(tensorToCanvas(xt, canvas).then(() => xt.dispose()));
  }

  Promise.all(renderPromises).then(() => epsilon.dispose());
}
