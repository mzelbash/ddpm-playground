import * as tf from '@tensorflow/tfjs';
import { buildTimeMLP } from './timeEmbedding.js';

// Channel presets.
export const MODEL_PRESETS = {
  small: { label: 'Small', channels: [8, 16, 32] },
  medium: { label: 'Medium', channels: [16, 32, 64] },
  large: { label: 'Large', channels: [32, 64, 128] },
};

let uid = 0;
function n(prefix) {
  return `${prefix}_${uid++}`;
}

// Two 3x3 same-padding relu convs.
function convBlock(x, C, prefix) {
  let h = tf.layers
    .conv2d({ filters: C, kernelSize: 3, padding: 'same', activation: 'relu', name: n(prefix + '_c1') })
    .apply(x);
  h = tf.layers
    .conv2d({ filters: C, kernelSize: 3, padding: 'same', activation: 'relu', name: n(prefix + '_c2') })
    .apply(h);
  return h;
}

// Inject the shared time-MLP output into a feature map of spatial size HxW and C channels.
// Project time features to C, reshape to [1,1,C], tile up to [H,W], concat, fuse back to C.
function timeInject(featureMap, timeFeatures, C, H, W, prefix) {
  const proj = tf.layers
    .dense({ units: C, name: n(prefix + '_tproj') })
    .apply(timeFeatures); // [batch, C]
  const reshaped = tf.layers
    .reshape({ targetShape: [1, 1, C], name: n(prefix + '_treshape') })
    .apply(proj); // [batch, 1, 1, C]
  const tiled = tf.layers
    .upSampling2d({ size: [H, W], interpolation: 'nearest', name: n(prefix + '_ttile') })
    .apply(reshaped); // [batch, H, W, C]
  const merged = tf.layers
    .concatenate({ axis: -1, name: n(prefix + '_tconcat') })
    .apply([featureMap, tiled]);
  const fused = tf.layers
    .conv2d({ filters: C, kernelSize: 3, padding: 'same', activation: 'relu', name: n(prefix + '_tfuse') })
    .apply(merged);
  return fused;
}

// Build the small U-Net noise predictor.
// Two inputs: image [imageSize,imageSize,1] and timeEmbed [timeEmbedDim].
export function buildUNet({ imageSize = 28, channels = [16, 32, 64], timeEmbedDim = 64 } = {}) {
  const [c0, c1, c2] = channels;

  const imageInput = tf.input({ shape: [imageSize, imageSize, 1], name: 'image' });
  const timeEmbedInput = tf.input({ shape: [timeEmbedDim], name: 'timeEmbed' });

  // Shared time MLP.
  const timeFeatures = buildTimeMLP(timeEmbedInput, 128);

  const S0 = imageSize; // 28
  const S1 = Math.ceil(imageSize / 2); // 14
  const S2 = Math.ceil(S1 / 2); // 7

  // Stem
  let x = tf.layers
    .conv2d({ filters: c0, kernelSize: 3, padding: 'same', activation: 'relu', name: n('stem') })
    .apply(imageInput);

  // Down1 @ S0
  x = convBlock(x, c0, 'down1');
  x = timeInject(x, timeFeatures, c0, S0, S0, 'down1');
  const skip1 = x;

  // Downsample -> S1
  x = tf.layers
    .conv2d({ filters: c1, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: n('downsample1') })
    .apply(x);

  // Down2 @ S1
  x = convBlock(x, c1, 'down2');
  x = timeInject(x, timeFeatures, c1, S1, S1, 'down2');
  const skip2 = x;

  // Downsample -> S2
  x = tf.layers
    .conv2d({ filters: c2, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: n('downsample2') })
    .apply(x);

  // Bottleneck @ S2
  x = convBlock(x, c2, 'bottleneck');
  x = timeInject(x, timeFeatures, c2, S2, S2, 'bottleneck');

  // Up -> S1
  x = tf.layers
    .conv2dTranspose({ filters: c1, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: n('up_t2') })
    .apply(x);
  x = tf.layers.concatenate({ axis: -1, name: n('up2_concat') }).apply([x, skip2]);
  x = convBlock(x, c1, 'up2');
  x = timeInject(x, timeFeatures, c1, S1, S1, 'up2');

  // Up -> S0
  x = tf.layers
    .conv2dTranspose({ filters: c0, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: n('up_t1') })
    .apply(x);
  x = tf.layers.concatenate({ axis: -1, name: n('up1_concat') }).apply([x, skip1]);
  x = convBlock(x, c0, 'up1');
  x = timeInject(x, timeFeatures, c0, S0, S0, 'up1');

  // Head: predict unbounded epsilon -> no activation.
  const output = tf.layers
    .conv2d({ filters: 1, kernelSize: 3, padding: 'same', name: n('head') })
    .apply(x);

  return tf.model({ inputs: [imageInput, timeEmbedInput], outputs: output });
}
