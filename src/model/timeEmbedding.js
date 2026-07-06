import * as tf from '@tensorflow/tfjs';

// Standard transformer-style sinusoidal timestep embedding.
// tTensor: int32 tensor of shape [batch]. Returns [batch, dim] float32.
export function sinusoidalEmbedding(tTensor, dim = 64) {
  return tf.tidy(() => {
    const half = Math.floor(dim / 2);
    const tFloat = tTensor.toFloat().reshape([-1, 1]); // [batch, 1]

    // frequencies: exp(-log(10000) * i / (half-1)) for i in [0, half)
    const logMax = Math.log(10000);
    const freqs = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      const denom = half > 1 ? half - 1 : 1;
      freqs[i] = Math.exp(-logMax * (i / denom));
    }
    const freqTensor = tf.tensor2d(freqs, [1, half]); // [1, half]

    const args = tFloat.mul(freqTensor); // [batch, half]
    const sin = tf.sin(args);
    const cos = tf.cos(args);
    let emb = tf.concat([sin, cos], 1); // [batch, 2*half]

    // If dim is odd, pad one zero column.
    if (2 * half < dim) {
      const pad = tf.zeros([emb.shape[0], dim - 2 * half]);
      emb = tf.concat([emb, pad], 1);
    }
    return emb;
  });
}

// Shared time-conditioning MLP applied inside the functional model.
// Takes a SymbolicTensor [timeEmbedDim] input, returns SymbolicTensor [hiddenDim].
export function buildTimeMLP(timeEmbedInput, hiddenDim = 128) {
  const h1 = tf.layers
    .dense({ units: hiddenDim, activation: 'relu', name: 'time_mlp_1' })
    .apply(timeEmbedInput);
  const h2 = tf.layers
    .dense({ units: hiddenDim, activation: 'relu', name: 'time_mlp_2' })
    .apply(h1);
  return h2;
}
