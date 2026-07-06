import * as tf from '@tensorflow/tfjs';
import { sinusoidalEmbedding } from '../model/timeEmbedding.js';

// Full ancestral DDPM sampling (Ho et al. Algorithm 2).
export async function sampleDDPM({
  model,
  schedule,
  numSamples = 16,
  imageSize = 28,
  timeEmbedDim = 64,
  onStep,
  throttle = 10,
}) {
  const T = schedule.T;
  let x = tf.randomNormal([numSamples, imageSize, imageSize, 1]);

  for (let t = T - 1; t >= 0; t--) {
    const beta = schedule.betas[t];
    const alpha = schedule.alphas[t];
    const alphaBar = schedule.alphaBars[t];
    const sqrtAlpha = Math.sqrt(alpha);
    const sqrtOneMinusAlphaBar = Math.sqrt(Math.max(1e-8, 1 - alphaBar));
    const postVar = schedule.posteriorVariance[t];
    const sqrtPostVar = Math.sqrt(Math.max(0, postVar));

    const next = tf.tidy(() => {
      const tTensor = tf.fill([numSamples], t, 'int32');
      const timeEmbed = sinusoidalEmbedding(tTensor, timeEmbedDim);
      const epsPred = model.apply([x, timeEmbed], { training: false });

      // mean = (1/sqrtAlpha) * (x - beta/sqrtOneMinusAlphaBar * epsPred)
      const mean = x
        .sub(epsPred.mul(beta / sqrtOneMinusAlphaBar))
        .mul(1 / sqrtAlpha);

      if (t > 0 && sqrtPostVar > 0) {
        const z = tf.randomNormal(x.shape);
        return mean.add(z.mul(sqrtPostVar));
      }
      return mean;
    });

    x.dispose();
    x = next;

    if (onStep && (t % throttle === 0 || t === 0)) {
      onStep(t, x);
      await tf.nextFrame();
    }
  }
  return x;
}

// Fast deterministic-by-default DDIM sampling.
export async function sampleDDIM({
  model,
  schedule,
  numSamples = 16,
  imageSize = 28,
  ddimSteps = 50,
  eta = 0,
  timeEmbedDim = 64,
  onStep,
  throttle = 5,
}) {
  const T = schedule.T;
  const steps = Array.from({ length: ddimSteps }, (_, i) => Math.floor((i * T) / ddimSteps));
  // ascending unique-ish subsequence; iterate from high t to low t.
  const seq = steps.slice().sort((a, b) => a - b);

  let x = tf.randomNormal([numSamples, imageSize, imageSize, 1]);

  for (let i = seq.length - 1; i >= 0; i--) {
    const t = seq[i];
    const tPrev = i > 0 ? seq[i - 1] : -1;

    const alphaBar = schedule.alphaBars[t];
    const alphaBarPrev = tPrev >= 0 ? schedule.alphaBars[tPrev] : 1;
    const sqrtAlphaBar = Math.sqrt(alphaBar);
    const sqrtOneMinusAlphaBar = Math.sqrt(Math.max(1e-8, 1 - alphaBar));

    // sigma = eta * sqrt((1-alphaBarPrev)/(1-alphaBar)) * sqrt(1 - alphaBar/alphaBarPrev)
    let sigma = 0;
    if (eta > 0 && tPrev >= 0) {
      const term1 = (1 - alphaBarPrev) / Math.max(1e-8, 1 - alphaBar);
      const term2 = 1 - alphaBar / Math.max(1e-8, alphaBarPrev);
      sigma = eta * Math.sqrt(Math.max(0, term1) * Math.max(0, term2));
    }
    const dirCoef = Math.sqrt(Math.max(0, 1 - alphaBarPrev - sigma * sigma));
    const sqrtAlphaBarPrev = Math.sqrt(alphaBarPrev);

    const next = tf.tidy(() => {
      const tTensor = tf.fill([numSamples], t, 'int32');
      const timeEmbed = sinusoidalEmbedding(tTensor, timeEmbedDim);
      const epsPred = model.apply([x, timeEmbed], { training: false });

      // x0_pred = (x - sqrtOneMinusAlphaBar * epsPred) / sqrtAlphaBar, clipped to [-1,1]
      const x0Pred = x
        .sub(epsPred.mul(sqrtOneMinusAlphaBar))
        .div(sqrtAlphaBar)
        .clipByValue(-1, 1);

      let out = x0Pred.mul(sqrtAlphaBarPrev).add(epsPred.mul(dirCoef));
      if (sigma > 0) {
        const z = tf.randomNormal(x.shape);
        out = out.add(z.mul(sigma));
      }
      return out;
    });

    x.dispose();
    x = next;

    if (onStep && (i % throttle === 0 || i === 0)) {
      onStep(t, x);
      await tf.nextFrame();
    }
  }
  return x;
}

// Dispatcher.
export async function sample({ method = 'ddim', ...opts }) {
  if (method === 'ddpm') return sampleDDPM(opts);
  return sampleDDIM(opts);
}
