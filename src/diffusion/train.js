import * as tf from '@tensorflow/tfjs';
import { sinusoidalEmbedding } from '../model/timeEmbedding.js';

// Creates a training controller for a DDPM noise-prediction model.
// config = { learningRate, batchSize, totalSteps, sampleEveryNSteps, timeEmbedDim }
export function createTrainer({ model, schedule, images, config = {} }) {
  const {
    learningRate = 2e-4,
    batchSize = 64,
    totalSteps = 1500,
    sampleEveryNSteps = 200,
    timeEmbedDim = 64,
  } = config;

  const T = schedule.T;
  const numImages = images.shape[0];

  const optimizer = tf.train.adam(learningRate);

  // Small resident tensors for the per-step scalar gather.
  const sqrtAB = tf.tensor1d(schedule.sqrtAlphaBars);
  const sqrtOMAB = tf.tensor1d(schedule.sqrtOneMinusAlphaBars);

  let running = false;
  let paused = false;
  let disposed = false;
  let stepIndex = 0;
  let resumeResolvers = [];

  const stepCbs = [];
  const sampleCbs = [];

  function onStep(cb) { stepCbs.push(cb); }
  function onSampleCheckpoint(cb) { sampleCbs.push(cb); }

  function waitWhilePaused() {
    if (!paused) return Promise.resolve();
    return new Promise((resolve) => resumeResolvers.push(resolve));
  }

  async function trainStep() {
    // Random batch indices.
    const idx = new Int32Array(batchSize);
    for (let i = 0; i < batchSize; i++) idx[i] = Math.floor(Math.random() * numImages);
    const idxTensor = tf.tensor1d(idx, 'int32');

    // Random timesteps.
    const tArr = new Int32Array(batchSize);
    for (let i = 0; i < batchSize; i++) tArr[i] = Math.floor(Math.random() * T);
    const tTensor = tf.tensor1d(tArr, 'int32');

    const x0 = tf.gather(images, idxTensor); // [b,28,28,1]
    const epsilon = tf.randomNormal(x0.shape);

    // xt = sqrtAB[t]*x0 + sqrtOMAB[t]*epsilon
    const xt = tf.tidy(() => {
      const sa = tf.gather(sqrtAB, tTensor).reshape([batchSize, 1, 1, 1]);
      const soma = tf.gather(sqrtOMAB, tTensor).reshape([batchSize, 1, 1, 1]);
      return x0.mul(sa).add(epsilon.mul(soma));
    });

    const timeEmbed = sinusoidalEmbedding(tTensor, timeEmbedDim);

    const lossTensor = optimizer.minimize(() => {
      const epsPred = model.apply([xt, timeEmbed], { training: true });
      return tf.mean(tf.square(epsPred.sub(epsilon)));
    }, true);

    const lossVal = (await lossTensor.data())[0];

    // Cleanup tensors that live outside tidy blocks.
    lossTensor.dispose();
    idxTensor.dispose();
    tTensor.dispose();
    x0.dispose();
    epsilon.dispose();
    xt.dispose();
    timeEmbed.dispose();

    return lossVal;
  }

  async function loop() {
    while (stepIndex < totalSteps && running) {
      await waitWhilePaused();
      if (!running) break;

      const loss = await trainStep();
      const currentStep = stepIndex;
      stepIndex++;

      for (const cb of stepCbs) cb(currentStep, loss, totalSteps);

      if ((stepIndex % sampleEveryNSteps === 0) || stepIndex === totalSteps) {
        for (const cb of sampleCbs) cb(stepIndex);
      }

      await tf.nextFrame();
    }
    if (stepIndex >= totalSteps) running = false;
  }

  return {
    onStep,
    onSampleCheckpoint,
    start() {
      if (disposed || running) return;
      running = true;
      paused = false;
      loop();
    },
    pause() {
      if (!running) return;
      paused = true;
    },
    resume() {
      if (!running || !paused) return;
      paused = false;
      const resolvers = resumeResolvers;
      resumeResolvers = [];
      for (const r of resolvers) r();
    },
    reset() {
      running = false;
      paused = false;
      const resolvers = resumeResolvers;
      resumeResolvers = [];
      for (const r of resolvers) r();
      stepIndex = 0;
    },
    dispose() {
      this.reset();
      disposed = true;
      optimizer.dispose();
      sqrtAB.dispose();
      sqrtOMAB.dispose();
    },
    isRunning() { return running; },
    isPaused() { return paused; },
    getStep() { return stepIndex; },
    get totalSteps() { return totalSteps; },
  };
}
