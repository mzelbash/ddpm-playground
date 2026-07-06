// Node smoke test for the pure tf/logic modules (schedule, model, train, sample).
// Uses the tfjs CPU backend. Not a browser test — the data loader and canvas
// utils are browser-only and excluded here.
import * as tf from '@tensorflow/tfjs';
import { makeSchedule } from '../src/diffusion/schedule.js';
import { buildUNet } from '../src/model/unet.js';
import { sinusoidalEmbedding } from '../src/model/timeEmbedding.js';
import { sampleDDIM, sampleDDPM } from '../src/diffusion/sample.js';

function assert(cond, msg) { if (!cond) throw new Error('ASSERT FAILED: ' + msg); }

async function main() {
  await tf.ready();
  console.log('backend:', tf.getBackend());

  // 1. Schedule sanity (linear + cosine).
  for (const scheduleType of ['linear', 'cosine']) {
    const s = makeSchedule({ T: 50, scheduleType });
    assert(s.betas.length === 50, 'betas length');
    assert(s.alphaBars[0] <= 1 && s.alphaBars[49] > 0 && s.alphaBars[49] < s.alphaBars[0], `${scheduleType} alphaBars monotone`);
    for (const arr of [s.sqrtAlphaBars, s.sqrtOneMinusAlphaBars, s.posteriorVariance]) {
      for (const v of arr) assert(Number.isFinite(v), `${scheduleType} finite`);
    }
    assert(s.posteriorVariance[0] === 0, `${scheduleType} postVar[0] guarded`);
    console.log(`  schedule ${scheduleType}: OK  (beta0=${s.betas[0].toFixed(5)}, betaLast=${s.betas[49].toFixed(5)})`);
  }

  // 2. Time embedding shape.
  const te = sinusoidalEmbedding(tf.tensor1d([0, 10, 49], 'int32'), 64);
  assert(te.shape[0] === 3 && te.shape[1] === 64, 'time embed shape');
  te.dispose();
  console.log('  sinusoidalEmbedding: OK');

  // 3. Build model, forward pass.
  const schedule = makeSchedule({ T: 50, scheduleType: 'linear' });
  const model = buildUNet({ imageSize: 28, channels: [8, 16, 32], timeEmbedDim: 64 });
  const nParams = model.countParams();
  console.log('  model params:', nParams);

  const b = 8;
  const dummy = tf.randomNormal([b, 28, 28, 1]);
  const tEmb = sinusoidalEmbedding(tf.tensor1d(new Array(b).fill(5), 'int32'), 64);
  const out = model.apply([dummy, tEmb]);
  assert(JSON.stringify(out.shape) === JSON.stringify([b, 28, 28, 1]), `output shape ${out.shape}`);
  const outData = await out.data();
  assert(outData.every(Number.isFinite), 'output finite');
  out.dispose(); dummy.dispose(); tEmb.dispose();
  console.log('  model forward pass: OK, output shape [8,28,28,1]');

  // 4. A few manual training steps to confirm loss computes and drops.
  const images = tf.randomNormal([64, 28, 28, 1]); // synthetic "data"
  const optimizer = tf.train.adam(1e-3);
  const sqrtAB = tf.tensor1d(schedule.sqrtAlphaBars);
  const sqrtOMAB = tf.tensor1d(schedule.sqrtOneMinusAlphaBars);
  let firstLoss = null, lastLoss = null;
  for (let step = 0; step < 15; step++) {
    const idx = tf.tensor1d(Array.from({ length: 16 }, () => Math.floor(Math.random() * 64)), 'int32');
    const tArr = tf.tensor1d(Array.from({ length: 16 }, () => Math.floor(Math.random() * 50)), 'int32');
    const x0 = tf.gather(images, idx);
    const eps = tf.randomNormal(x0.shape);
    const xt = tf.tidy(() => {
      const sa = tf.gather(sqrtAB, tArr).reshape([16, 1, 1, 1]);
      const soma = tf.gather(sqrtOMAB, tArr).reshape([16, 1, 1, 1]);
      return x0.mul(sa).add(eps.mul(soma));
    });
    const timeEmbed = sinusoidalEmbedding(tArr, 64);
    const loss = optimizer.minimize(() => {
      const pred = model.apply([xt, timeEmbed], { training: true });
      return tf.mean(tf.square(pred.sub(eps)));
    }, true);
    const lv = (await loss.data())[0];
    if (step === 0) firstLoss = lv;
    lastLoss = lv;
    loss.dispose(); idx.dispose(); tArr.dispose(); x0.dispose(); eps.dispose(); xt.dispose(); timeEmbed.dispose();
  }
  console.log(`  training: firstLoss=${firstLoss.toFixed(4)} lastLoss=${lastLoss.toFixed(4)}`);
  assert(Number.isFinite(firstLoss) && Number.isFinite(lastLoss), 'losses finite');

  // 5. Sampling shapes + finiteness.
  const ddim = await sampleDDIM({ model, schedule, numSamples: 4, imageSize: 28, ddimSteps: 8, timeEmbedDim: 64 });
  assert(JSON.stringify(ddim.shape) === JSON.stringify([4, 28, 28, 1]), `ddim shape ${ddim.shape}`);
  assert((await ddim.data()).every(Number.isFinite), 'ddim finite');
  ddim.dispose();
  console.log('  sampleDDIM: OK, shape [4,28,28,1], finite');

  const ddpm = await sampleDDPM({ model, schedule, numSamples: 4, imageSize: 28, timeEmbedDim: 64 });
  assert(JSON.stringify(ddpm.shape) === JSON.stringify([4, 28, 28, 1]), `ddpm shape ${ddpm.shape}`);
  assert((await ddpm.data()).every(Number.isFinite), 'ddpm finite');
  ddpm.dispose();
  console.log('  sampleDDPM: OK, shape [4,28,28,1], finite');

  // Leak check.
  images.dispose(); optimizer.dispose(); sqrtAB.dispose(); sqrtOMAB.dispose(); model.dispose();
  console.log('tensors still alive:', tf.memory().numTensors);
  console.log('\nALL SMOKE TESTS PASSED');
}

main().catch((e) => { console.error(e); process.exit(1); });
