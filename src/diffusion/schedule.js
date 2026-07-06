// Noise schedule as plain JS typed arrays (indexed synchronously thousands of times
// per training run, so we deliberately keep these out of tf tensors).
export function makeSchedule({
  T = 300,
  scheduleType = 'linear',
  betaStart = 1e-4,
  betaEnd = 0.02,
} = {}) {
  const betas = new Float32Array(T);

  if (scheduleType === 'cosine') {
    // Nichol & Dhariwal cosine schedule.
    const s = 0.008;
    const f = (t) => {
      const x = ((t / T + s) / (1 + s)) * (Math.PI / 2);
      const c = Math.cos(x);
      return c * c;
    };
    const f0 = f(0);
    const alphaBar = new Float32Array(T + 1);
    for (let t = 0; t <= T; t++) alphaBar[t] = f(t) / f0;
    for (let t = 0; t < T; t++) {
      const beta = 1 - alphaBar[t + 1] / alphaBar[t];
      betas[t] = Math.min(Math.max(beta, 0), 0.999);
    }
  } else {
    // linear
    for (let t = 0; t < T; t++) {
      betas[t] = T === 1 ? betaStart : betaStart + (betaEnd - betaStart) * (t / (T - 1));
    }
  }

  const alphas = new Float32Array(T);
  const alphaBars = new Float32Array(T);
  const alphaBarsPrev = new Float32Array(T);
  const sqrtAlphaBars = new Float32Array(T);
  const sqrtOneMinusAlphaBars = new Float32Array(T);
  const posteriorVariance = new Float32Array(T);

  let cumProd = 1;
  for (let t = 0; t < T; t++) {
    alphas[t] = 1 - betas[t];
    const prev = cumProd; // alphaBar_{t-1}, with alphaBar_{-1} = 1
    cumProd *= alphas[t];
    alphaBars[t] = cumProd;
    alphaBarsPrev[t] = prev;
    sqrtAlphaBars[t] = Math.sqrt(cumProd);
    sqrtOneMinusAlphaBars[t] = Math.sqrt(Math.max(0, 1 - cumProd));
  }

  for (let t = 0; t < T; t++) {
    // beta_t * (1 - alphaBar_{t-1}) / (1 - alphaBar_t)
    const denom = 1 - alphaBars[t];
    if (t === 0 || denom <= 1e-8) {
      posteriorVariance[t] = 0;
    } else {
      posteriorVariance[t] = (betas[t] * (1 - alphaBarsPrev[t])) / denom;
    }
  }

  return {
    T,
    scheduleType,
    betaStart,
    betaEnd,
    betas,
    alphas,
    alphaBars,
    alphaBarsPrev,
    sqrtAlphaBars,
    sqrtOneMinusAlphaBars,
    posteriorVariance,
  };
}
