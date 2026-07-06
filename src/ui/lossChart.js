// Hand-rolled canvas line chart for training loss. No charting dependency.
export function createLossChart(canvasEl) {
  const maxPoints = 500;
  let points = []; // {step, loss}

  const dpr = window.devicePixelRatio || 1;

  function resize() {
    const rect = canvasEl.getBoundingClientRect();
    const w = Math.max(1, rect.width || canvasEl.width || 400);
    const h = Math.max(1, rect.height || canvasEl.height || 160);
    canvasEl.width = w * dpr;
    canvasEl.height = h * dpr;
  }

  function themeColor(varName, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    return v || fallback;
  }

  function draw() {
    resize();
    const ctx = canvasEl.getContext('2d');
    const W = canvasEl.width;
    const H = canvasEl.height;
    ctx.clearRect(0, 0, W, H);

    const colorPanel2 = themeColor('--panel-2', '#1a1a1a');
    const colorBorder = themeColor('--border', '#444');
    const colorMuted = themeColor('--muted', '#aaa');
    const colorAccent = themeColor('--accent', '#4ea1ff');

    // Background.
    ctx.fillStyle = colorPanel2;
    ctx.fillRect(0, 0, W, H);

    const pad = 32 * dpr;
    const plotW = W - pad * 1.5;
    const plotH = H - pad * 1.5;
    const x0 = pad;
    const y0 = pad * 0.5;

    if (points.length === 0) {
      ctx.fillStyle = colorMuted;
      ctx.font = `${12 * dpr}px sans-serif`;
      ctx.fillText('Loss will appear here once training starts.', x0, H / 2);
      return;
    }

    let minL = Infinity;
    let maxL = -Infinity;
    for (const p of points) {
      if (p.loss < minL) minL = p.loss;
      if (p.loss > maxL) maxL = p.loss;
    }
    if (minL === maxL) { minL -= 0.001; maxL += 0.001; }
    const range = maxL - minL;

    const firstStep = points[0].step;
    const lastStep = points[points.length - 1].step;
    const stepRange = Math.max(1, lastStep - firstStep);

    // Axes.
    ctx.strokeStyle = colorBorder;
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0, y0 + plotH);
    ctx.lineTo(x0 + plotW, y0 + plotH);
    ctx.stroke();

    // Y labels (min/max).
    ctx.fillStyle = colorMuted;
    ctx.font = `${11 * dpr}px sans-serif`;
    ctx.fillText(maxL.toFixed(4), 2 * dpr, y0 + 10 * dpr);
    ctx.fillText(minL.toFixed(4), 2 * dpr, y0 + plotH);

    // Line.
    ctx.strokeStyle = colorAccent;
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const px = x0 + ((p.step - firstStep) / stepRange) * plotW;
      const py = y0 + plotH - ((p.loss - minL) / range) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Latest value label.
    const last = points[points.length - 1];
    ctx.fillStyle = colorAccent;
    ctx.fillText(`step ${last.step}  loss ${last.loss.toFixed(4)}`, x0 + 4 * dpr, y0 + 14 * dpr);
  }

  draw();
  window.addEventListener('app:theme-changed', draw);

  return {
    push(step, loss) {
      if (!Number.isFinite(loss)) return;
      points.push({ step, loss });
      if (points.length > maxPoints) points = points.slice(points.length - maxPoints);
      draw();
    },
    reset() {
      points = [];
      draw();
    },
  };
}
