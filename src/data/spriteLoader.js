import * as tf from '@tensorflow/tfjs';

// Loads one of our prepared sprite datasets into resident tf tensors.
//
// Returns { images: tf.Tensor4d [count,28,28,1] in [-1,1], labels: tf.Tensor1d int32 }.
export async function loadSpriteDataset({ manifestUrl, imagesUrl, labelsUrl }) {
  const manifestRes = await fetch(manifestUrl);
  if (!manifestRes.ok) throw new Error(`Failed to load manifest: ${manifestRes.status}`);
  const manifest = await manifestRes.json();
  const { count, imageSize, tilesPerRow } = manifest;

  // Load sprite image.
  const img = new Image();
  img.src = imagesUrl;
  await img.decode();

  const gridRows = Math.ceil(count / tilesPerRow);
  const canvas = document.createElement('canvas');
  canvas.width = tilesPerRow * imageSize;
  canvas.height = gridRows * imageSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height); // RGBA

  // Extract red channel per tile into a flat Float32Array [count*imageSize*imageSize].
  const pixelsPerImage = imageSize * imageSize;
  const pixels = new Float32Array(count * pixelsPerImage);
  const fullWidth = canvas.width;
  for (let i = 0; i < count; i++) {
    const tileRow = Math.floor(i / tilesPerRow);
    const tileCol = i % tilesPerRow;
    const x0 = tileCol * imageSize;
    const y0 = tileRow * imageSize;
    for (let r = 0; r < imageSize; r++) {
      for (let c = 0; c < imageSize; c++) {
        const srcIdx = ((y0 + r) * fullWidth + (x0 + c)) * 4; // red channel
        pixels[i * pixelsPerImage + r * imageSize + c] = data[srcIdx];
      }
    }
  }

  // Labels.
  const labelsRes = await fetch(labelsUrl);
  if (!labelsRes.ok) throw new Error(`Failed to load labels: ${labelsRes.status}`);
  const labelBuf = new Uint8Array(await labelsRes.arrayBuffer());
  const labelBytes = labelBuf.subarray(0, count);

  const images = tf.tidy(() => {
    const raw = tf.tensor4d(pixels, [count, imageSize, imageSize, 1]);
    return raw.div(127.5).sub(1); // -> [-1, 1]
  });
  const labels = tf.tensor1d(Array.from(labelBytes), 'int32');

  return { images, labels, manifest };
}
