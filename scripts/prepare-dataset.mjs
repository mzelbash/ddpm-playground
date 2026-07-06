// Offline dataset preparation for the DDPM playground.
//
// Usage:
//   node scripts/prepare-dataset.mjs --dataset=mnist --count=6000
//   node scripts/prepare-dataset.mjs --dataset=fashion-mnist --count=6000
//
// Produces public/data/<dataset>/{manifest.json, train-images.png, train-labels.bin}
// The output sprite is a square-ish grid of 28x28 grayscale tiles laid out row-major,
// re-encoded from the source datasets into our own uniform format so the browser only
// ever has to understand one thing.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');

const IMAGE_SIZE = 28;
const NUM_CLASSES = 10;
const PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE; // 784

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {};
  for (const raw of argv.slice(2)) {
    const m = /^--([^=]+)=(.*)$/.exec(raw);
    if (m) args[m[1]] = m[2];
    else if (raw.startsWith('--')) args[raw.slice(2)] = true;
  }
  return args;
}

async function fetchBuffer(url) {
  console.log(`  fetching ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  const ab = await res.arrayBuffer();
  return Buffer.from(ab);
}

// ---------------------------------------------------------------------------
// Source decoders -> { images: Uint8Array (N*784), labels: Uint8Array (N) }
// ---------------------------------------------------------------------------

// MNIST: single sprite PNG (learnjs). The learnjs sprite is 784px wide and
// 65000px tall, one flattened 28x28 image per row. We also defensively handle a
// vertical 28-wide tile-stack layout in case the source ever changes.
async function decodeMnist() {
  const SPRITE_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
  const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

  const [spriteBuf, labelBuf] = await Promise.all([
    fetchBuffer(SPRITE_URL),
    fetchBuffer(LABELS_URL),
  ]);

  const png = PNG.sync.read(spriteBuf);
  const { width, height, data } = png; // data is RGBA
  console.log(`  MNIST sprite decoded: ${width}x${height}`);

  let count;
  const extract = (i) => {
    // returns Uint8Array(784) for image i
    const out = new Uint8Array(PIXELS_PER_IMAGE);
    if (width === PIXELS_PER_IMAGE) {
      // one image per row
      const rowStart = i * width * 4;
      for (let p = 0; p < PIXELS_PER_IMAGE; p++) {
        out[p] = data[rowStart + p * 4]; // red channel
      }
    } else if (width === IMAGE_SIZE) {
      // vertical stack of 28x28 tiles
      const tileTop = i * IMAGE_SIZE;
      for (let r = 0; r < IMAGE_SIZE; r++) {
        for (let c = 0; c < IMAGE_SIZE; c++) {
          const srcIdx = ((tileTop + r) * width + c) * 4;
          out[r * IMAGE_SIZE + c] = data[srcIdx];
        }
      }
    } else {
      throw new Error(`Unexpected MNIST sprite width ${width}`);
    }
    return out;
  };

  if (width === PIXELS_PER_IMAGE) count = height;
  else if (width === IMAGE_SIZE) count = Math.floor(height / IMAGE_SIZE);
  else throw new Error(`Unexpected MNIST sprite width ${width}`);

  // Labels are one-hot, 10 uint8 bytes per label.
  const labelCount = Math.floor(labelBuf.length / NUM_CLASSES);
  count = Math.min(count, labelCount);
  console.log(`  MNIST usable images: ${count}`);

  const images = new Uint8Array(count * PIXELS_PER_IMAGE);
  const labels = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    images.set(extract(i), i * PIXELS_PER_IMAGE);
    // argmax over the 10-byte one-hot chunk
    let best = 0;
    let bestVal = -1;
    for (let k = 0; k < NUM_CLASSES; k++) {
      const v = labelBuf[i * NUM_CLASSES + k];
      if (v > bestVal) { bestVal = v; best = k; }
    }
    labels[i] = best;
  }
  return { images, labels, count };
}

// Fashion-MNIST: official Zalando IDX gzip files.
async function decodeFashionMnist() {
  const BASE = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/';
  const IMAGES_URL = BASE + 'train-images-idx3-ubyte.gz';
  const LABELS_URL = BASE + 'train-labels-idx1-ubyte.gz';

  const [imgGz, lblGz] = await Promise.all([
    fetchBuffer(IMAGES_URL),
    fetchBuffer(LABELS_URL),
  ]);

  const imgBuf = zlib.gunzipSync(imgGz);
  const lblBuf = zlib.gunzipSync(lblGz);

  // idx3: magic(4) count(4) rows(4) cols(4) then data
  const imgMagic = imgBuf.readUInt32BE(0);
  if (imgMagic !== 0x00000803) throw new Error(`Bad idx3 magic ${imgMagic.toString(16)}`);
  const imgCount = imgBuf.readUInt32BE(4);
  const rows = imgBuf.readUInt32BE(8);
  const cols = imgBuf.readUInt32BE(12);
  if (rows !== IMAGE_SIZE || cols !== IMAGE_SIZE) {
    throw new Error(`Unexpected Fashion-MNIST dims ${rows}x${cols}`);
  }
  const imgData = imgBuf.subarray(16);

  // idx1: magic(4) count(4) then labels
  const lblMagic = lblBuf.readUInt32BE(0);
  if (lblMagic !== 0x00000801) throw new Error(`Bad idx1 magic ${lblMagic.toString(16)}`);
  const lblCount = lblBuf.readUInt32BE(4);
  const lblData = lblBuf.subarray(8);

  const count = Math.min(imgCount, lblCount);
  console.log(`  Fashion-MNIST usable images: ${count}`);

  const images = new Uint8Array(imgData.subarray(0, count * PIXELS_PER_IMAGE));
  const labels = new Uint8Array(lblData.subarray(0, count));
  return { images, labels, count };
}

// ---------------------------------------------------------------------------
// Class-balanced deterministic subset
// ---------------------------------------------------------------------------
function balancedSubset({ images, labels, count }, target) {
  const perClass = Math.floor(target / NUM_CLASSES);
  const buckets = Array.from({ length: NUM_CLASSES }, () => []);
  for (let i = 0; i < count; i++) {
    const c = labels[i];
    if (c < NUM_CLASSES && buckets[c].length < perClass) buckets[c].push(i);
  }

  // Interleave classes so the grid is visually mixed and deterministic.
  const chosen = [];
  for (let k = 0; k < perClass; k++) {
    for (let c = 0; c < NUM_CLASSES; c++) {
      if (k < buckets[c].length) chosen.push(buckets[c][k]);
    }
  }

  const finalCount = chosen.length;
  const outImages = new Uint8Array(finalCount * PIXELS_PER_IMAGE);
  const outLabels = new Uint8Array(finalCount);
  for (let j = 0; j < finalCount; j++) {
    const srcIdx = chosen[j];
    outImages.set(images.subarray(srcIdx * PIXELS_PER_IMAGE, (srcIdx + 1) * PIXELS_PER_IMAGE), j * PIXELS_PER_IMAGE);
    outLabels[j] = labels[srcIdx];
  }
  console.log(`  balanced subset: ${finalCount} images (${perClass}/class)`);
  return { images: outImages, labels: outLabels, count: finalCount };
}

// ---------------------------------------------------------------------------
// Encode as our own square-grid sprite PNG (RGBA, grayscale replicated)
// ---------------------------------------------------------------------------
function encodeSprite({ images, count }) {
  const tilesPerRow = Math.ceil(Math.sqrt(count));
  const gridRows = Math.ceil(count / tilesPerRow);
  const width = tilesPerRow * IMAGE_SIZE;
  const height = gridRows * IMAGE_SIZE;

  const png = new PNG({ width, height, colorType: 6 });
  png.data.fill(0);

  for (let i = 0; i < count; i++) {
    const tileRow = Math.floor(i / tilesPerRow);
    const tileCol = i % tilesPerRow;
    const x0 = tileCol * IMAGE_SIZE;
    const y0 = tileRow * IMAGE_SIZE;
    for (let r = 0; r < IMAGE_SIZE; r++) {
      for (let c = 0; c < IMAGE_SIZE; c++) {
        const v = images[i * PIXELS_PER_IMAGE + r * IMAGE_SIZE + c];
        const dstIdx = ((y0 + r) * width + (x0 + c)) * 4;
        png.data[dstIdx] = v;
        png.data[dstIdx + 1] = v;
        png.data[dstIdx + 2] = v;
        png.data[dstIdx + 3] = 255;
      }
    }
  }

  const buffer = PNG.sync.write(png);
  return { buffer, tilesPerRow, width, height };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const args = parseArgs(process.argv);
  const dataset = args.dataset;
  const count = parseInt(args.count ?? '6000', 10);

  if (!dataset) {
    console.error('Missing --dataset=mnist|fashion-mnist');
    process.exit(1);
  }

  console.log(`Preparing dataset "${dataset}" with target count ${count}`);

  let decoded;
  if (dataset === 'mnist') decoded = await decodeMnist();
  else if (dataset === 'fashion-mnist') decoded = await decodeFashionMnist();
  else {
    console.error(`Unknown dataset "${dataset}"`);
    process.exit(1);
  }

  const subset = balancedSubset(decoded, count);
  const { buffer, tilesPerRow } = encodeSprite(subset);

  const outDir = path.join(PROJECT_ROOT, 'public', 'data', dataset);
  fs.mkdirSync(outDir, { recursive: true });

  const spritePath = path.join(outDir, 'train-images.png');
  const labelsPath = path.join(outDir, 'train-labels.bin');
  const manifestPath = path.join(outDir, 'manifest.json');

  fs.writeFileSync(spritePath, buffer);
  fs.writeFileSync(labelsPath, Buffer.from(subset.labels));
  const manifest = {
    count: subset.count,
    imageSize: IMAGE_SIZE,
    channels: 1,
    tilesPerRow,
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

  console.log(`Wrote:`);
  console.log(`  ${spritePath} (${buffer.length} bytes)`);
  console.log(`  ${labelsPath} (${subset.labels.length} bytes)`);
  console.log(`  ${manifestPath} -> ${JSON.stringify(manifest)}`);
  console.log('Done.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
