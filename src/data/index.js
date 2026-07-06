import { DATASETS } from './datasets.js';
import { loadSpriteDataset } from './spriteLoader.js';

// Module-level cache so re-selecting a dataset doesn't re-decode the sprite.
const cache = new Map();

export function listDatasets() {
  return Object.values(DATASETS);
}

export async function loadDataset(id) {
  if (cache.has(id)) return cache.get(id);

  const def = DATASETS[id];
  if (!def) throw new Error(`Unknown dataset "${id}"`);
  if (def.disabled) throw new Error(`Dataset "${id}" is not available yet`);

  const loaded = await loadSpriteDataset({
    manifestUrl: def.manifestUrl,
    imagesUrl: def.imagesUrl,
    labelsUrl: def.labelsUrl,
  });
  const result = { id, ...loaded };
  cache.set(id, result);
  return result;
}
