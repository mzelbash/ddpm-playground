import { defineConfig } from 'vite';

// Base must match the GitHub Pages project-site path (https://<user>.github.io/<repo>/)
// so built asset URLs resolve correctly when served from a subpath instead of the root.
export default defineConfig({
  base: '/ddpm-playground/',
});
