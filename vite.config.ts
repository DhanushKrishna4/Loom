import { defineConfig } from 'vite';

export default defineConfig({
  root: 'web',
  // Relative, not '/repo-name/'. GitHub Pages serves a project site from a
  // subpath, and hardcoding it would tie the build to one repository name and
  // break local preview. Every asset reference in this site is already
  // relative, so './' keeps it working anywhere it is served from.
  base: './',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'es2022',
    rollupOptions: {
      // Two entry points, not one: the benchmark is a separate page and should
      // not pull transformers.js into the main bundle.
      input: {
        main: 'web/index.html',
        bench: 'web/bench.html',
      },
    },
  },
  server: { port: 8080 },
});
