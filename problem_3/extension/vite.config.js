import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const extRoot = dirname(__filename);

export default defineConfig({
    plugins: [
        vue()
    ],
    resolve: {
        alias: {
            '@': extRoot,
        },
    },
    build: {
        outDir: resolve(extRoot, 'dist'),
        emptyOutDir: true,
        lib: {
            entry: resolve(extRoot, 'content/main.js'),
            name: 'VisionMarkContent',
            formats: ['iife'],
            fileName: () => 'main.js'
        },
        cssCodeSplit: false
    },
    define: {
        'process.env.NODE_ENV': '"production"'
    }
});
