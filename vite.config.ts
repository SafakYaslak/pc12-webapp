import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'backend/public', // 🔹 Build çıktısı Flask tarafından servis edilecek klasöre gider
    emptyOutDir: true
  },
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
});