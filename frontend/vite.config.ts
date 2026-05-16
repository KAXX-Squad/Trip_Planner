import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import { config as dotenvConfig } from 'dotenv'

// 加载根目录的 .env 文件
const rootDir = resolve(__dirname, '..')
const envFile = resolve(rootDir, '.env')

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  // 自动加载根目录的 .env 文件
  envDir: rootDir,
  // 指定需要暴露的前端环境变量
  envPrefix: ['VITE_', 'AMAP_']
})

