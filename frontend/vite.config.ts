import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import { config as dotenvConfig } from 'dotenv'
import { readFileSync } from 'fs'

// 加载根目录的 .env 文件
const rootDir = resolve(__dirname, '..')
const envFile = resolve(rootDir, '.env')
try {
  const envConfig = dotenvConfig({ path: envFile })
  // 将环境变量暴露给 Vite
  Object.entries(envConfig.parsed || {}).forEach(([key, value]) => {
    process.env[`VITE_${key}`] = value
  })
} catch (error) {
  console.warn('无法加载根目录 .env 文件，使用默认配置')
}

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
        target: process.env.VITE_API_BASE_URL || 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})

