import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5分钟超时(SSE流式场景下可能更久)
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 流式生成旅行计划(SSE)
 * 通过fetch读取SSE流,实时获取后端处理进度,不再需要前端模拟进度
 */
export async function generateTripPlanStream(
  formData: TripFormData,
  onProgress?: (progress: number, message: string) => void,
  signal?: AbortSignal
): Promise<TripPlanResponse> {
  const response = await fetch(`${API_BASE_URL}/api/trip/plan/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData),
    signal,
  })

  if (!response.ok) {
    throw new Error(`服务器错误: ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('无法读取响应流')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  return new Promise<TripPlanResponse>((resolve, reject) => {
    const pump = () => {
      reader.read().then(({ done, value }) => {
        if (done) {
          reject(new Error('响应流意外结束'))
          return
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data: ')) continue

          try {
            const event = JSON.parse(trimmed.slice(6))

            if (event.type === 'progress' && onProgress) {
              onProgress(event.progress, event.message)
            } else if (event.type === 'result') {
              resolve({
                success: true,
                message: '旅行计划生成成功',
                data: event.data,
              })
              reader.cancel()
              return
            } else if (event.type === 'error') {
              reject(new Error(event.message || '生成旅行计划失败'))
              reader.cancel()
              return
            }
          } catch (e) {
            console.warn('解析SSE事件失败:', trimmed)
          }
        }

        pump()
      }).catch(reject)
    }

    pump()
  })
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient

