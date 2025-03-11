import requests
import time
from ..config import DEEPSEEK_API_KEY

class APIService:
    @staticmethod
    def check_deepseek_api():
        """檢查 DeepSeek API 的可用性和配置狀態"""
        if not DEEPSEEK_API_KEY:
            return False, "未配置 DeepSeek API 密鑰"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        test_payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        }
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=test_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return True, "API 連接正常"
                elif response.status_code == 401:
                    return False, "API 密鑰無效"
                elif response.status_code == 400:
                    error_msg = response.json().get("error", {}).get("message", "未知錯誤")
                    return False, f"API 請求錯誤: {error_msg}"
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return False, f"API 響應異常 (狀態碼: {response.status_code})"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False, "API 連接超時，請檢查網絡連接"
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False, f"API 連接失敗: {str(e)}"
        
        return False, "API 連接失敗，已達到最大重試次數"

    @staticmethod
    def analyze_with_deepseek(image_base64):
        """使用 DeepSeek API 進行面部分析"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        prompt = """請分析這張面部照片，並提供以下方面的專業評估：
1. 面部輪廓與對稱性
2. 皮膚狀況與紋理
3. 面部特徵分析（眼睛、鼻子、嘴巴等）
4. 衰老跡象評估
5. 色素問題
請使用專業的醫學美容術語進行描述。"""

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }

        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 401:
                    return "API 認證失敗，請檢查 API 密鑰設置。"
                elif response.status_code == 422:
                    if attempt == max_retries - 1:
                        error_data = response.json()
                        error_message = error_data.get("error", {}).get("message", "未知錯誤")
                        return f"API 請求格式錯誤: {error_message}"
                
                time.sleep(retry_delay)
                
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                if attempt == max_retries - 1:
                    return f"API 請求異常: {str(e)}"
                time.sleep(retry_delay)
        
        return "API 請求失敗，已達到最大重試次數"

    @staticmethod
    def generate_report(analysis_text):
        """使用 DeepSeek API 生成醫美建議報告"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        prompt = f"""基於以下面部分析結果，生成一份專業的醫美建議報告。報告應包含：
1. 面部狀況總結
2. 各區域問題分析
3. 建議的醫美治療方案
4. 優先處理順序
5. 注意事項和維護建議

分析結果：
{analysis_text}

請使用專業但易懂的語言，並確保建議的安全性和可行性。"""

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                report = result["choices"][0]["message"]["content"]
                disclaimer = "\n\n---\n\n⚠️ 免責聲明：本報告由 AI 系統生成，僅供參考。進行任何醫美治療前，請務必諮詢專業醫生的意見。"
                return report + disclaimer
            else:
                return None

        except Exception as e:
            print(f"報告生成過程中發生錯誤: {str(e)}")
            return None 