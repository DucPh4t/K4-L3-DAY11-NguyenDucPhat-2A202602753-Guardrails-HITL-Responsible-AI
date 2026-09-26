"""
Quick test script for DeepSeek API (deepseek-chat).
Run with: python scripts/test_deepseek.py
"""
import os
import sys
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.config import (
    get_red_provider,
    get_red_model,
    get_deepseek_api_key,
    red_openai_client_kwargs,
)

def test_deepseek_connection():
    api_key = get_deepseek_api_key()
    if not api_key:
        print("[!] Chưa cấu hình DEEPSEEK_API_KEY trong file .env!")
        print("    Vui lòng mở file .env và điền key thật của bạn:")
        print("    --------------------------------------------")
        print("    RED_TEAM_PROVIDER=deepseek")
        print("    DEEPSEEK_API_KEY=sk-...")
        print("    DEEPSEEK_MODEL=deepseek-chat")
        print("    DEEPSEEK_BASE_URL=https://api.deepseek.com")
        print("    --------------------------------------------")
        return False

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()

    print(f"[*] Provider: {get_red_provider()}")
    print(f"[*] Model: {model}")
    print(f"[*] Base URL: {base_url}")
    print(f"[*] API Key: {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        print("[*] Gửi request test tới deepseek-chat...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello! Reply with 1 short sentence."}],
            max_tokens=30,
        )
        reply = response.choices[0].message.content
        print(f"[PASS] Kết nối thành công tới DeepSeek API!")
        print(f"[*] Phản hồi từ model: {reply}")
        return True
    except Exception as e:
        print(f"[FAIL] Lỗi khi gọi DeepSeek API: {e}")
        return False

if __name__ == "__main__":
    test_deepseek_connection()
