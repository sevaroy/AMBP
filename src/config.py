import os
from dotenv import load_dotenv
import subprocess
import sys

# 加載環境變量
load_dotenv()

# API配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# 檢查wkhtmltopdf是否已安裝
def is_wkhtmltopdf_installed():
    try:
        if sys.platform.startswith('win'):
            result = subprocess.run(['where', 'wkhtmltopdf'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            result = subprocess.run(['which', 'wkhtmltopdf'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False

WKHTMLTOPDF_INSTALLED = is_wkhtmltopdf_installed()

# 頁面配置
PAGE_CONFIG = {
    'page_title': "AI醫美智能評估系統",
    'page_icon': "",
    'layout': "wide",
    'initial_sidebar_state': "expanded",
    'menu_items': {
        'About': "AI醫美智能評估系統 - 專業版"
    }
}

# 自定義主題配置
CUSTOM_THEME = """
<style>
    :root {
        --primary-color: #4e97d1;
        --background-color: #f8fcff;
        --secondary-background-color: #e6f2ff;
        --text-color: #2c3e50;
        --font: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    /* ... 其他樣式配置 ... */
</style>
""" 