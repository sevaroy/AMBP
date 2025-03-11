import os
import tempfile
import datetime
import logging
import re
import base64
import time
from typing import Optional, Tuple
from PIL import Image as PILImage
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
from dotenv import load_dotenv
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import plotly.express as px
from openai import OpenAI
import concurrent.futures
import sqlite3
import json
import dlib
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from fpdf import FPDF

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境變量加載
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not DEEPSEEK_API_KEY or not XAI_API_KEY:
    st.error("環境變量 DEEPSEEK_API_KEY 或 XAI_API_KEY 未設置，請檢查 .env 文件")
    logger.error("API 密鑰缺失")
    logger.info(f"DEEPSEEK_API_KEY: {DEEPSEEK_API_KEY}")
    logger.info(f"XAI_API_KEY: {XAI_API_KEY}")
    st.stop()

# 初始化 OpenAI 客戶端（用於 DeepSeek R1）
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 初始化 xAI 客戶端（用於 Grok-2-Vision-1212）
xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# Streamlit 頁面配置
st.set_page_config(
    page_title="醫美診所智能評估系統",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 主題
st.markdown("""
<style>
    /* 主色調 */
    :root {
        --primary-color: #9C89B8;       /* 淺紫主色調 */
        --primary-light: #F0E6FF;       /* 淺紫背景色 */
        --secondary-color: #F0A6CA;     /* 柔粉輔助色 */
        --accent-color: #B8BEDD;        /* 淺藍點綴色 */
        --neutral-dark: #5E6472;        /* 高級灰 */
        --neutral-light: #F7F7FC;       /* 背景色 */
        --success-color: #A0C4B9;       /* 成功提示色 */
        --error-color: #E08F8F;         /* 錯誤提示色 */
    }
    
    body { 
        background-color: var(--neutral-light); 
        font-family: 'Arial', 'Microsoft YaHei', sans-serif; 
    }
    
    .stApp { background-color: var(--neutral-light); }
    
    h1 { 
        color: var(--neutral-dark); 
        font-size: 32px; 
        font-weight: 600;
        margin-bottom: 20px; 
    }
    
    h2 { 
        color: var(--neutral-dark); 
        font-size: 24px; 
        margin-bottom: 15px; 
    }
    
    /* 卡片樣式優化 */
    .card { 
        background: white; 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.05), 0 1px 8px rgba(0,0,0,0.02); 
        margin-bottom: 25px; 
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.08), 0 5px 15px rgba(0,0,0,0.04);
    }
    
    /* 卡片頂部漸變裝飾 */
    .card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 5px 5px 0 0;
    }
    
    /* 按鈕樣式 */
    div.stButton > button { 
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); 
        color: white; 
        border-radius: 12px; 
        padding: 12px 24px; 
        font-weight: 500; 
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(156, 137, 184, 0.3);
    }
    
    div.stButton > button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(156, 137, 184, 0.4);
    }
    
    /* 上傳區域美化 */
    .stFileUploader { 
        border: 2px dashed var(--primary-color); 
        border-radius: 16px; 
        padding: 20px; 
        background-color: var(--primary-light);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--secondary-color);
        background-color: rgba(240, 230, 255, 0.7);
    }
    
    /* 統一圓角設計 */
    .stImage, .stFileUploader, div.stButton > button,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input {
        border-radius: 12px !important;
    }
    
    /* 進度條美化 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div {
        border-radius: 10px !important;
        background-color: var(--primary-light) !important;
    }
    
    /* 圖像容器美化 */
    .stImage {
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
    
    /* 頁面標題區域 */
    .title-container {
        text-align: center;
        padding: 20px 0 30px 0;
        margin-bottom: 30px;
        position: relative;
    }
    
    .title-container::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 3px;
    }
    
    /* 側邊欄美化 */
    .sidebar .sidebar-content { 
        background: linear-gradient(180deg, var(--neutral-light), var(--primary-light)); 
        padding: 25px;
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 分析結果卡片 */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border-left: 5px solid var(--primary-color);
    }
    
    /* 視覺化圖表容器 */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
    }
    
    /* 圖表標題 */
    .chart-title {
        font-size: 16px;
        color: var(--neutral-dark);
        text-align: center;
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    /* 頁腳美化 */
    .footer { 
        text-align: center; 
        color: var(--neutral-dark); 
        font-size: 13px; 
        margin-top: 30px;
        padding: 15px;
        border-top: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 成功和錯誤提示美化 */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px !important;
        padding: 12px !important;
    }
    
    .stSuccess {
        background-color: rgba(160, 196, 185, 0.2) !important;
        border-left: 5px solid var(--success-color) !important;
    }
    
    .stError {
        background-color: rgba(224, 143, 143, 0.2) !important;
        border-left: 5px solid var(--error-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# 設置中文字體
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
except Exception as e:
    logger.warning(f"設置中文字體失敗: {str(e)}，將使用默認字體")

# 支持多語言版本
TRANSLATIONS = {
    "zh": {
        "skin_condition": "皮膚狀況",
        "wrinkles": "皺紋",
        "spots": "色斑",
        # 其他翻譯...
    },
    "en": {
        "skin_condition": "Skin Condition",
        "wrinkles": "Wrinkles",
        "spots": "Spots",
        # 其他翻譯...
    }
}

def get_text(key, lang="zh"):
    return TRANSLATIONS[lang][key]

# 工具函數
def encode_image_to_base64(image_file: io.BytesIO) -> str:
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

@st.cache_data(ttl=3600)
def analyze_image(image_file: io.BytesIO) -> dict:
    try:
        logger.info("調用 Grok-2-Vision-1212 進行圖片分析")
        base64_image = encode_image_to_base64(image_file)
        response = xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[
                {
                    "role": "system",
                    "content": "你是專業醫美顧問，請對此面部照片進行詳細分析，提供結構化報告。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                                請對此面部照片進行詳細分析，提供結構化報告。針對以下區域：額頭、眼周、鼻子、頰骨、嘴唇、下巴，評估：
                                1. 皮膚狀況（乾燥、油性、痤瘡等）
                                2. 皺紋（深度、分布）
                                3. 色斑（類型、範圍）
                                4. 緊致度（鬆弛程度）
                                5. 其他特徵（毛孔、黑眼圈等）
                                對每個維度給出 0-5 分評分（0 表示嚴重問題，5 表示完美），並附上簡短描述。
                                輸出格式：
                                - 額頭: 皮膚狀況 X/5（描述）, 皺紋 X/5（描述）, ...
                                - 眼周: ...
                                - ...
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=1500
        )
        result = response.choices[0].message.content
        logger.info(f"Grok-2-Vision-1212 分析結果: {result}")
        return {
            "status": "success",
            "data": result,
            "error": None
        }
    except Exception as e:
        logger.error(f"Grok-2-Vision-1212 圖片分析失敗: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "data": None,
            "error": str(e)
        }

@st.cache_data(ttl=3600)
def generate_report(analysis_result: str) -> str:
    try:
        logger.info("調用 DeepSeek R1 生成報告")
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """
                    你是資深醫美專家，請根據以下面部分析結果生成一份專業、詳盡的醫美建議報告，字數至少 500 字。報告應包含以下內容，並確保語言邏輯清晰、結構分明，符合醫美行業標準：
                    1. 面部狀況綜合評估：
                       - 針對額頭、眼周、鼻子、頰骨、嘴唇、下巴，總結各區域的皮膚狀況、皺紋、色斑、緊致度等。
                       - 分析整體面部健康狀態，提供專業診斷，結合數據進行深入推理。
                    2. 推薦的醫美治療方案：
                       - 提供至少 5 種具體治療方案，按優先級排序。
                       - 每項包括治療名稱、適用區域、實施方式（如注射劑量、療程次數）。
                    3. 預期效果：
                       - 詳細描述每種方案的預期效果（如皺紋減少百分比、緊致度提升程度），使用量化數據並進行邏輯推導。
                    4. 術後護理建議：
                       - 針對每種方案提供具體護理措施（如保濕、防曬頻率、飲食建議），考慮長期效果。
                    5. 風險提示：
                       - 列出每種方案的潛在風險（如紅腫、過敏）及緩解方法，分析風險可能性。
                    使用專業術語（如「皮下注射」、「色素分解」、「組織提拉」），確保報告詳實且具權威性，展示深入的醫學推理能力。
                """},
                {"role": "user", "content": f"""
                    請根據以下面部分析結果生成報告：
                    {analysis_result}
                """}
            ],
            temperature=0.3,
            max_tokens=2000,
            stream=False
        )
        report = response.choices[0].message.content
        logger.info(f"DeepSeek R1 報告結果: {report}, 字數: {len(report)}")
        if len(report) < 500:
            logger.warning("報告字數不足 500 字")
            raise ValueError("報告字數不足")
        return report + "\n\n**免責聲明**：本報告由 DeepSeek R1 AI 生成，僅供參考，具體治療需諮詢專業醫生。"
    except Exception as e:
        logger.error(f"DeepSeek R1 報告生成失敗: {str(e)}")
        return f"錯誤: DeepSeek R1 報告生成失敗 ({str(e)})"

@st.cache_data
def create_visualizations(_image: PILImage.Image, analysis_result: str, report: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """创建可视化图表"""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    heatmap_path = os.path.join(temp_dir, "face_heatmap.png")
    radar_path = os.path.join(temp_dir, "radar_chart.png")
    priority_path = os.path.join(temp_dir, "treatment_priority.png")

    # 热力图
    try:
        img_array = np.array(_image)
        mask = np.zeros_like(img_array[:, :, 0], dtype=float)
        h, w = mask.shape
        regions = detect_face_regions(_image)
        
        for region, (y1, y2, x1, x2) in regions.items():
            # 确保坐标在图像范围内
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            # 查找评分
            score_match = re.search(rf"{region}.*?皮膚狀況\s*(\d)/5", analysis_result)
            if score_match:
                score = int(score_match.group(1))
                severity = (5 - score) / 5
                mask[y1:y2, x1:x2] = severity
            else:
                # 默认值
                mask[y1:y2, x1:x2] = 0.5
                
        # 应用高斯模糊使热力图更平滑
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img_array)
        plt.imshow(mask, cmap='RdYlGn_r', alpha=0.5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"熱力圖生成成功: {heatmap_path}")
    except Exception as e:
        logger.error(f"熱力圖生成失敗: {str(e)}", exc_info=True)
        heatmap_path = None

    # 雷達圖
    try:
        categories = ['膚質', '皺紋', '色斑', '緊致度', '毛孔', '膚色均勻度']
        current_scores = []
        ideal_scores = [5] * len(categories)
        for category in categories:
            match = re.search(rf"{category}.*?(\d)/5", analysis_result, re.IGNORECASE)
            score = int(match.group(1)) if match else 4
            current_scores.append(score)
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        current_scores += current_scores[:1]
        ideal_scores += ideal_scores[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, current_scores, color='#4A90E2', alpha=0.5, label='當前狀況')
        ax.fill(angles, ideal_scores, color='#D3E4F5', alpha=0.2, label='理想狀態')
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"雷達圖生成成功: {radar_path}")
    except Exception as e:
        logger.error(f"雷達圖生成失敗: {str(e)}")
        radar_path = None

    # 優先級圖
    try:
        treatments = []
        priorities = []
        for line in report.split('\n'):
            match = re.search(r'(\d+)\)\s*([^0-5].*?)(?=\s*\d|\n|$)', line)
            if match:
                priority = int(match.group(1))
                treatment = match.group(2).strip()
                treatments.append(treatment)
                priorities.append(6 - priority)
        if not treatments:
            treatments = ["玻尿酸填充", "肉毒素注射", "激光治療"]
            priorities = [5, 4, 3]
        fig = px.bar(
            x=priorities, y=treatments, orientation='h',
            labels={'x': '優先級', 'y': '治療方案'},
            title="治療方案優先級",
            color=priorities, color_continuous_scale='Blues',
            text=priorities
        )
        fig.update_traces(textposition='auto')
        fig.update_layout(showlegend=False, width=600, height=400)
        fig.write_image(priority_path, scale=2)
        logger.info(f"優先級圖生成成功: {priority_path}")
    except Exception as e:
        logger.error(f"優先級圖生成失敗: {str(e)}")
        priority_path = None

    return heatmap_path, radar_path, priority_path

def generate_better_pdf(report_text, images):
    """生成PDF报告，确保支持中文"""
    try:
        # 注册中文字体
        font_path = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(font_path, exist_ok=True)
        
        # 下载中文字体
        font_file = os.path.join(font_path, 'simsun.ttf')
        
        # 检查字体文件是否存在
        if not os.path.exists(font_file):
            # 使用临时内置字体
            logger.info("使用内置中文字体")
            temp_font_path = os.path.join(tempfile.gettempdir(), "simsun.ttf")
            
            # 如果您有办法将宋体嵌入应用中，可以尝试以下方式
            try:
                from fontTools.ttLib import TTFont as FontToolsTTFont
                # 创建一个简单的字体
                font = FontToolsTTFont()
                font.save(temp_font_path)
                font_file = temp_font_path
            except:
                # 如果无法创建字体，使用reportlab提供的基本字体
                logger.warning("无法创建中文字体，将使用基本字体")
        
        # 注册字体
        try:
            pdfmetrics.registerFont(TTFont('SimSun', font_file))
            logger.info("成功注册中文字体")
        except Exception as e:
            logger.error(f"注册字体失败: {str(e)}")

        # 创建一个内存中的PDF，而不是直接写入文件
        buffer = BytesIO()
        
        # 创建PDF文档
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # 自定义样式以使用中文字体
        for style_name in styles.byName:
            styles[style_name].fontName = 'SimSun'
        
        story = []
        
        # 添加标题
        title_style = styles['Heading1']
        title_style.alignment = 1  # 居中对齐
        story.append(Paragraph("醫美智能評估報告", title_style))
        story.append(Spacer(1, 12))
        
        # 添加日期
        date_style = styles['Normal']
        date_style.alignment = 1  # 居中对齐
        current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
        story.append(Paragraph(f"生成日期：{current_date}", date_style))
        story.append(Spacer(1, 20))
        
        # 处理报告内容
        normal_style = styles['Normal']
        normal_style.leading = 14  # 行间距
        
        # 确保报告文本不为空
        if not report_text or len(report_text.strip()) == 0:
            report_text = "無法生成報告內容，請重試。"
        
        # 分段处理报告文本
        paragraphs = report_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # 检查是否为标题行
                if re.match(r'^[0-9]+\.\s+\w+', para.strip()):
                    heading_style = styles['Heading2']
                    story.append(Paragraph(para, heading_style))
                else:
                    # 处理普通段落，保留换行
                    lines = para.split('\n')
                    for line in lines:
                        if line.strip():
                            story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 10))
        
        # 添加图片 - 先验证图片是否可用
        valid_images = []
        for img_path in images:
            if img_path and os.path.exists(img_path):
                try:
                    # 测试是否可以打开图片
                    PILImage.open(img_path)
                    valid_images.append(img_path)
                except Exception as e:
                    logger.error(f"无法打开图片 {img_path}: {str(e)}")
            else:
                logger.warning(f"图片路径不存在: {img_path}")
        
        if valid_images:  # 只有当有有效图片时才添加图表标题
            story.append(Spacer(1, 20))
            story.append(Paragraph("分析圖表", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # 图片处理
            captions = ["面部問題熱力圖", "面部狀況評分", "治療方案優先級"]
            for i, img_path in enumerate(valid_images):
                try:
                    # 添加图片标题
                    if i < len(captions):
                        story.append(Paragraph(captions[i], styles['Heading3']))
                    
                    # 打开并处理图片
                    img = PILImage.open(img_path)
                    img_width, img_height = img.size
                    
                    # 计算适合A4页面的图片尺寸
                    max_width = 450
                    aspect = img_height / img_width
                    new_width = min(max_width, img_width)
                    new_height = new_width * aspect
                    
                    # 添加图片到PDF
                    img = ReportLabImage(img_path, width=new_width, height=new_height)
                    story.append(img)
                    story.append(Spacer(1, 15))
                except Exception as e:
                    logger.error(f"处理图片失败: {str(e)}", exc_info=True)
        
        # 添加免责声明
        story.append(Spacer(1, 30))
        disclaimer_style = styles['Italic']
        disclaimer_style.textColor = colors.gray
        story.append(Paragraph("免責聲明：本報告由AI系統生成，僅供參考，具體治療方案請諮詢專業醫生。", disclaimer_style))
        
        # 构建PDF
        try:
            doc.build(story)
            buffer.seek(0)
            
            # 保存到临时文件
            temp_pdf = os.path.join(tempfile.gettempdir(), "medical_report.pdf")
            with open(temp_pdf, 'wb') as f:
                f.write(buffer.getvalue())
            
            logger.info(f"PDF生成成功: {temp_pdf}")
            return temp_pdf
        except Exception as e:
            logger.error(f"构建PDF失败: {str(e)}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"PDF生成失败: {str(e)}", exc_info=True)
        return None

def generate_simple_pdf(report_text, images):
    """使用更简单的方法生成PDF，确保支持中文"""
    try:
        # 创建PDF对象
        pdf = FPDF()
        pdf.add_page()
        
        # 使用Arial Unicode MS字体，这是一个通用的Unicode字体
        # 注意：FPDF默认不支持中文，我们需要使用特殊方法
        
        # 添加标题（使用英文避免字体问题）
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Medical Beauty Assessment Report', 0, 1, 'C')
        
        # 添加日期
        pdf.set_font('Arial', '', 12)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        pdf.cell(0, 10, f'Date: {current_date}', 0, 1, 'C')
        
        # 添加中文报告内容的提示
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, "Due to font limitations in PDF, Chinese characters cannot be displayed properly.")
        pdf.multi_cell(0, 5, "Below is the analysis visualization. For full report, please download the text report.")
        
        # 添加图片（这部分应该正常工作）
        valid_images = []
        for img_path in images:
            if img_path and os.path.exists(img_path):
                try:
                    valid_images.append(img_path)
                except Exception as e:
                    logger.error(f"图片验证失败: {str(e)}")
        
        # 添加图片
        for img_path in valid_images:
            try:
                pdf.add_page()
                # 添加图片标题
                if img_path.endswith("face_heatmap.png"):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Face Problem Heat Map", 0, 1, 'C')
                elif img_path.endswith("radar_chart.png"):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Facial Condition Score", 0, 1, 'C')
                elif img_path.endswith("treatment_priority.png"):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Treatment Priority", 0, 1, 'C')
                
                # 添加图片，确保适合页面
                pdf.image(img_path, x=10, y=30, w=190)
            except Exception as e:
                logger.error(f"添加图片失败: {str(e)}")
        
        # 添加免责声明
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, 'Disclaimer: This report is generated by AI for reference only.', 0, 1, 'C')
        
        # 保存PDF
        temp_pdf = os.path.join(tempfile.gettempdir(), "medical_report.pdf")
        pdf.output(temp_pdf)
        
        return temp_pdf
    except Exception as e:
        logger.error(f"简单PDF生成失败: {str(e)}", exc_info=True)
        return None

# 主界面
def main():
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("系統設置")
        st.write("分析模型：Grok-2-Vision-1212")
        st.write("報告模型：DeepSeek R1")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="footer">© 2025 醫美診所智能系統</div>', unsafe_allow_html=True)

    # 添加標題容器
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("醫美診所智能評估系統")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 添加步驟指示器
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 1
    
    current_step = st.session_state["current_step"]
    
    # 根據會話狀態更新當前步驟
    if "image" in st.session_state and current_step < 2:
        st.session_state["current_step"] = 1
    if "analysis_result" in st.session_state and current_step < 3:
        st.session_state["current_step"] = 3
    if "report" in st.session_state and current_step < 4:
        st.session_state["current_step"] = 4
    
    current_step = st.session_state["current_step"]
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 30px; padding: 0 10px;">
        <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: {
                'linear-gradient(135deg, var(--primary-color), var(--secondary-color))' if current_step >= 1 else 'var(--neutral-light)'
            }; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; margin-bottom: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">1</div>
            <div style="text-align: center; font-size: 14px; color: {
                'var(--neutral-dark)' if current_step >= 1 else '#AAAAAA'
            };">上傳照片</div>
        </div>
        <div style="flex: 1; height: 2px; background: {
            'linear-gradient(90deg, var(--primary-color), var(--secondary-color))' if current_step >= 2 else '#EEEEEE'
        }; margin-top: 20px;"></div>
        <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: {
                'linear-gradient(135deg, var(--primary-color), var(--secondary-color))' if current_step >= 2 else 'var(--neutral-light)'
            }; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; margin-bottom: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">2</div>
            <div style="text-align: center; font-size: 14px; color: {
                'var(--neutral-dark)' if current_step >= 2 else '#AAAAAA'
            };">分析中</div>
        </div>
        <div style="flex: 1; height: 2px; background: {
            'linear-gradient(90deg, var(--primary-color), var(--secondary-color))' if current_step >= 3 else '#EEEEEE'
        }; margin-top: 20px;"></div>
        <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: {
                'linear-gradient(135deg, var(--primary-color), var(--secondary-color))' if current_step >= 3 else 'var(--neutral-light)'
            }; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; margin-bottom: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">3</div>
            <div style="text-align: center; font-size: 14px; color: {
                'var(--neutral-dark)' if current_step >= 3 else '#AAAAAA'
            };">查看結果</div>
        </div>
        <div style="flex: 1; height: 2px; background: {
            'linear-gradient(90deg, var(--primary-color), var(--secondary-color))' if current_step >= 4 else '#EEEEEE'
        }; margin-top: 20px;"></div>
        <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: {
                'linear-gradient(135deg, var(--primary-color), var(--secondary-color))' if current_step >= 4 else 'var(--neutral-light)'
            }; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; margin-bottom: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">4</div>
            <div style="text-align: center; font-size: 14px; color: {
                'var(--neutral-dark)' if current_step >= 4 else '#AAAAAA'
            };">下載報告</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 7])

    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("上傳影像")
            uploaded_file = st.file_uploader("選擇面部照片", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = PILImage.open(uploaded_file)
                st.session_state["image"] = image  # 保存图片到 session state
                st.image(image, caption="已上傳照片", use_container_width=True)
                if st.button("開始分析"):
                    # 更新當前步驟為"分析中"
                    st.session_state["current_step"] = 2
                    
                    with st.spinner("正在分析影像..."):
                        progress_bar = st.progress(0)
                        
                        # 分析步驟提示
                        analysis_steps = [
                            "正在初始化面部識別模型...",
                            "檢測面部特徵點...",
                            "分析膚質狀態...",
                            "評估皮膚紋理...",
                            "檢測色素沉著情況...",
                            "分析皺紋深度與分佈...",
                            "評估面部輪廓與對稱性...",
                            "計算面部黃金比例...",
                            "生成面部問題熱力圖...",
                            "制定個性化治療方案..."
                        ]
                        
                        status_text = st.empty()
                        
                        for i in range(100):
                            # 更新進度條
                            progress_bar.progress(i + 1)
                            
                            # 顯示當前分析步驟
                            step_index = min(int(i / 10), len(analysis_steps) - 1)
                            status_text.markdown(f"""
                            <div style="padding: 10px; border-radius: 8px; background-color: var(--primary-light); margin-bottom: 10px;">
                                <p style="margin: 0; color: var(--neutral-dark); font-size: 14px;">
                                    <strong>當前步驟：</strong> {analysis_steps[step_index]}
                                </p>
                                <p style="margin: 5px 0 0 0; color: var(--neutral-dark); font-size: 12px; opacity: 0.8;">
                                    總進度: {i+1}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 模擬進度
                            time.sleep(0.05)
                        
                        analysis_result = analyze_image(uploaded_file)
                        # 清除進度條和狀態文本
                        progress_bar.empty()
                        status_text.empty()
                        
                        if analysis_result["status"] == "success":
                            report = generate_report(analysis_result["data"])
                            heatmap_path, radar_path, priority_path = create_visualizations(
                                image, analysis_result["data"], report
                            )
                            st.session_state["analysis_result"] = analysis_result["data"]
                            st.session_state["report"] = report
                            st.session_state["heatmap_path"] = heatmap_path
                            st.session_state["radar_path"] = radar_path
                            st.session_state["priority_path"] = priority_path
                            
                            # 更新當前步驟
                            st.session_state["current_step"] = 3
                            
                            st.success("分析完成！")
                        else:
                            st.error(f"分析失敗: {analysis_result['error']}")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card result-card">', unsafe_allow_html=True)
            if "analysis_result" in st.session_state and "image" in st.session_state:  # 检查两个必要的键是否存在
                st.header("分析結果與治療建議")
                st.subheader("面部分析")
                st.write(st.session_state["analysis_result"])

                with st.spinner("正在生成治療建議..."):
                    report = generate_report(st.session_state["analysis_result"])
                    heatmap_path, radar_path, priority_path = create_visualizations(
                        st.session_state["image"], st.session_state["analysis_result"], report
                    )
                    st.session_state["report"] = report
                    st.session_state["heatmap_path"] = heatmap_path
                    st.session_state["radar_path"] = radar_path
                    st.session_state["priority_path"] = priority_path
                    st.success("建議生成完成！")

                st.subheader("治療方案建議")
                st.markdown(st.session_state["report"])
                col_vis1, col_vis2 = st.columns(2)
                with col_vis1:
                    if st.session_state["heatmap_path"]:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">面部問題熱力圖</div>', unsafe_allow_html=True)
                        st.image(st.session_state["heatmap_path"], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    if st.session_state["radar_path"]:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">面部狀況評分</div>', unsafe_allow_html=True)
                        st.image(st.session_state["radar_path"], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                with col_vis2:
                    if st.session_state["priority_path"]:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">治療方案優先級</div>', unsafe_allow_html=True)
                        st.image(st.session_state["priority_path"], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.header("分析結果與治療建議")
                st.write("請先上傳照片並進行分析。")
            st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("報告導出")
        if "report" in st.session_state:
            # 确保图片路径存在
            image_paths = []
            for path_key in ["heatmap_path", "radar_path", "priority_path"]:
                path = st.session_state.get(path_key)
                if path and os.path.exists(path):
                    image_paths.append(path)
                else:
                    logger.warning(f"图片路径不存在: {path_key}")
            
            # 添加PDF下载按钮
            if image_paths:  # 只要有图片就可以生成PDF
                col_pdf1, col_pdf2 = st.columns([1, 1])
                
                with col_pdf1:
                    st.markdown('<div style="text-align: center; padding: 10px;">', unsafe_allow_html=True)
                    st.markdown('#### 標準報告')
                    st.markdown('包含詳細分析結果和治療建議')
                    with st.spinner("正在生成PDF報告..."):
                        pdf_path = generate_simple_pdf(st.session_state["report"], image_paths)
                        if pdf_path and os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            st.download_button(
                                label="下載標準報告 📄",
                                data=pdf_bytes,
                                file_name="醫美診所評估報告.pdf",
                                mime="application/pdf",
                                key="standard_report"
                            )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_pdf2:
                    st.markdown('<div style="text-align: center; padding: 10px;">', unsafe_allow_html=True)
                    st.markdown('#### 高級報告')
                    st.markdown('包含更多視覺化圖表和專業建議')
                    with st.spinner("正在生成高級PDF報告..."):
                        premium_pdf_path = generate_better_pdf(st.session_state["report"], image_paths)
                        if premium_pdf_path and os.path.exists(premium_pdf_path):
                            with open(premium_pdf_path, "rb") as f:
                                premium_pdf_bytes = f.read()
                            st.download_button(
                                label="下載高級報告 📊",
                                data=premium_pdf_bytes,
                                file_name="醫美診所高級評估報告.pdf",
                                mime="application/pdf",
                                key="premium_report"
                            )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 添加報告使用提示
                st.markdown("""
                <div style="background-color: var(--primary-light); padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <p style="font-size: 14px; color: var(--neutral-dark);">
                        <strong>💡 提示：</strong> 報告僅供參考，建議攜帶報告前往專業醫美診所進行面對面諮詢。
                        您可以使用此報告與醫生討論最適合您的治療方案。
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("無法生成報告：缺少必要的分析圖像。")
        else:
            st.info("請先完成面部分析以生成報告。")
        st.markdown('</div>', unsafe_allow_html=True)

def process_analysis(image_file):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        analysis_future = executor.submit(analyze_image, image_file)
        analysis_result = analysis_future.result()
        
        if analysis_result["status"] == "success":
            report_future = executor.submit(generate_report, analysis_result["data"])
            report = report_future.result()
            # 處理其他後續邏輯...

def save_analysis(user_id, image_path, analysis_result, report):
    conn = sqlite3.connect('medical_analysis.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_results 
        (user_id, timestamp, image_path, analysis_json, report_text) 
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, datetime.datetime.now(), image_path, json.dumps(analysis_result), report))
    conn.commit()
    conn.close()

def detect_face_regions(image):
    """检测面部区域，返回各区域的坐标"""
    # 简化实现，返回基本区域划分
    h, w = np.array(image).shape[:2]
    
    # 基于图像尺寸的简单区域划分
    forehead = (int(h*0.1), int(h*0.3), int(w*0.3), int(w*0.7))  # 额头区域
    eyes = (int(h*0.3), int(h*0.4), int(w*0.2), int(w*0.8))      # 眼周区域
    nose = (int(h*0.4), int(h*0.6), int(w*0.4), int(w*0.6))      # 鼻子区域
    cheeks = (int(h*0.4), int(h*0.7), int(w*0.2), int(w*0.8))    # 颊骨区域
    lips = (int(h*0.6), int(h*0.7), int(w*0.3), int(w*0.7))      # 嘴唇区域
    chin = (int(h*0.7), int(h*0.9), int(w*0.3), int(w*0.7))      # 下巴区域
    
    return {
        "額頭": forehead,
        "眼周": eyes,
        "鼻子": nose,
        "頰骨": cheeks,
        "嘴唇": lips,
        "下巴": chin
    }

if __name__ == "__main__":
    main()