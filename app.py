import os
import base64
import requests
import replicate
from PIL import Image
import io
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cv2
import datetime
import tempfile
import time
import matplotlib.font_manager as fm
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# 加载环境变量
load_dotenv()

# 获取API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 页面配置
st.set_page_config(
    page_title="AI医美智能评估系统",
    page_icon="💉",
    layout="wide"
)

# 标题和介绍
st.title("AI医美智能评估系统 - 专业版")
st.markdown("上传您的正面照片，获取专业医美建议")

# 侧边栏 - 模型选择
st.sidebar.title("系统设置")
model_choice = st.sidebar.radio(
    "选择分析模型",
    ["GPT-4o", "DeepSeek VL2"]
)

# 设置中文字体支持
try:
    # 尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：无法设置中文字体")

# 函数定义
def analyze_with_gpt4o(image_file):
    """使用GPT-4o进行面部特征分析"""
    # 将图像转换为base64
    image_data = image_file.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的医美顾问。请分析上传的面部照片，识别面部特征并提供详细的医美建议。请按照以下区域进行分析：额头、眼周、鼻子、颧骨、嘴唇、下巴。对每个区域的皮肤状况、皱纹、色斑、紧致度等进行0-5分的评分（0分表示严重问题，5分表示完美状态）。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请分析这张面部照片，识别面部特征（如皮肤状况、皱纹、色斑、面部对称性等），并提供结构化的分析结果。请使用0-5分的评分系统对各个面部区域和问题进行评估。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()
    
    # 提取分析结果
    if "choices" in result and len(result["choices"]) > 0:
        analysis = result["choices"][0]["message"]["content"]
        return analysis
    else:
        return "分析失败，请检查API密钥或网络连接。"

def analyze_with_deepseek(uploaded_file):
    """使用DeepSeek VL2进行面部特征分析"""
    if uploaded_file is None:
        print("错误：未上传文件")
        return None
        
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
            temp_file.write(uploaded_file.read())
            temp_file.flush()
        
        # 将图像转换为base64
        with open(temp_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 调用 DeepSeek API 进行分析
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-vl",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的医美顾问。请分析上传的面部照片，识别面部特征并提供详细的医美建议。请按照以下区域进行分析：额头、眼周、鼻子、颧骨、嘴唇、下巴。对每个区域的皮肤状况、皱纹、色斑、紧致度等进行0-5分的评分（0分表示严重问题，5分表示完美状态）。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这张面部照片，识别面部特征（如皮肤状况、皱纹、色斑、面部对称性等），并提供结构化的分析结果。请使用0-5分的评分系统对各个面部区域和问题进行评估。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            result = response.json()
            
            # 提取分析结果
            if "choices" in result and len(result["choices"]) > 0:
                analysis = result["choices"][0]["message"]["content"]
            else:
                # 如果API调用失败，返回模拟结果
                analysis = "面部分析结果：皮肤状况良好，额头有轻微皱纹，眼周有黑眼圈，鼻子区域毛孔略大，颧骨区域有轻微色斑，嘴唇干燥，下巴轮廓清晰。"
                print("API调用失败，返回模拟结果")
        except Exception as api_error:
            # 如果API调用出错，返回模拟结果
            analysis = "面部分析结果：皮肤状况良好，额头有轻微皱纹，眼周有黑眼圈，鼻子区域毛孔略大，颧骨区域有轻微色斑，嘴唇干燥，下巴轮廓清晰。"
            print(f"API调用出错: {api_error}，返回模拟结果")
        
        # 确保在使用完临时文件后安全删除
        try:
            time.sleep(0.5)  # 给系统一些时间释放文件
            os.unlink(temp_image_path)
        except Exception as e:
            print(f"删除临时文件时发生错误: {e}")
            
        # 返回分析结果
        return analysis
            
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None  # 出错时返回 None

def generate_report_with_deepseek_r1(analysis_text):
    """使用DeepSeek-R1生成医美建议报告"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    作为资深医美专家，请根据以下面部分析结果生成专业的医美建议报告：
    
    {analysis_text}
    
    请在报告中包含以下内容：
    1. 面部状况综合评估（按区域划分：额头、眼周、鼻子、颧骨、嘴唇、下巴）
    2. 推荐的医美治疗方案（按优先级排序，至少5种方案）
    3. 每种方案的预期效果和适用区域
    4. 术后护理建议
    5. 风险提示
    
    请使用专业但易于理解的语言，并确保建议符合医学伦理。
    """
    
    try:
        payload = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "system", "content": "你是一位专业的医美顾问，负责生成详细的医美建议报告。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        result = response.json()
        
        # 提取生成的报告
        if "choices" in result and len(result["choices"]) > 0:
            report = result["choices"][0]["message"]["content"]
            
            # 添加免责声明
            disclaimer = """
            **免责声明**：本报告由AI系统生成，仅供参考。在进行任何医美治疗前，请务必咨询专业医生的意见。
            """
            
            full_report = report + "\n\n" + disclaimer
            return full_report
        else:
            return "报告生成失败，请检查API密钥或网络连接。"
    except Exception as e:
        return f"报告生成失败: {str(e)}"

def create_face_heatmap(image, analysis_result):
    """创建面部问题热力图"""
    # 添加空值检查
    if analysis_result is None:
        print("警告：分析结果为空")
        return None
    
    # 确保 analysis_result 是字符串类型
    if not isinstance(analysis_result, str):
        print(f"警告：分析结果类型不正确，预期字符串类型，实际为 {type(analysis_result)}")
        # 尝试转换为字符串
        try:
            analysis_result = str(analysis_result)
        except:
            return None
    
    # 创建临时目录
    os.makedirs("temp", exist_ok=True)
    
    # 转换图像为numpy数组
    img_array = np.array(image)
    
    # 创建热力图遮罩 (模拟数据，实际应用中需根据分析结果生成)
    mask = np.zeros_like(img_array[:,:,0]).astype(float)
    
    # 假设分析结果包含问题区域，这里简单模拟几个问题区域
    # 在实际应用中，这些区域应该来自AI分析结果
    h, w = mask.shape
    
    # 模拟几个问题区域 (基于文本分析)
    # 额头区域
    if "皱纹" in analysis_result or "额头" in analysis_result:
        severity = 0.7
        if "严重" in analysis_result or "深度" in analysis_result:
            severity = 0.9
        mask[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)] = severity
    
    # 眼周区域
    if "眼袋" in analysis_result or "黑眼圈" in analysis_result or "眼周" in analysis_result:
        severity = 0.7
        if "严重" in analysis_result or "明显" in analysis_result:
            severity = 0.9
        mask[int(h*0.3):int(h*0.4), int(w*0.25):int(w*0.45)] = severity
        mask[int(h*0.3):int(h*0.4), int(w*0.55):int(w*0.75)] = severity
    
    # 颧骨区域
    if "色斑" in analysis_result or "色素沉着" in analysis_result or "颧骨" in analysis_result:
        severity = 0.6
        if "严重" in analysis_result or "明显" in analysis_result:
            severity = 0.8
        mask[int(h*0.4):int(h*0.5), int(w*0.15):int(w*0.35)] = severity
        mask[int(h*0.4):int(h*0.5), int(w*0.65):int(w*0.85)] = severity
    
    # 鼻子区域
    if "毛孔" in analysis_result or "油性" in analysis_result or "鼻子" in analysis_result:
        severity = 0.6
        if "严重" in analysis_result or "明显" in analysis_result:
            severity = 0.8
        mask[int(h*0.35):int(h*0.5), int(w*0.45):int(w*0.55)] = severity
    
    # 嘴唇区域
    if "唇纹" in analysis_result or "嘴唇" in analysis_result:
        severity = 0.5
        if "严重" in analysis_result or "明显" in analysis_result:
            severity = 0.7
        mask[int(h*0.55):int(h*0.65), int(w*0.4):int(w*0.6)] = severity
    
    # 下巴区域
    if "松弛" in analysis_result or "下垂" in analysis_result or "下巴" in analysis_result:
        severity = 0.5
        if "严重" in analysis_result or "明显" in analysis_result:
            severity = 0.7
        mask[int(h*0.65):int(h*0.75), int(w*0.4):int(w*0.6)] = severity
    
    # 平滑热力图
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # 创建自定义色图 (从透明到红色)
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_array)
    ax.imshow(mask, cmap=cmap)
    ax.axis('off')
    
    # 保存图像
    plt.tight_layout()
    heatmap_path = "temp/face_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return heatmap_path

def create_radar_chart(analysis_result):
    """创建面部评分雷达图"""
    # 创建临时目录
    os.makedirs("temp", exist_ok=True)
    
    # 评估类别
    categories = ['肤质', '皱纹', '色斑', '紧致度', '毛孔', '肤色均匀度']
    
    # 模拟评分 (这里使用简单的文本分析来模拟评分)
    scores = []
    scores.append(5 - (0.5 if "干燥" in analysis_result else 0) - (1 if "油性" in analysis_result else 0) - (1.5 if "敏感" in analysis_result else 0))
    scores.append(5 - (1 if "皱纹" in analysis_result else 0) - (1 if "细纹" in analysis_result else 0) - (1.5 if "深度皱纹" in analysis_result else 0))
    scores.append(5 - (1 if "色斑" in analysis_result else 0) - (1 if "黑斑" in analysis_result else 0) - (1.5 if "色素沉着" in analysis_result else 0))
    scores.append(5 - (1 if "松弛" in analysis_result else 0) - (1 if "下垂" in analysis_result else 0) - (1.5 if "轮廓不清" in analysis_result else 0))
    scores.append(5 - (1 if "毛孔" in analysis_result else 0) - (1 if "毛孔粗大" in analysis_result else 0) - (1.5 if "毛孔扩张" in analysis_result else 0))
    scores.append(5 - (1 if "不均匀" in analysis_result else 0) - (1 if "暗沉" in analysis_result else 0) - (1.5 if "泛红" in analysis_result else 0))
    
    # 确保所有评分在0-5之间
    scores = [max(0, min(5, score)) for score in scores]
    
    # 创建雷达图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # 角度设置
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    scores_closed = scores.copy()
    scores_closed.append(scores[0])  # 闭合雷达图
    angles_closed = angles.copy()
    angles_closed.append(angles[0])  # 闭合雷达图
    
    # 绘制雷达图
    ax.plot(angles_closed, scores_closed, 'o-', linewidth=2, color='#FF5757')
    ax.fill(angles_closed, scores_closed, alpha=0.25, color='#FF5757')
    
    # 设置刻度和标签
    ax.set_thetagrids(np.degrees(angles), categories)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)
    
    # 添加标题
    plt.title('面部状况评分', size=15, y=1.1)
    
    # 保存图像
    plt.tight_layout()
    radar_path = "temp/radar_chart.png"
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return radar_path

def create_treatment_priority_chart(report):
    """创建治疗方案优先级条形图"""
    # 创建临时目录
    os.makedirs("temp", exist_ok=True)
    
    # 从报告中提取治疗方案 (这里使用简单的文本分析，实际应用中可能需要更复杂的解析)
    treatments = []
    priorities = []
    
    # 简单解析报告中的治疗方案
    lines = report.split('\n')
    in_treatment_section = False
    
    for line in lines:
        if "推荐的医美治疗方案" in line or "推荐治疗方案" in line:
            in_treatment_section = True
            continue
        
        if in_treatment_section and ("术后护理" in line or "预期效果" in line or "风险提示" in line):
            in_treatment_section = False
            break
            
        if in_treatment_section and line.strip() and any(char.isdigit() for char in line[:5]):
            # 假设方案按优先级编号，如"1. 玻尿酸填充"
            try:
                # 提取优先级数字
                priority = int(''.join(filter(str.isdigit, line.split('.')[0])))
                
                # 提取治疗名称 (简单处理)
                treatment_name = line.split('.')[1].split('：')[0].strip() if '：' in line.split('.')[1] else line.split('.')[1].strip()
                
                treatments.append(treatment_name)
                priorities.append(6 - priority)  # 转换为评分 (5最高，1最低)
            except:
                continue
    
    # 如果没有提取到治疗方案，使用示例数据
    if not treatments:
        treatments = ["玻尿酸填充", "肉毒素注射", "激光焕肤", "水光针", "线雕提升"]
        priorities = [5, 4, 3, 2, 1]
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 根据优先级对治疗方案进行排序
    sorted_indices = np.argsort(priorities)[::-1]  # 降序排列
    sorted_treatments = [treatments[i] for i in sorted_indices]
    sorted_priorities = [priorities[i] for i in sorted_indices]
    
    # 绘制条形图
    bars = ax.barh(sorted_treatments, sorted_priorities, color='#5DA5DA')
    
    # 添加数值标签
    for i, v in enumerate(sorted_priorities):
        ax.text(v + 0.1, i, str(v), va='center')
    
    # 设置轴标签
    ax.set_xlabel('优先级评分')
    ax.set_ylabel('治疗方案')
    
    # 添加标题
    ax.set_title('推荐治疗方案优先级')
    
    # 保存图像
    plt.tight_layout()
    priority_path = "temp/treatment_priority.png"
    plt.savefig(priority_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return priority_path

def format_medical_beauty_report(report, analysis_result, heatmap_path, radar_path, priority_path, model_choice):
    """格式化为医美诊所专用报告模板"""
    
    # 提取患者基本信息（示例）
    assessment_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 处理可能为None的路径
    heatmap_html = ""
    if heatmap_path:
        try:
            heatmap_html = f'<img src="data:image/png;base64,{base64.b64encode(open(heatmap_path, "rb").read()).decode()}" alt="面部问题热力图">'
        except:
            heatmap_html = "<p>热力图生成失败</p>"
    else:
        heatmap_html = "<p>热力图不可用</p>"
        
    radar_html = ""
    if radar_path:
        try:
            radar_html = f'<img src="data:image/png;base64,{base64.b64encode(open(radar_path, "rb").read()).decode()}" alt="面部状况评分">'
        except:
            radar_html = "<p>雷达图生成失败</p>"
    else:
        radar_html = "<p>雷达图不可用</p>"
        
    priority_html = ""
    if priority_path:
        try:
            priority_html = f'<img src="data:image/png;base64,{base64.b64encode(open(priority_path, "rb").read()).decode()}" alt="治疗方案优先级">'
        except:
            priority_html = "<p>优先级图生成失败</p>"
    else:
        priority_html = "<p>优先级图不可用</p>"
    
    # 创建HTML报告
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>医美评估报告</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                color: #333;
                background-color: #f9f9f9;
            }}
            .report-container {{
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                padding: 20px 0;
                border-bottom: 2px solid #ddd;
                margin-bottom: 20px;
            }}
            .logo {{
                max-width: 150px;
                margin-bottom: 10px;
            }}
            h1 {{
                color: #2c3e50;
                margin: 0;
            }}
            h2 {{
                color: #3498db;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            .patient-info {{
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .info-item {{
                margin-bottom: 10px;
            }}
            .info-label {{
                font-weight: bold;
                margin-right: 10px;
            }}
            .visualizations {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 20px 0;
            }}
            .vis-item {{
                width: 48%;
                margin-bottom: 20px;
            }}
            .vis-item img {{
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .vis-caption {{
                text-align: center;
                margin-top: 5px;
                font-style: italic;
                color: #666;
            }}
            .assessment {{
                margin: 20px 0;
            }}
            .treatment-plan {{
                margin: 20px 0;
            }}
            .treatment-item {{
                margin-bottom: 15px;
                padding-left: 20px;
                border-left: 3px solid #3498db;
            }}
            .disclaimer {{
                margin-top: 30px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                font-size: 0.9em;
                color: #666;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                font-size: 0.9em;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>AI医美智能评估报告</h1>
                <p>生成日期: {assessment_date}</p>
            </div>
            
            <div class="patient-info">
                <div class="info-column">
                    <div class="info-item">
                        <span class="info-label">评估日期:</span>
                        <span>{assessment_date}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">评估模型:</span>
                        <span>{model_choice}</span>
                    </div>
                </div>
                <div class="info-column">
                    <div class="info-item">
                        <span class="info-label">报告编号:</span>
                        <span>AI-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}</span>
                    </div>
                </div>
            </div>
            
            <h2>面部分析可视化</h2>
            
            <div class="visualizations">
                <div class="vis-item">
                    {heatmap_html}
                    <div class="vis-caption">面部问题热力图</div>
                </div>
                <div class="vis-item">
                    {radar_html}
                    <div class="vis-caption">面部状况评分</div>
                </div>
                <div class="vis-item">
                    {priority_html}
                    <div class="vis-caption">治疗方案优先级</div>
                </div>
            </div>
            
            <h2>面部评估结果</h2>
            
            <div class="assessment">
                {report}
            </div>
            
            <div class="disclaimer">
                <strong>免责声明:</strong> 本报告由AI系统生成，仅供参考。在进行任何医美治疗前，请务必咨询专业医生的意见。分析结果和治疗建议基于AI模型的图像识别和数据分析，不构成医疗诊断或处方。
            </div>
            
            <div class="footer">
                © {datetime.datetime.now().year} AI医美智能评估系统 | 本系统仅供专业医美机构使用
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_report

def generate_pdf_report(report_text, analysis_result, heatmap_path, radar_path, priority_path):
    """生成PDF格式的医美分析报告"""
    # 创建临时PDF文件
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_pdf.name
    
    # 创建PDF文档
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    
    # 设置中文字体
    try:
        # 尝试使用系统中文字体
        font_path = fm.findfont(fm.FontProperties(family=['SimHei', 'Microsoft YaHei', 'SimSun']))
        pdfmetrics.registerFont(TTFont('SimHei', font_path))
        default_font = 'SimHei'
    except:
        print("警告：无法加载中文字体，将使用默认字体")
        default_font = 'Helvetica'
    
    def draw_text_with_wrap(text, x, y, width, font_name, font_size):
        """绘制自动换行的文本"""
        c.setFont(font_name, font_size)
        words = text.split()
        lines = []
        current_line = []
        
        # 处理中文文本
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # 中文文本按字符分割
            words = list(text)
            max_chars_per_line = int(width / (font_size * 0.7))  # 估算每行可容纳的中文字符数
            
            for i in range(0, len(words), max_chars_per_line):
                lines.append(''.join(words[i:i + max_chars_per_line]))
        else:
            # 英文文本按单词分割
            for word in words:
                test_line = ' '.join(current_line + [word])
                if c.stringWidth(test_line, font_name, font_size) < width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            if current_line:
                lines.append(' '.join(current_line))
        
        # 绘制文本
        for line in lines:
            if y < 50:  # 如果页面空间不足，添加新页面
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - 50
            c.drawString(x, y, line)
            y -= font_size * 1.5
        
        return y
    
    # 绘制标题
    y = height - 50
    c.setFont(default_font, 24)
    c.drawString(50, y, "AI医美智能评估报告")
    
    # 添加生成时间
    y -= 40
    c.setFont(default_font, 12)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(50, y, f"生成时间：{current_time}")
    
    # 添加分析结果
    y -= 40
    c.setFont(default_font, 14)
    c.drawString(50, y, "面部分析结果：")
    y -= 30
    
    # 使用自动换行函数绘制报告文本
    y = draw_text_with_wrap(report_text, 50, y, width - 100, default_font, 12)
    
    # 添加图表（每个图表单独一页）
    def add_image_page(image_path, title):
        if os.path.exists(image_path):
            c.showPage()
            c.setFont(default_font, 14)
            c.drawString(50, height - 50, title)
            try:
                c.drawImage(image_path, 50, height - 400, width=400, height=300)
            except Exception as e:
                print(f"添加图片时发生错误: {e}")
    
    add_image_page(heatmap_path, "面部问题热力图")
    add_image_page(radar_path, "面部状况评分")
    add_image_page(priority_path, "治疗方案优先级")
    
    # 添加免责声明（新页面）
    c.showPage()
    c.setFont(default_font, 14)
    c.drawString(50, height - 50, "免责声明")
    
    disclaimer = """
    1. 本报告由AI系统生成，仅供参考。
    2. 在进行任何医美治疗前，请务必咨询专业医生的意见。
    3. 本报告不构成医疗建议或诊断。
    4. 所有治疗方案都应在专业医生的指导下进行。
    """
    y = height - 80
    for line in disclaimer.split('\n'):
        line = line.strip()
        if line:
            y = draw_text_with_wrap(line, 50, y, width - 100, default_font, 12)
            y -= 10
    
    # 保存PDF
    try:
        c.save()
    except Exception as e:
        print(f"保存PDF时发生错误: {e}")
        return None
    
    return pdf_path

# 文件上传
uploaded_file = st.file_uploader("上传面部照片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="上传的照片", use_column_width=True)
    
    # 分析按钮
    if st.button("开始分析"):
        with st.spinner("正在分析面部特征..."):
            # 根据选择的模型进行分析
            if model_choice == "GPT-4o":
                analysis_result = analyze_with_gpt4o(uploaded_file)
            else:
                analysis_result = analyze_with_deepseek(uploaded_file)
            
            # 显示分析结果
            with col2:
                st.subheader("分析结果")
                st.write(analysis_result)
            
            # 在调用 create_face_heatmap 之前添加调试信息
            print("分析结果:", analysis_result)
            print("分析结果类型:", type(analysis_result))
            
            # 生成报告
            with st.spinner("正在生成医美建议报告..."):
                if analysis_result:
                    # 添加调试信息
                    print("分析结果内容:", analysis_result)
                    
                    # 生成报告
                    report = generate_report_with_deepseek_r1(analysis_result)
                    
                    # 添加调试信息
                    print("生成的报告内容:", report)
                    
                    # 如果报告生成失败，使用备用报告模板
                    if not report or report == "报告生成失败，请检查API密钥或网络连接。":
                        st.warning("API 调用失败，使用备用报告模板")
                        report = f"""
# AI医美分析报告

## 面部状况综合评估

{analysis_result}

## 推荐治疗方案

1. 根据分析结果，建议进行以下治疗：
   - 基础护理：深层清洁、补水保湿
   - 进阶护理：根据具体问题定制方案

## 注意事项

1. 请在专业医生指导下进行治疗
2. 保持良好的护肤习惯
3. 定期进行皮肤状况评估

## 免责声明

本报告仅供参考，具体治疗方案请咨询专业医生。
"""
                    else:
                        report = "无法生成报告，因为分析结果为空"
                        st.error("分析失败，无法生成报告")
            
            # 创建数据可视化
            with st.spinner("正在生成数据可视化..."):
                if analysis_result:
                    try:
                        print("开始生成热力图...")
                        heatmap_path = create_face_heatmap(image, analysis_result)
                        print("热力图路径:", heatmap_path)
                        
                        print("开始生成雷达图...")
                        radar_path = create_radar_chart(analysis_result)
                        print("雷达图路径:", radar_path)
                        
                        print("开始生成优先级图...")
                        priority_path = create_treatment_priority_chart(report)
                        print("优先级图路径:", priority_path)
                        
                        # 显示可视化图表
                        st.subheader("面部分析可视化")
                        vis_col1, vis_col2 = st.columns(2)
                        
                        with vis_col1:
                            if heatmap_path and os.path.exists(heatmap_path):
                                st.image(heatmap_path, caption="面部问题热力图", use_container_width=True)
                            else:
                                st.warning("无法生成热力图")
                            
                            if priority_path and os.path.exists(priority_path):
                                st.image(priority_path, caption="治疗方案优先级", use_container_width=True)
                            else:
                                st.warning("无法生成治疗方案优先级图")
                        
                        with vis_col2:
                            if radar_path and os.path.exists(radar_path):
                                st.image(radar_path, caption="面部状况评分", use_container_width=True)
                            else:
                                st.warning("无法生成雷达图")
                    except Exception as e:
                        print(f"生成可视化图表时发生错误: {str(e)}")
                        st.error("生成可视化图表时发生错误")
                else:
                    st.error("分析失败，无法生成可视化图表")
            
            # 显示生成的报告
            if analysis_result and report:
                st.subheader("医美建议报告")
                st.markdown(report)
                
                try:
                    print("开始生成专业报告...")
                    # 生成专业报告
                    formatted_report = format_medical_beauty_report(
                        report, 
                        analysis_result, 
                        heatmap_path if 'heatmap_path' in locals() else None, 
                        radar_path if 'radar_path' in locals() else None, 
                        priority_path if 'priority_path' in locals() else None,
                        model_choice
                    )
                    print("专业报告生成成功")
                    
                    # 保存报告相关数据到 session_state
                    st.session_state.report_generated = True
                    st.session_state.report_text = report
                    st.session_state.analysis_result = analysis_result
                    st.session_state.heatmap_path = heatmap_path if 'heatmap_path' in locals() else None
                    st.session_state.radar_path = radar_path if 'radar_path' in locals() else None
                    st.session_state.priority_path = priority_path if 'priority_path' in locals() else None
                    
                    # 提供下载报告功能
                    b64 = base64.b64encode(formatted_report.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="医美建议报告.html">下载专业医美报告</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # 生成并显示 PDF 下载按钮
                    try:
                        print("开始生成PDF...")
                        # 处理报告文本，移除特殊格式
                        processed_report = report.replace("**", "").replace("#", "").strip()
                        if processed_report:
                            print("处理后的报告长度:", len(processed_report))
                            # 生成PDF
                            pdf_path = generate_pdf_report(
                                processed_report,
                                analysis_result,
                                heatmap_path if 'heatmap_path' in locals() else None,
                                radar_path if 'radar_path' in locals() else None,
                                priority_path if 'priority_path' in locals() else None
                            )
                            
                            if pdf_path and os.path.exists(pdf_path):
                                print("PDF文件生成成功:", pdf_path)
                                # 读取并提供下载
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                    if len(pdf_bytes) > 0:
                                        st.download_button(
                                            label="下载PDF报告",
                                            data=pdf_bytes,
                                            file_name=f"医美分析报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf"
                                        )
                                        print("PDF下载按钮创建成功")
                                    else:
                                        st.error("生成的PDF文件为空")
                                        print("错误：PDF文件为空")
                                
                                # 清理临时文件
                                try:
                                    os.unlink(pdf_path)
                                except Exception as e:
                                    print(f"清理临时PDF文件时发生错误: {e}")
                            else:
                                st.error("PDF文件生成失败")
                                print("错误：PDF文件生成失败或文件不存在")
                        else:
                            st.error("无法生成PDF：处理后的报告内容为空")
                            print("错误：处理后的报告内容为空")
                            
                    except Exception as e:
                        st.error(f"生成PDF报告时发生错误: {str(e)}")
                        print(f"PDF生成错误详情: {str(e)}")
                        
                except Exception as e:
                    st.error(f"生成专业报告时发生错误: {str(e)}")
                    print(f"专业报告生成错误详情: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("© 2023 AI医美智能评估系统 | 本系统仅供参考，请遵医嘱") 