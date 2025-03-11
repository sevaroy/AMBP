# AI医美智能评估系统 - MVP版本

这是一个使用多模态AI模型进行面部分析并生成医美建议报告的应用程序。

## 功能特点

- 支持上传面部照片进行分析
- 可选择使用GPT-4o或DeepSeek VL2进行面部特征分析
- 使用DeepSeek-R1生成专业医美建议报告
- 提供报告下载功能

## 安装与使用

### 环境要求

- Python 3.8+
- 相关API密钥（OpenAI、Replicate、DeepSeek）

### 安装步骤

1. 克隆仓库或下载源代码

2. 安装依赖包
   ```bash
   pip install -r requirements.txt
   ```

3. 配置API密钥
   - 复制`.env.example`文件并重命名为`.env`
   - 在`.env`文件中填入您的API密钥

4. 运行应用
   ```bash
   streamlit run app.py
   ```

5. 在浏览器中访问应用（默认地址：http://localhost:8501）

## 使用说明

1. 在应用界面上传您的面部照片（支持JPG、JPEG、PNG格式）
2. 在侧边栏选择分析模型（GPT-4o或DeepSeek VL2）
3. 点击"开始分析"按钮
4. 等待系统分析面部特征并生成医美建议报告
5. 查看分析结果和建议报告
6. 点击"下载完整报告"保存报告

## 免责声明

本系统生成的医美建议仅供参考，在进行任何医美治疗前，请务必咨询专业医生的意见。

## 技术架构

- 前端界面：Streamlit
- 图像分析：OpenAI GPT-4o / DeepSeek VL2
- 报告生成：DeepSeek-R1
- 数据处理：Python 