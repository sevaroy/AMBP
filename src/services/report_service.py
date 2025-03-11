import datetime
import pdfkit
import tempfile
import os
import base64
from ..config import WKHTMLTOPDF_INSTALLED

class ReportService:
    @staticmethod
    def format_medical_beauty_report(report, analysis_result, heatmap_path, radar_path, priority_path, model_choice):
        """格式化為醫美診所專用報告模板"""
        assessment_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 處理圖片路徑
        def get_image_html(path, alt_text):
            if path:
                try:
                    return f'<img src="data:image/png;base64,{base64.b64encode(open(path, "rb").read()).decode()}" alt="{alt_text}">'
                except:
                    return f"<p>{alt_text}生成失敗</p>"
            return f"<p>{alt_text}生成失敗</p>"
        
        heatmap_html = get_image_html(heatmap_path, "面部問題熱力圖")
        radar_html = get_image_html(radar_path, "面部狀況評分")
        priority_html = get_image_html(priority_path, "治療方案優先級")
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>醫美評估報告</title>
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
                h1, h2 {{
                    color: #2c3e50;
                    margin: 0;
                }}
                .patient-info {{
                    display: flex;
                    justify-content: space-between;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
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
                    <h1>AI醫美智能評估報告</h1>
                    <p>生成日期: {assessment_date}</p>
                </div>
                
                <div class="patient-info">
                    <div class="info-column">
                        <div class="info-item">
                            <span class="info-label">評估日期:</span>
                            <span>{assessment_date}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">評估模型:</span>
                            <span>{model_choice}</span>
                        </div>
                    </div>
                    <div class="info-column">
                        <div class="info-item">
                            <span class="info-label">報告編號:</span>
                            <span>AI-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}</span>
                        </div>
                    </div>
                </div>
                
                <h2>面部分析可視化</h2>
                
                <div class="visualizations">
                    <div class="vis-item">
                        {heatmap_html}
                        <div class="vis-caption">面部問題熱力圖</div>
                    </div>
                    <div class="vis-item">
                        {radar_html}
                        <div class="vis-caption">面部狀況評分</div>
                    </div>
                    <div class="vis-item">
                        {priority_html}
                        <div class="vis-caption">治療方案優先級</div>
                    </div>
                </div>
                
                <h2>面部評估結果</h2>
                
                <div class="assessment">
                    {report}
                </div>
                
                <div class="disclaimer">
                    <strong>免責聲明:</strong> 本報告由AI系統生成，僅供參考。在進行任何醫美治療前，請務必諮詢專業醫生的意見。分析結果和治療建議基於AI模型的圖像識別和數據分析，不構成醫療診斷或處方。
                </div>
                
                <div class="footer">
                    © {datetime.datetime.now().year} AI醫美智能評估系統 | 本系統僅供專業醫美機構使用
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_report

    @staticmethod
    def generate_pdf_report(html_content):
        """將HTML報告轉換為PDF格式"""
        if not WKHTMLTOPDF_INSTALLED:
            print("未檢測到wkhtmltopdf，無法生成PDF")
            return None
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                f.write(html_content.encode('utf-8'))
                temp_html_path = f.name
            
            temp_pdf_path = temp_html_path.replace('.html', '.pdf')
            
            options = {
                'encoding': 'UTF-8',
                'page-size': 'A4',
                'margin-top': '1cm',
                'margin-right': '1cm',
                'margin-bottom': '1cm',
                'margin-left': '1cm',
                'enable-local-file-access': None
            }
            
            pdfkit.from_file(temp_html_path, temp_pdf_path, options=options)
            
            with open(temp_pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
            
            os.remove(temp_html_path)
            os.remove(temp_pdf_path)
            
            return pdf_content
        except Exception as e:
            print(f"生成PDF時出錯: {str(e)}")
            return None 