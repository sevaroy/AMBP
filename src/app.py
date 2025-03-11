import streamlit as st
from PIL import Image
import base64
from io import BytesIO

from config import PAGE_CONFIG, CUSTOM_THEME
from services.api_service import APIService
from services.visualization_service import VisualizationService
from services.report_service import ReportService
from ui.components import UIComponents

# 初始化會話狀態
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'api_debug_info' not in st.session_state:
    st.session_state.api_debug_info = {}
if 'show_api_error' not in st.session_state:
    st.session_state.show_api_error = False
if 'api_error_message' not in st.session_state:
    st.session_state.api_error_message = ""
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "GPT-4o"

# 配置頁面
st.set_page_config(**PAGE_CONFIG)

# 應用自定義主題
st.markdown(CUSTOM_THEME, unsafe_allow_html=True)

def main():
    # 顯示頁面標題和介紹
    UIComponents.show_header()
    
    # 顯示側邊欄
    debug_mode, model_choice = UIComponents.show_sidebar(
        st.session_state.debug_mode,
        ReportService.WKHTMLTOPDF_INSTALLED,
        st.session_state.api_debug_info if st.session_state.debug_mode else None
    )
    
    # 更新會話狀態
    st.session_state.debug_mode = debug_mode
    st.session_state.model_choice = model_choice
    
    # 顯示文件上傳區域
    upload_placeholder, uploaded_file = UIComponents.show_upload_section()
    
    if uploaded_file is not None:
        # 清空上傳區域
        upload_placeholder.empty()
        
        # 顯示上傳的圖片
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; text-align: center;">
                <h4 style="color: #2c5282; margin-top: 0;">已上傳照片</h4>
            </div>
            """, unsafe_allow_html=True)
            
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # 分析按鈕
            st.markdown("<div style='text-align: center; margin: 1.5rem 0;'>", unsafe_allow_html=True)
            if st.button("開始分析", key="analyze_button", help="點擊開始分析上傳的照片"):
                st.markdown("</div>", unsafe_allow_html=True)
                
                with st.spinner("正在進行面部分析..."):
                    # 將圖片轉換為base64
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # 進行面部分析
                    analysis_result = APIService.analyze_with_deepseek(image_base64)
                    
                    # 生成報告
                    if analysis_result and "分析失敗" not in analysis_result:
                        report = APIService.generate_report(analysis_result)
                    else:
                        report = "無法生成報告，因為分析結果為空"
                        st.error("分析失敗，無法生成報告")
                
                # 創建數據可視化
                with st.spinner("正在生成數據可視化..."):
                    if analysis_result:
                        heatmap_path = VisualizationService.create_face_heatmap(image, analysis_result)
                        radar_path = VisualizationService.create_radar_chart(analysis_result)
                        priority_path = VisualizationService.create_treatment_priority_chart(report)
                        
                        # 顯示分析結果
                        UIComponents.show_analysis_results(
                            analysis_result,
                            report,
                            heatmap_path,
                            radar_path,
                            priority_path
                        )
                        
                        # 生成專業報告
                        formatted_report = ReportService.format_medical_beauty_report(
                            report,
                            analysis_result,
                            heatmap_path,
                            radar_path,
                            priority_path,
                            model_choice
                        )
                        
                        # 生成PDF報告
                        with st.spinner("正在生成PDF報告..."):
                            pdf_content = ReportService.generate_pdf_report(formatted_report)
                        
                        # 顯示下載選項
                        UIComponents.show_download_options(formatted_report, pdf_content)
    
    # 顯示頁腳
    UIComponents.show_footer()

if __name__ == "__main__":
    main() 