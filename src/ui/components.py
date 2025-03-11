import streamlit as st
import base64

class UIComponents:
    @staticmethod
    def show_header():
        """顯示頁面標題和介紹"""
        st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>AI醫美智能評估系統 - 專業版</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;">
            <h3 style="color: #2c5282; margin-top: 0;">💎 專業面部分析</h3>
            <p>上傳您的正面照片，AI將分析您的面部特徵，提供專業醫美建議和個性化治療方案。</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_sidebar(debug_mode, wkhtmltopdf_installed, api_debug_info=None):
        """顯示側邊欄"""
        with st.sidebar:
            st.image("https://img.icons8.com/bubbles/100/null/spa-flower.png", width=80)
            st.markdown("<h2 style='text-align: center; color: #2c5282; margin-bottom: 1.5rem;'>系統設置</h2>", unsafe_allow_html=True)
            
            debug_mode = st.checkbox("開啟調試模式", value=debug_mode, help="開啟後將顯示更多技術信息，有助於排查問題")
            
            st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="color: #2c5282; margin-top: 0;">選擇分析模型</h4>
            </div>
            """, unsafe_allow_html=True)
            
            model_choice = st.radio(
                "選擇分析模型",
                ["GPT-4o", "DeepSeek VL2"],
                index=0,
                key="model_choice_radio",
                label_visibility="collapsed"
            )
            
            st.markdown("<p style='font-size: 0.8rem; color: #718096;'>選擇不同的模型可能會影響分析結果的準確性和深度</p>", unsafe_allow_html=True)
            
            # 顯示系統狀態
            st.markdown("---")
            st.markdown("<h3 style='color: #2c5282;'>系統狀態</h3>", unsafe_allow_html=True)
            
            if wkhtmltopdf_installed:
                st.markdown("""
                <div style="background-color: #e6ffed; padding: 0.8rem; border-radius: 10px; border-left: 4px solid #48bb78; margin-bottom: 1rem;">
                    <p style="margin: 0; color: #2f855a; font-weight: 500;">✅ PDF導出功能已啟用</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fff5f5; padding: 0.8rem; border-radius: 10px; border-left: 4px solid #f56565; margin-bottom: 1rem;">
                    <p style="margin: 0; color: #c53030; font-weight: 500;">⚠️ PDF導出功能未啟用</p>
                    <p style="margin-top: 0.5rem; font-size: 0.8rem; color: #718096;">請安裝wkhtmltopdf</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 如果在調試模式下，顯示更多信息
            if debug_mode and api_debug_info:
                with st.expander("API 配置詳情"):
                    st.json(api_debug_info)
            
            return debug_mode, model_choice

    @staticmethod
    def show_upload_section():
        """顯示文件上傳區域"""
        upload_placeholder = st.empty()
        with upload_placeholder.container():
            st.markdown("""
            <div style="background-color: #e6f2ff; padding: 1.5rem; border-radius: 10px; border: 1px dashed #4e97d1; text-align: center; margin-bottom: 1rem;">
                <p style="color: #2c5282;">請上傳一張清晰的正面照片以獲得最佳分析效果</p>
                <p style="font-size: 0.8rem; color: #718096;">支持的格式：JPG、JPEG、PNG</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("上傳面部照片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            return upload_placeholder, uploaded_file

    @staticmethod
    def show_analysis_results(analysis_result, report, heatmap_path, radar_path, priority_path):
        """顯示分析結果"""
        tab1, tab2, tab3 = st.tabs(["分析結果", "視覺化圖表", "醫美建議報告"])
        
        with tab1:
            st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;">
                <h3 style="color: #2c5282;">面部分析結果</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if "API連接失敗" in analysis_result:
                st.warning("⚠️ 以下是模擬數據，因為API連接失敗")
            
            st.write(analysis_result)
        
        with tab2:
            st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;">
                <h3 style="color: #2c5282;">面部分析可視化</h3>
            </div>
            """, unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if heatmap_path:
                    st.image(heatmap_path, caption="面部問題熱力圖", use_column_width=True)
                else:
                    st.warning("無法生成熱力圖")
                
                if priority_path:
                    st.image(priority_path, caption="治療方案優先級", use_column_width=True)
                else:
                    st.warning("無法生成治療方案優先級圖")
            
            with viz_col2:
                if radar_path:
                    st.image(radar_path, caption="面部狀況評分", use_column_width=True)
                else:
                    st.warning("無法生成雷達圖")
        
        with tab3:
            if report:
                st.markdown("""
                <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;">
                    <h3 style="color: #2c5282;">醫美建議報告</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(report)

    @staticmethod
    def show_download_options(formatted_report, pdf_content):
        """顯示下載選項"""
        st.markdown("""
        <div style="background-color: #e6f2ff; padding: 1rem; border-radius: 10px; margin: 1.5rem 0;">
            <h4 style="color: #2c5282; margin-top: 0;">下載報告</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_html, col_pdf = st.columns(2)
        
        with col_html:
            b64_html = base64.b64encode(formatted_report.encode()).decode()
            st.markdown(f"""
            <div style="text-align: center;">
                <a href="data:text/html;base64,{b64_html}" download="醫美建議報告.html" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4e97d1; color: white; text-decoration: none; border-radius: 5px; font-weight: 500;">
                    <i class="fas fa-file-code"></i> 下載HTML格式報告
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        with col_pdf:
            if pdf_content:
                b64_pdf = base64.b64encode(pdf_content).decode()
                st.markdown(f"""
                <div style="text-align: center;">
                    <a href="data:application/pdf;base64,{b64_pdf}" download="醫美建議報告.pdf" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4e97d1; color: white; text-decoration: none; border-radius: 5px; font-weight: 500;">
                        <i class="fas fa-file-pdf"></i> 下載PDF格式報告
                    </a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("PDF生成失敗，請嘗試下載HTML格式報告")

    @staticmethod
    def show_footer():
        """顯示頁腳"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-top: 2rem; color: #718096; font-size: 0.9rem;">
            <p>© 2023 AI醫美智能評估系統 | 本系統僅供參考，請遵醫囑</p>
            <p style="font-size: 0.8rem;">使用先進的AI技術提供醫美評估和建議</p>
        </div>
        """, unsafe_allow_html=True) 