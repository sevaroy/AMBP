import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2

class VisualizationService:
    @staticmethod
    def create_face_heatmap(image, analysis_result):
        """創建面部問題熱力圖"""
        if analysis_result is None:
            print("警告：分析結果為空")
            return None
        
        if not isinstance(analysis_result, str):
            print(f"警告：分析結果類型不正確，預期字符串類型，實際為 {type(analysis_result)}")
            try:
                analysis_result = str(analysis_result)
            except:
                return None
        
        os.makedirs("temp", exist_ok=True)
        
        img_array = np.array(image)
        mask = np.zeros_like(img_array[:,:,0]).astype(float)
        
        h, w = mask.shape
        
        # 分析不同區域並設置熱力值
        regions = {
            "額頭": {"keywords": ["皺紋", "額頭"], "coords": [0.1, 0.3, 0.3, 0.7]},
            "眼周": {"keywords": ["眼袋", "黑眼圈", "眼周"], "coords": [0.3, 0.4, 0.25, 0.75]},
            "顴骨": {"keywords": ["色斑", "色素沉著", "顴骨"], "coords": [0.4, 0.5, 0.15, 0.85]},
            "鼻子": {"keywords": ["毛孔", "油性", "鼻子"], "coords": [0.35, 0.5, 0.45, 0.55]},
            "嘴唇": {"keywords": ["唇紋", "嘴唇"], "coords": [0.55, 0.65, 0.4, 0.6]},
            "下巴": {"keywords": ["鬆弛", "下垂", "下巴"], "coords": [0.65, 0.75, 0.4, 0.6]}
        }
        
        for region, data in regions.items():
            if any(keyword in analysis_result for keyword in data["keywords"]):
                severity = 0.6
                if "嚴重" in analysis_result or "明顯" in analysis_result:
                    severity = 0.8
                y1, y2, x1, x2 = data["coords"]
                mask[int(h*y1):int(h*y2), int(w*x1):int(w*x2)] = severity
        
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_array)
        ax.imshow(mask, cmap=cmap)
        ax.axis('off')
        
        plt.tight_layout()
        heatmap_path = "temp/face_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_path

    @staticmethod
    def create_radar_chart(analysis_result):
        """創建面部評分雷達圖"""
        os.makedirs("temp", exist_ok=True)
        
        categories = ['膚質', '皺紋', '色斑', '緊緻度', '毛孔', '膚色均勻度']
        
        # 根據關鍵詞分析評分
        scoring_rules = {
            '膚質': [
                ('乾燥', 0.5),
                ('油性', 1.0),
                ('敏感', 1.5)
            ],
            '皺紋': [
                ('皺', 1.0),
                ('細紋', 1.0),
                ('深度皺紋', 1.5)
            ],
            '色斑': [
                ('色斑', 1.0),
                ('黑斑', 1.0),
                ('色素沉著', 1.5)
            ],
            '緊緻度': [
                ('鬆弛', 1.0),
                ('下垂', 1.0),
                ('輪廓不清', 1.5)
            ],
            '毛孔': [
                ('毛孔', 1.0),
                ('毛孔粗大', 1.0),
                ('毛孔擴張', 1.5)
            ],
            '膚色均勻度': [
                ('不均勻', 1.0),
                ('暗沉', 1.0),
                ('泛紅', 1.5)
            ]
        }
        
        scores = []
        for category, rules in scoring_rules.items():
            base_score = 5.0
            for keyword, deduction in rules:
                if keyword in analysis_result:
                    base_score -= deduction
            scores.append(max(0, min(5, base_score)))
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        scores_closed = scores + [scores[0]]
        angles_closed = angles + [angles[0]]
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        ax.plot(angles_closed, scores_closed, 'o-', linewidth=2, color='#FF5757')
        ax.fill(angles_closed, scores_closed, alpha=0.25, color='#FF5757')
        
        ax.set_thetagrids(np.degrees(angles), categories)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.grid(True)
        
        plt.title('面部狀況評分', size=15, y=1.1)
        
        plt.tight_layout()
        radar_path = "temp/radar_chart.png"
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return radar_path

    @staticmethod
    def create_treatment_priority_chart(report):
        """創建治療方案優先級條形圖"""
        if not report:
            print("警告：報告內容為空")
            return None
            
        os.makedirs("temp", exist_ok=True)
        treatments = []
        priorities = []
        
        try:
            lines = report.split('\n')
            in_treatment_section = False
            
            for line in lines:
                if "推薦的醫美治療方案" in line or "推薦治療方案" in line:
                    in_treatment_section = True
                    continue
                
                if in_treatment_section and ("術後護理" in line or "預期效果" in line or "風險提示" in line):
                    in_treatment_section = False
                    break
                    
                if in_treatment_section and line.strip() and any(char.isdigit() for char in line[:5]):
                    try:
                        priority = int(''.join(filter(str.isdigit, line.split('.')[0])))
                        treatment_name = line.split('.')[1].split('：')[0].strip() if '：' in line.split('.')[1] else line.split('.')[1].strip()
                        treatments.append(treatment_name)
                        priorities.append(6 - priority)
                    except:
                        continue
            
            if not treatments:
                treatments = ["玻尿酸填充", "肉毒素注射", "激光祛疤", "水光針", "線雕提升"]
                priorities = [5, 4, 3, 2, 1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sorted_indices = np.argsort(priorities)[::-1]
            sorted_treatments = [treatments[i] for i in sorted_indices]
            sorted_priorities = [priorities[i] for i in sorted_indices]
            
            bars = ax.barh(sorted_treatments, sorted_priorities, color='#5DA5DA')
            
            for i, v in enumerate(sorted_priorities):
                ax.text(v + 0.1, i, str(v), va='center')
            
            ax.set_xlabel('優先級評分')
            ax.set_ylabel('治療方案')
            ax.set_title('推薦治療方案優先級')
            
            plt.tight_layout()
            priority_path = "temp/treatment_priority.png"
            plt.savefig(priority_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return priority_path
                
        except Exception as e:
            print(f"創建治療方案優先級圖表時出錯: {str(e)}")
            return None 