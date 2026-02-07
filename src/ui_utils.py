import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SmartHUD:
    def __init__(self, font_path='SimHei.ttf', fontsize=30):
        self.font_path = font_path
        self.fontsize = fontsize
        try:
            self.font_big = ImageFont.truetype(font_path, 40) # 大标题
            self.font_std = ImageFont.truetype(font_path, 24) # 正文
            self.font_small = ImageFont.truetype(font_path, 18) # 脚注
        except:
            print("❌ 警告：找不到字体文件，将使用默认字体（不支持中文）")
            self.font_big = ImageFont.load_default()
            self.font_std = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def draw_chinese(self, img_cv2, text, pos, color=(255, 255, 255), font_size='std'):
        """
        核心函数：将 CV2 图片转为 PIL -> 画中文 -> 转回 CV2
        """
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        font = self.font_std
        if font_size == 'big': font = self.font_big
        if font_size == 'small': font = self.font_small

        # 画文字阴影 (黑色描边效果)，让字在任何背景下都看清
        shadow_color = (0, 0, 0)
        x, y = pos
        draw.text((x+1, y+1), text, font=font, fill=shadow_color)
        draw.text((x, y), text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def render_panel(self, frame, data):
        """
        绘制完整的仪表盘
        data: 包含 status, diagnosis, cmd, latency 等信息的字典
        """
        h, w, _ = frame.shape
        
        # 1. 绘制顶部半透明黑条 (状态栏)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # 2. 绘制底部半透明黑条 (信息栏)
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        
        # 混合图层 (透明度 0.6)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # --- 顶部信息 ---
        # 左上角：系统状态
        status_color = (0, 255, 0) if data['state'] == '巡航中 (Scanning)' else (0, 255, 255)
        frame = self.draw_chinese(frame, f"系统状态: {data['state']}", (20, 15), status_color, 'std')
        
        # 右上角：帧率/延迟
        frame = self.draw_chinese(frame, f"延迟: {data['latency']:.0f}ms | 帧: {data['frame_id']}", (w-250, 20), (200, 200, 200), 'small')

        # --- 底部信息 ---
        # 诊断结果 (左下)
        diag_text = f"诊断结果: {data['diagnosis']}"
        diag_color = (255, 255, 255)
        if "病" in data['diagnosis'] or "Disease" in data['diagnosis']:
            diag_color = (255, 50, 50) # 红色预警
        elif "健康" in data['diagnosis'] or "Healthy" in data['diagnosis']:
            diag_color = (50, 255, 50) # 绿色
            
        frame = self.draw_chinese(frame, diag_text, (20, h-100), diag_color, 'big')

        # 专家建议 (左下，小字)
        frame = self.draw_chinese(frame, f"专家建议: {data['advice']}", (20, h-50), (200, 200, 200), 'std')

        # 动作指令 (右下，大字)
        cmd_color = (0, 255, 255) # 黄色默认
        if "加速" in data['cmd']: cmd_color = (0, 255, 0)
        if "停止" in data['cmd']: cmd_color = (0, 0, 255)
        
        cmd_text = data['cmd']
        # 简单的计算让文字靠右
        text_w = len(cmd_text) * 20 
        frame = self.draw_chinese(frame, cmd_text, (w - text_w - 150, h-80), cmd_color, 'big')

        return frame