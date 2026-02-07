import cv2
import time
from PIL import Image
from agri_agent_qwen import AgriAgentQwen
from ui_utils import SmartHUD

# --- 🚀 用户配置区 ---
# 切换模型只需修改这里。例如：
# 3B 路径: '../models/Qwen2.5-VL-3B-Instruct'
# 7B 路径: '../models/Qwen2.5-VL-7B-Instruct'
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct"

VIDEO_PATH = '../assets/demo_video.mp4'
OUTPUT_PATH = 'output_qwen_dashboard.mp4'
THINK_INTERVAL = 15 # Qwen 3B 速度很快，我们可以把间隔设小一点，更丝滑

# --- 知识库 ---
KNOWLEDGE_BASE = {
    "Healthy": "作物生长状况良好，保持巡航。",
    "Disease": "疑似病害，建议停车进行精细扫描。",
    "Pest":    "发现虫害，建议标记位置并喷洒。",
    "Unknown": "环境复杂，请人工接管。"
}

def main():
    print("🤖 Agri-Agent-Qwen System Booting...")
    
    # 1. 加载模型
    agent = AgriAgentQwen(model_path=MODEL_PATH)
    try:
        agent.load_model()
    except:
        print("❌ 无法加载模型，请先运行 download_qwen.py 下载模型！")
        return

    # 2. 初始化 UI
    hud = SmartHUD(font_path='../assets/SimHei.ttf')
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    system_data = {
        'frame_id': 0, 'state': '初始化', 'diagnosis': '等待数据...',
        'advice': '...', 'cmd': '待命', 'latency': 0
    }

    print(f"🚀 开始巡航推理 (Model: {MODEL_PATH})")

    while True:
        ret, frame = cap.read()
        if not ret: break
        system_data['frame_id'] += 1
        
        # === 推理核心 ===
        if system_data['frame_id'] % THINK_INTERVAL == 0:
            system_data['state'] = 'Qwen 思考中...'
            
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.time()
            result = agent.predict(img_pil) # 比如输出 "Healthy"
            t1 = time.time()
            
            # 简单的结果清洗
            key = "Unknown"
            if "Healthy" in result: key = "Healthy"
            elif "Disease" in result: key = "Disease"
            elif "Pest" in result: key = "Pest"
            
            system_data['latency'] = (t1 - t0) * 1000
            
            cn_map = {"Healthy": "健康", "Disease": "病害", "Pest": "虫害", "Unknown": "未知"}
            system_data['diagnosis'] = cn_map.get(key, "未知")
            system_data['advice'] = KNOWLEDGE_BASE.get(key, "注意观察")
            
            if key == "Healthy": system_data['cmd'] = "全速巡航"
            elif key in ["Disease", "Pest"]: system_data['cmd'] = "停车/处理"
            else: system_data['cmd'] = "减速慢行"
            
            print(f"Frame {system_data['frame_id']}: {key} ({system_data['latency']:.1f}ms)")
            
        else:
            system_data['state'] = '巡航中'

        frame = hud.render_panel(frame, system_data)
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 处理完成: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()