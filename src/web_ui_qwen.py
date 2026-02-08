import gradio as gr
import cv2
import time
import pandas as pd
import threading
import torch
import gc
from PIL import Image
from agri_agent_qwen import AgriAgentQwen  # 假设你使用的是 Qwen 的 Agent 类
from ui_utils import SmartHUD

# --- 1. 模型配置 (字典映射) ---
# 这里的 Key 是显示给用户看的名字，Value 是服务器上的真实路径
MODEL_MAP = {
    "Qwen2.5-VL-3B (极速版 - 推荐)": "/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct",
    "Qwen2.5-VL-7B (高精版 - 需16G显存)": "/root/autodl-tmp/models/Qwen/Qwen2___5-VL-7B-Instruct",
}

VIDEO_PATH = 'demo_video.mp4'
THINK_INTERVAL = 15

# --- 全局变量 ---
agent = None
hud = None
is_running = False

mock_sensors = {
    "battery": 98,
    "speed": 0.0,
    "lat": 34.0522,
    "lon": 118.2437
}

data_lock = threading.Lock()
latest_inference_result = {
    "diagnosis": "等待数据...",
    "cmd": "待命"
}

def load_system(model_selection):
    """
    修改点：增加 model_selection 参数
    """
    global agent, hud
    
    # 获取真实路径
    target_path = MODEL_MAP.get(model_selection)
    if not target_path:
        return "❌ 错误：请先选择一个模型！"
    
    if is_running:
        return "⚠️ 安全警告：巡航模式下禁止切换内核！请先点击[紧急停止]。"

    print(f"🔄 正在切换/加载模型: {model_selection}...")
    
    # --- 关键：切换模型前清理显存 ---
    if agent is not None:
        del agent
        agent = None
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 旧模型显存已清理")

    try:
        # 初始化新的 Agent
        # 注意：这里假设你的 AgriAgentQwen 类兼容这些路径
        agent = AgriAgentQwen(model_path=target_path)
        agent.load_model() 
        
        # 初始化 HUD (如果还没初始化)
        if hud is None:
            hud = SmartHUD(font_path='SimHei.ttf')
            
        return f"✅ 系统就绪: 当前内核 {model_selection}"
        
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}\n(请检查路径或显存是否足够)"

# --- 后台 AI 线程 (逻辑不变) ---
def run_ai_background(img_pil):
    global latest_inference_result, agent, mock_sensors
    if agent is None: return

    try:
        raw_result = agent.predict(img_pil)
        # 简单的清洗逻辑
        key = "Unknown"
        if "Healthy" in raw_result: key = "Healthy"
        elif "Disease" in raw_result: key = "Disease"
        elif "Pest" in raw_result: key = "Pest"
        
        cn_map = {"Healthy": "健康", "Disease": "病害", "Pest": "虫害", "Unknown": "未知"}
        diagnosis = cn_map.get(key, key)
        
        if key == "Healthy":
            cmd = "全速巡航"
            target_speed = 1.5 
        elif key in ["Disease", "Pest"]:
            cmd = "停车/喷洒"
            target_speed = 0.0
        else:
            cmd = "减速观察"
            target_speed = 0.5
            
        with data_lock:
            latest_inference_result["diagnosis"] = diagnosis
            latest_inference_result["cmd"] = cmd
            mock_sensors['speed'] = target_speed
    except Exception as e:
        print(f"Inference Error: {e}")

# --- 视频处理循环 (逻辑不变) ---
def processing_loop():
    global is_running, mock_sensors, latest_inference_result
    
    if agent is None:
        yield None, "🚫 传感器离线", pd.DataFrame(), "⚠️ 请先在左侧选择模型并点击初始化"
        return

    # 尝试打开视频 (兼容路径)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        # 备用路径尝试
        cap = cv2.VideoCapture('../assets/demo_video.mp4')
    
    if not cap.isOpened():
        yield None, "❌ 错误：找不到视频文件", pd.DataFrame(), "🔴 启动失败：请检查 demo_video.mp4 是否存在"
        return

    frame_count = 0
    logs = []
    
    try:
        while is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_count += 1
            
            # 异步推理
            if frame_count % THINK_INTERVAL == 0 :
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                threading.Thread(target=run_ai_background, args=(img_pil,), daemon=True).start()
            
            # 读取状态
            with data_lock:
                current_diagnosis = latest_inference_result["diagnosis"]
                current_cmd = latest_inference_result["cmd"]
                
                if frame_count % THINK_INTERVAL == 0 :
                     timestamp = time.strftime("%H:%M:%S")
                     logs.insert(0, [timestamp, frame_count, current_diagnosis, current_cmd])
                     if len(logs) > 10: logs.pop()

            # 模拟传感器
            mock_sensors['battery'] -= 0.01 
            if mock_sensors['battery'] < 0: mock_sensors['battery'] = 100
            
            # HUD
            hud_data = {
                'frame_id': frame_count,
                'state': 'AI 监测中', 
                'diagnosis': current_diagnosis,
                'advice': f"目标速度: {mock_sensors['speed']} m/s",
                'cmd': current_cmd,
                'latency': 0
            }
            frame = hud.render_panel(frame, hud_data)
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            status_text = (
                f"🔋 电池: {int(mock_sensors['battery'])}%\n"
                f"🚀 速度: {mock_sensors['speed']} m/s\n"
                f"📍 坐标: ({mock_sensors['lat']}, {mock_sensors['lon']})"
            )
            
            log_df = pd.DataFrame(logs, columns=["时间", "帧号", "识别结果", "执行指令"])
            
            yield out_frame, status_text, log_df, "🟢 正在巡航"
            time.sleep(0.03) # 控制网页端帧率

    finally:
        cap.release()
        yield None, "已停止", pd.DataFrame(), "🔴 任务结束"

def start_patrol():
    global is_running
    is_running = True
    for output in processing_loop():
        yield output

def stop_patrol():
    global is_running
    is_running = False
    return "🔴 停止指令已发送"

# --- 3. Gradio 界面 (新增 Dropdown) ---
with gr.Blocks(title="AgriAgent Qwen 指挥中心", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚜 AgriAgent 智能农业机器人 (多模型版)")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_display = gr.Image(label="实时回传画面", type="numpy")
        
        with gr.Column(scale=1):
            # === 新增：模型选择区域 ===
            gr.Markdown("### ⚙️ 系统设置")
            model_selector = gr.Dropdown(
                choices=list(MODEL_MAP.keys()), 
                value=list(MODEL_MAP.keys())[0], # 默认选第一个
                label="选择 AI 内核 (Model Selection)",
                interactive=True
            )
            
            init_btn = gr.Button("1. 加载/切换模型", variant="primary")
            system_status = gr.Textbox(label="系统日志", value="等待初始化...", interactive=False)
            
            gr.Markdown("### 📡 遥测控制")
            sensor_display = gr.Textbox(label="传感器数据", lines=3, interactive=False)
            
            with gr.Row():
                start_btn = gr.Button("2. 开始巡航", variant="secondary")
                stop_btn = gr.Button("3. 紧急停止", variant="stop")

    gr.Markdown("### 📋 AI 诊断日志")
    log_table = gr.Dataframe(headers=["时间", "帧号", "识别结果", "执行指令"], interactive=False)

    # 事件绑定
    # 注意：init_btn 现在需要把 model_selector 的值传给 load_system
    init_btn.click(
        load_system, 
        inputs=[model_selector], 
        outputs=[system_status]
    )
    
    start_btn.click(
        start_patrol, 
        inputs=[], 
        outputs=[video_display, sensor_display, log_table, system_status]
    )
    stop_btn.click(stop_patrol, inputs=[], outputs=[system_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)