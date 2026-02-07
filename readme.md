# Agri-Agent-Qwen: On-Board Intelligent Agricultural Inspection System

# Agri-Agent-Qwen：基于 Qwen2.5-VL 的高性能车载农业巡检系统

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL-violet)](https://github.com/QwenLM/Qwen2.5-VL)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Gradio-orange)](https://pytorch.org/)

> 🎓 **Project Origin**: Nankai University | Intelligent Science & Technology
> 🚜 **Core Tech**: Qwen2.5-VL (3B/7B) + Dynamic Model Switching + Edge Computing
> 👤 **Author**: Fengyuan

## 📖 Introduction (项目简介)

**Agri-Agent-Qwen** is a specialized branch of the Agri-Agent system, optimized for the **Qwen2.5-VL** family of models. It is designed for agricultural autonomous vehicles to perform real-time crop health monitoring (Healthy/Disease/Pest).

本项目是 Agri-Agent 的 **Qwen 专属高性能版本**。相比于通用版，本项目针对 Qwen2.5-VL 进行了深度适配，利用其强大的高分辨率视觉推理能力，实现了更精准的病虫害诊断。

**关键突破 (Key Breakthrough)**：
系统支持在 Web 指挥中心**动态切换模型 (Dynamic Switching)**。用户可以根据算力需求，在 **3B (极速版)** 和 **7B (高精版)** 之间无缝切换，无需重启系统。

## ✨ Key Features (核心功能)

- **🚀 Multi-Scale Model Support (多参数模型支持)**
  - **3B Model**: Ultra-fast inference (<200ms), ideal for real-time edge computing on restricted hardware.
  - **7B Model**: High-precision diagnosis for complex disease identification.
  - _支持在仪表盘中通过下拉菜单实时切换模型，系统自动管理显存释放。_

- **🖥️ Interactive Command Center (交互式指挥中心)**
  - A Gradio-based dashboard that visualizes real-time video feeds, AI reasoning logs, and simulated telemetry data (Speed, Battery, GPS).
  - _基于 Gradio 构建的界面，集成视频流回传与传感器数据监控。_

- **🧠 Qwen2.5-VL Native (原生适配)**
  - Utilizes `qwen-vl-utils` for native resolution processing, ensuring no detail loss in crop images.
  - _采用 Qwen 原生视觉处理流程，支持动态分辨率输入，大幅提升细微病斑的检测率。_

## 📂 File Structure (目录结构)

```text
Agri-Agent-Qwen/
├── assets/                 # [Resources] Demo video & Fonts
├── models/                 # [Model Weights] Downloaded models (Git ignored)
├── src/                    # [Source Code]
│   ├── agri_agent_qwen.py  # Core inference engine for Qwen2.5-VL
│   ├── dashboard_qwen.py   # Web UI entry point (Gradio)
│   ├── download_qwen.py    # Model downloader script
│   ├── simulation_qwen.py  # Local simulation script (No UI)
│   └── ui_utils.py         # HUD rendering utilities
├── requirements.txt        # Dependency list
└── README.md               # Documentation
```

## 🚀 Quick Start (快速开始)

### 1. Environment Setup (环境配置)

It is highly recommended to use Python 3.10 to ensure compatibility with Qwen2.5-VL and Flash Attention.

```Bash
# Create Conda environment
conda create -n agriagent python=3.10 -y
conda activate agriagent

# Clone the repository
git clone [https://github.com/summerwind0131/Agri-Agent-Qwen.git](https://github.com/summerwind0131/Agri-Agent-Qwen.git)
cd Agri-Agent-Qwen

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download (模型下载)

Use the built-in script to download models from ModelScope (optimized for CN network).

```Bash
python src/download_qwen.py
```

Follow the prompt to select model 1 (3B) or 2 (7B).

### 3. Run Command Center (启动指挥中心)

Start the Web UI. This allows you to switch between 3B and 7B models dynamically.

```Bash
python src/dashboard_qwen.py
```

Access: Open your browser and go to http://localhost:6006

Usage:

Select a model from the Dropdown Menu (e.g., Qwen2.5-VL-3B).

Click "1. 加载/切换模型" and wait for the "System Ready" message.

Click "2. 开始巡航" to start the simulation.

## ⚙️ Configuration (高级配置)

If your model download path is different, please update the MODEL_MAP in src/dashboard_qwen.py:

```Python
# src/dashboard_qwen.py
MODEL_MAP = {
    "Qwen2.5-VL-3B": "/path/to/your/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B": "/path/to/your/Qwen2.5-VL-7B-Instruct",
}
```

## 📝 Future Work

[ ] Integrate Flash Attention 2 for 30% faster inference.

[ ] Add support for Qwen2.5-VL-72B (API mode) for cloud-based expert diagnosis.

[ ] Deploy on NVIDIA Jetson Orin edge devices.

---

Developed by Fengyuan @ Nankai University.

```

```
