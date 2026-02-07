import os
from modelscope import snapshot_download

# === Qwen 家族全家桶 ===
MODEL_REGISTRY = {
    "1": {
        "name": "Qwen2.5-VL-3B-Instruct (极速/显存<8G)",
        "id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "folder": "Qwen2.5-VL-3B-Instruct"
    },
    "2": {
        "name": "Qwen2.5-VL-7B-Instruct (均衡/显存<16G)",
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "folder": "Qwen2.5-VL-7B-Instruct"
    },
    "3": {
        "name": "Qwen2.5-VL-7B-Instruct-AWQ (7B量化版/显存<10G)",
        "id": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "folder": "Qwen2.5-VL-7B-Instruct-AWQ"
    }
}

SAVE_ROOT = '/root/autodl-tmp/models' # 存在上级目录的 models 文件夹里

def main():
    print("="*50)
    print("🚜 Agri-Agent-Qwen 模型下载器")
    print("="*50)
    
    for k, v in MODEL_REGISTRY.items():
        print(f"[{k}] {v['name']}")
    
    choice = input("\n请选择要下载的模型 (推荐 1 或 2): ").strip()
    
    if choice not in MODEL_REGISTRY:
        print("❌ 选项无效")
        return

    info = MODEL_REGISTRY[choice]
    print(f"\n🚀 开始下载: {info['name']} ...")
    
    try:
        path = snapshot_download(
            model_id=info['id'],
            cache_dir=SAVE_ROOT,
            revision='master'
        )
        # ModelScope 下载后的路径可能包含 hash，我们这里只提示成功
        print(f"\n✅ 下载成功！")
        print(f"请将 main_simulation.py 中的 MODEL_PATH 修改为:\n{path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")

if __name__ == "__main__":
    main()