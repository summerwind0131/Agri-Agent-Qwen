import torch
from PIL import Image,ImageDraw
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class AgriAgentQwen:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None 

    def load_model(self):
        print(f"🔄 Loading Qwen2.5-VL from {self.model_path} ...")
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype="auto", 
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("✅ AgriAgent-Qwen Brain Online!")
            
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
            raise e

    def predict(self, image_input, prompt_text=""):
        if self.model is None: return "Error: Model not loaded"
        
        # 默认提示词：针对 Qwen 优化
        if not prompt_text:
            prompt_text = "Analyze the crop health in the image. Determine if it is Healthy, Disease, or Pest. Provide ONLY the category name."

        try:
            if image_input.mode != "RGB": 
                image_input = image_input.convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # 准备推理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # 生成
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=64  # 诊断不需要太长，64够了，速度更快
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]

        except Exception as e:
            print(f"Inference Error: {e}")
            return "Unknown"
if __name__ == "__main__":
    # 请修改为你服务器上的真实路径
    # 3B版本
    MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct"
    
    # 7B版本
    #MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-VL-7B-Instruct"
    agent = AgriAgentQwen(model_path=MODEL_PATH)
    agent.load_model()

    # --- 使用更大的图片尺寸 ---
    # Qwen2-VL 对极小分辨率(224x224)支持不佳，建议至少 512x512
    # 纯色图片有时会被模型当做 padding，这里画个圆让它看着像个东西
    print("🎨 Creating test image...")
    test_img = Image.new('RGB', (640, 640), color='red')
    draw = ImageDraw.Draw(test_img)
    draw.ellipse((200, 200, 440, 440), fill = 'blue', outline ='white')
    
    print("🧠 Testing prediction...")
    result = agent.predict(test_img, "Describe this image in detail. What colors do you see?")
    print(f"Result: {result}")
    
    # 创建一个纯色图片测试一下通路
    #test_img = Image.new('RGB', (224, 224), color='red')
    #print("Testing prediction...")
    #result = agent.predict(test_img, "图里有什么颜色？")
    #print(f"Result: {result}")