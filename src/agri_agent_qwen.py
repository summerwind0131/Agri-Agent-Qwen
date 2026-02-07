import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
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
            # 自动适配：如果是 AWQ 量化模型，需要特殊的加载方式
            # 但 Qwen2.5 的 AutoModel 通常能自动处理，这里显式指定类
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto", 
                device_map="auto"
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
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