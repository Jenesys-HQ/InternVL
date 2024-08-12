import ast
import json
import logging
import os

from transformers import AutoTokenizer, AutoModel
import torch

from img_utils import load_image_bs64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
global_counter = 0


class JackVision:
    def __init__(self, model_path: str = None, **kwargs):
        # Load model
        assert model_path is not None, "Model path cannot be None."
        self.model_path = model_path

        logger.info(f"Loading model from {model_path}")

        # check if model name has 4b as suffix after split "_" if it doesn't add cuda device "auto"
        if "4b" in model_path.split("_"):
            self.model = AutoModel.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16,
                # low_cpu_mem_usage=True,
                trust_remote_code=True).eval().cuda()
            kwargs_default = dict(do_sample=True, max_new_tokens=1024, num_beams=1, top_p=0.7, temperature=0.8)
        else:
            device = 'auto'
            self.device = device
            if device == 'auto':
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='balanced'
            ).eval()
            kwargs_default = dict(do_sample=True, max_new_tokens=1024, num_beams=1, top_p=0.7, temperature=0.8)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True,  use_fast=True)

        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        logger.warning(f'Following kwargs received: {self.kwargs}, will use as generation config.')

    def generate_value(self, prompt: str, image_bs64: str, max_num: int = 6) -> dict:
        self.max_num = max_num
        pixel_values = load_image_bs64(image_bs64, max_num=self.max_num).to(torch.bfloat16).cuda()
        # Run inference
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.kwargs
            )

        return response

    def generate_json(self, prompt: str, image_bs64: str, max_num: int = 6) -> dict:
        self.max_num = max_num
        pixel_values = load_image_bs64(image_bs64, max_num=self.max_num).to(torch.bfloat16).cuda()
        # Run inference
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.kwargs
            )

        if "```" in response and "```json" not in response:
            # Extract just the dictionary-like part
            start = response.find("```") + 3
            end = response.find("\n```", start)

            response = response[start:end].strip()
        elif "```json" in response:
            # Extract just the dictionary-like part
            start = response.find("json") + len("json\n")
            end = response.find("\n```", start)

            response = response[start:end].strip()

        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            logger.info(f"Predicted data: {response}")
            return {}

    def chat(self, prompt: str, image_bs64: str) -> dict:
        self.max_num = 6
        
        # Run inference
        if image_bs64:
            pixel_values = load_image_bs64(image_bs64, max_num=self.max_num).to(torch.bfloat16).cuda()
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=self.kwargs
                )

            if "```" in response and "```json" not in response:
                # Extract just the dictionary-like part
                start = response.find("```") + 3
                end = response.find("\n```", start)

                response = response[start:end].strip()
            elif "```json" in response:
                # Extract just the dictionary-like part
                start = response.find("json") + len("json\n")
                end = response.find("\n```", start)

                response = response[start:end].strip()
            
            try:
                response = json.loads(response)
            except:
                try:
                    response = ast.literal_eval(response)
                except Exception as e:
                    logger.error(f"Failed to parse response: {e}")
        else:
            pixel_values = None
            with torch.no_grad():
                response = self.model.chat(
                    AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=True),
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=self.kwargs
                )
        
        return response


# model_path = "/home/ubuntu/ark-kit/pretrained/ark_lvlm_4b"
# # model_path = "/home/ubuntu/ark-lvlm/pretrained/ark_lvlm_4b"

# model = JackVision(model_path=model_path)
# with open("/home/ubuntu/ark-lvlm/data/test.txt", "r") as f:
#     image_bs64 = f.read() 
# response = model.generate(image_bs64)

# print(response)