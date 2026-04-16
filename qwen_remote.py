"""
Gemma-4 Vision Captioner + Prompt Enhancer for ComfyUI
FINAL FIXED VERSION - thinking_budget is now properly sent to the server
"""

import base64
from io import BytesIO
import re
import gc
import os
from datetime import datetime

import torch
import httpx
from PIL import Image
from openai import OpenAI

import folder_paths  # noqa: F401

# Optional: llama-cpp-python for Local Standalone mode
try:
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_chat_format import Gemma3ChatHandler
        HAS_GEMMA_HANDLER = True
    except ImportError:
        HAS_GEMMA_HANDLER = False
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    HAS_GEMMA_HANDLER = False


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _tensor_to_pil(tensor) -> Image.Image:
    frame = tensor[0] if tensor.ndim == 4 else tensor
    arr = (frame * 255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def _make_client(server_url: str) -> OpenAI:
    base = server_url.rstrip("/") + "/"
    return OpenAI(
        base_url=base,
        api_key="lm-studio",
        http_client=httpx.Client(base_url=base, follow_redirects=True),
    )


# STRICT system prompt
DEFAULT_IMAGE_SYSTEM = (
    "You are a world-class visual analyst and master prompt engineer specializing in photorealistic, cinematic image generation. You have been given {image_count} image(s) labeled {image_labels}. Always reference images strictly by their exact label in your internal thinking ONLY."
)

DEFAULT_TEXT_SYSTEM = DEFAULT_IMAGE_SYSTEM


class VisionLLMCaptioner:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["Remote API (llama-server)", "Local Standalone (llama-cpp-python)"], {"default": "Remote API (llama-server)"}),
                "mode": (["Image Caption", "Text -> Detailed Image Prompt"], {"default": "Image Caption"}),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8080/v1", "multiline": False}),
                "model_name": ("STRING", {"default": "gemma-4-E2B.gguf", "multiline": False}),
                "model_path": ("STRING", {"default": "/media/john/A024FBBA24FB9210/LLAMA_GGUF/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf", "multiline": False}),
                "mmproj_path": ("STRING", {"default": "/media/john/A024FBBA24FB9210/LLAMA_GGUF/mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "multiline": False}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 512}),
                "n_batch": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "attention_mode": (["Flash Attention (recommended)", "Standard Attention"], {"default": "Flash Attention (recommended)"}),
                "system_prompt": ("STRING", {"default": DEFAULT_IMAGE_SYSTEM, "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe the scene in extreme detail.", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 16384, "step": 64}),
                "enable_thinking": ("BOOLEAN", {"default": True, "label_on": "Thinking ON", "label_off": "Thinking OFF (No Think)"}),
                "thinking_budget": ("INT", {"default": 8192, "min": 256, "max": 32768, "step": 128}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.9, "max": 2.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 1.30, "min": -2.0, "max": 2.0, "step": 0.05}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "unload_after_inference": ("BOOLEAN", {"default": True, "label_on": "Unload after run (recommended)", "label_off": "Keep model in memory"}),
                "save_to_file": ("BOOLEAN", {"default": False, "label_on": "Save caption to file", "label_off": "Don't save"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "input_text": ("STRING", {"default": "", "multiline": True, "placeholder": "Short idea → e.g. 'cyberpunk girl with neon hair'"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "cuda_graphs": ("BOOLEAN", {"default": False, "label_on": "CUDA Graphs (faster on 40/50 series)", "label_off": "No CUDA Graphs"}),
                "mlock": ("BOOLEAN", {"default": True, "label_on": "mlock (prevent swapping)", "label_off": "No mlock"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "full_raw_debug", "saved_file_path")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "vision"

    def __init__(self):
        self.llm = None

    def _build_system_prompt(self, raw: str, enable_thinking: bool, image_count: int, image_labels: str) -> str:
        prompt = (raw or "").strip()
        prompt = prompt.replace("{image_count}", str(image_count))
        prompt = prompt.replace("{image_labels}", image_labels)
        if enable_thinking:
            prompt = "<|think|>\n" + prompt
        return prompt

    @staticmethod
    def _extract_caption(content: str) -> str:
        if not content or not content.strip():
            return "[EMPTY OR THINKING ONLY — check debug output]"

        content = re.sub(r"<\|channel\>thought.*?<channel\|>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<\|?channel\|?>", "", content, flags=re.IGNORECASE)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"\*\*Image Generation Prompt:\*\*.*", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"Image Generation Prompt:.*", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"Thinking Process:.*", "", content, flags=re.DOTALL | re.IGNORECASE)
        return content.strip()

    def _collect_images(self, image_1, kwargs: dict) -> dict:
        images = {}
        if image_1 is not None:
            images["image_1"] = image_1
        for k, v in kwargs.items():
            if k.startswith("image_") and v is not None:
                images[k] = v
        return dict(sorted(images.items(), key=lambda x: int(x[0].split("_")[1])))

    def _unload_model(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[VisionLLMCaptioner] Model unloaded from VRAM/RAM")

    def _save_caption(self, caption: str) -> str:
        output_dir = os.path.join(folder_paths.get_output_directory(), "captions")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"caption_{timestamp}.txt"
        full_path = os.path.join(output_dir, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"[VisionLLMCaptioner] Caption saved → {full_path}")
        return full_path

    def generate(self, backend, mode, server_url, model_name, model_path, mmproj_path,
                 n_gpu_layers, n_ctx, n_batch, attention_mode,
                 system_prompt, user_prompt, max_tokens, enable_thinking, thinking_budget,
                 temperature, top_p, top_k, repeat_penalty, presence_penalty, min_p,
                 unload_after_inference, save_to_file,
                 image_1=None, input_text="", seed=None,
                 cuda_graphs=False, mlock=True, **kwargs):

        print(f"[DEBUG] generate() STARTED - Backend: {backend} | Mode: {mode} | Thinking: {enable_thinking} | Budget: {thinking_budget}")

        max_tokens_sent = max_tokens + (thinking_budget if enable_thinking else 0)
        reasoning_budget = thinking_budget if enable_thinking else 0
        image_labels_str = "(text-only mode)"

        if backend == "Remote API (llama-server)":
            client = _make_client(server_url)

            if mode == "Image Caption":
                images = self._collect_images(image_1, kwargs)
                print(f"[DEBUG] Remote - Collected {len(images)} images")
                if not images:
                    raise ValueError("Connect at least one image in 'Image Caption' mode.")
                sorted_labels = list(images.keys())
                image_labels_str = ", ".join(sorted_labels)

                content_parts = []
                for label in sorted_labels:
                    pil_img = _tensor_to_pil(images[label])
                    b64 = _pil_to_b64(pil_img)
                    content_parts.append({"type": "text", "text": f"[{label}]"})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                content_parts.append({"type": "text", "text": user_prompt})

                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt or DEFAULT_IMAGE_SYSTEM, enable_thinking, len(images), image_labels_str)},
                    {"role": "user", "content": content_parts}
                ]
            else:
                if not (input_text or "").strip():
                    raise ValueError("Enter text in the 'input_text' field.")
                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt or DEFAULT_TEXT_SYSTEM, enable_thinking, 0, image_labels_str)},
                    {"role": "user", "content": f"{input_text}\n\n{user_prompt}".strip()}
                ]

            if not enable_thinking:
                for msg in messages:
                    if msg["role"] == "user":
                        if isinstance(msg["content"], list):
                            last = next((p for p in reversed(msg["content"]) if p.get("type") == "text"), None)
                            if last:
                                last["text"] = last.get("text", "").rstrip() + " /no_think"
                        elif isinstance(msg["content"], str):
                            msg["content"] = msg["content"].rstrip() + " /no_think"

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_sent,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=0.0,
                seed=seed if seed != 0 else None,
                stream=False,
                extra_body={
                    "top_k": top_k,
                    "min_p": min_p,
                    "repeat_penalty": repeat_penalty,
                    "reasoning_budget": reasoning_budget,
                },
            )

        else:  # Local Standalone (llama-cpp-python)
            print("[DEBUG] Using Local Standalone mode")
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python is not installed.")

            if self.llm is None:
                print(f"[DEBUG] Loading Gemma-4 model: {model_path}")
                flash = (attention_mode == "Flash Attention (recommended)")
                if HAS_GEMMA_HANDLER:
                    chat_handler = Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=False)
                    self.llm = Llama(
                        model_path=model_path,
                        chat_handler=chat_handler,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        flash_attn=flash,
                        cuda_graphs=cuda_graphs,
                        mlock=mlock,
                        verbose=False,
                    )
                else:
                    self.llm = Llama(
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        flash_attn=flash,
                        cuda_graphs=cuda_graphs,
                        mlock=mlock,
                        verbose=False,
                    )
                print("[DEBUG] Model + vision projector loaded successfully")

            if mode == "Image Caption":
                images = self._collect_images(image_1, kwargs)
                print(f"[DEBUG] Collected {len(images)} images for analysis")
                if not images:
                    raise ValueError("Connect at least one image in 'Image Caption' mode.")
                sorted_labels = list(images.keys())
                image_labels_str = ", ".join(sorted_labels)

                content_parts = []
                for label in sorted_labels:
                    pil_img = _tensor_to_pil(images[label])
                    b64 = _pil_to_b64(pil_img)
                    content_parts.append({"type": "text", "text": f"[{label}]"})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                content_parts.append({"type": "text", "text": user_prompt})

                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt or DEFAULT_IMAGE_SYSTEM, enable_thinking, len(images), image_labels_str)},
                    {"role": "user", "content": content_parts}
                ]
            else:
                if not (input_text or "").strip():
                    raise ValueError("Enter text in the 'input_text' field.")
                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt or DEFAULT_TEXT_SYSTEM, enable_thinking, 0, image_labels_str)},
                    {"role": "user", "content": f"{input_text}\n\n{user_prompt}".strip()}
                ]

            if not enable_thinking:
                for msg in messages:
                    if msg["role"] == "user":
                        if isinstance(msg["content"], list):
                            last = next((p for p in reversed(msg["content"]) if p.get("type") == "text"), None)
                            if last:
                                last["text"] = last.get("text", "").rstrip() + " /no_think"
                        elif isinstance(msg["content"], str):
                            msg["content"] = msg["content"].rstrip() + " /no_think"

            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_sent,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                seed=seed if seed != 0 else None,
                reasoning_budget=reasoning_budget,
            )

        # Normalize response
        if isinstance(response, dict):
            choices = response.get("choices", [{}])
            choice = choices[0] if choices else {}
            msg_dict = choice.get("message", {})
            content = msg_dict.get("content", "") or ""
        else:
            msg = response.choices[0].message
            content = getattr(msg, "content", "") or ""

        caption = self._extract_caption(content)

        saved_file_path = ""
        if save_to_file and caption.strip():
            saved_file_path = self._save_caption(caption)

        debug_str = f"=== BACKEND: {backend} ===\nMode: {mode}\nThinking: {enable_thinking}\nReasoning Budget: {reasoning_budget}\n=== CAPTION ===\n{caption}\n"

        if backend == "Local Standalone (llama-cpp-python)" and unload_after_inference:
            self._unload_model()

        return (caption, debug_str, saved_file_path)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {"VisionLLMCaptioner": VisionLLMCaptioner}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMCaptioner": "Gemma-4 Vision Captioner + Prompt Enhancer (Remote API or Local Standalone)"
}
