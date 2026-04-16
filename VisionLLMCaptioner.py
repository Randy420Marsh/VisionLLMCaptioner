--- VisionLLMCaptioner.py (原始)


+++ VisionLLMCaptioner.py (修改后)
"""
VisionLLMCaptioner - Gemma-4 Vision Captioner + Prompt Enhancer for ComfyUI
OPTIMIZED FOR Randy420Marsh/llama-cpp-python fork (Gemma4ChatHandler + vision)
"""

import base64
from io import BytesIO
import re
import gc
import os
from datetime import datetime

import torch
from PIL import Image

# ComfyUI-specific imports - handle gracefully for import scanning
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    folder_paths = None
    COMFYUI_AVAILABLE = False
    print("[VisionLLMCaptioner] Warning: folder_paths not available (not running in ComfyUI context)")

# Your custom fork
try:
    from llama_cpp import Llama
    # Your fork provides Gemma4ChatHandler
    try:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        HAS_GEMMA4_HANDLER = True
    except ImportError:
        HAS_GEMMA4_HANDLER = False
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    HAS_GEMMA4_HANDLER = False


def _pil_to_b64(pil_img):
    """Convert PIL image to base64 string."""
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _tensor_to_pil(tensor):
    """Convert ComfyUI image tensor to PIL Image."""
    frame = tensor[0] if tensor.ndim == 4 else tensor
    arr = (frame * 255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def _make_client(server_url):
    """Create OpenAI-compatible client for remote API."""
    import httpx
    from openai import OpenAI
    base = server_url.rstrip("/") + "/"
    return OpenAI(
        base_url=base,
        api_key="lm-studio",
        http_client=httpx.Client(base_url=base, follow_redirects=True),
    )


DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class visual analyst and master prompt engineer specializing in photorealistic, cinematic image generation. "
    "You have been given {image_count} image(s) labeled {image_labels}. Always reference images strictly by their exact label in your internal thinking ONLY."
)


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
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
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
        """Build system prompt with image count and labels, optionally wrapped in think tags."""
        prompt = (raw or DEFAULT_SYSTEM_PROMPT).strip()
        prompt = prompt.replace("{image_count}", str(image_count))
        prompt = prompt.replace("{image_labels}", image_labels)
        if enable_thinking:
            prompt = "<|think|>\n" + prompt
        return prompt

    @staticmethod
    def _extract_caption(content: str) -> str:
        """Extract clean caption from response, removing thinking tags and metadata."""
        if not content or not content.strip():
            return "[EMPTY OR THINKING ONLY — check debug output]"

        # Remove channel markers
        content = re.sub(r"<\|?channel\|?>.*?<\|?channel\|?>", "", content, flags=re.DOTALL | re.IGNORECASE)
        # Remove think blocks (various formats)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<\|think\|>.*?<\|/think\|>", "", content, flags=re.DOTALL | re.IGNORECASE)
        # Remove prompt generation metadata
        content = re.sub(r"\*\*Image Generation Prompt:\*\*.*", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"Image Generation Prompt:.*", "", content, flags=re.DOTALL | re.IGNORECASE)
        return content.strip()

    def _collect_images(self, image_1, kwargs):
        """Collect all connected image inputs and sort them by index."""
        images = {}
        if image_1 is not None:
            images["image_1"] = image_1
        for k, v in kwargs.items():
            if k.startswith("image_") and v is not None:
                images[k] = v
        sorted_images = dict(sorted(images.items(), key=lambda x: int(x[0].split("_")[1])))
        print(f"[VisionLLMCaptioner] Collected {len(sorted_images)} image(s) → {list(sorted_images.keys())}")
        return sorted_images

    def _unload_model(self):
        """Unload model from memory and clear GPU cache."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[VisionLLMCaptioner] Model unloaded")

    def _save_caption(self, caption: str) -> str:
        """Save caption to output directory."""
        if not COMFYUI_AVAILABLE or folder_paths is None:
            print("[VisionLLMCaptioner] Cannot save file: ComfyUI folder_paths not available")
            return ""
        output_dir = os.path.join(folder_paths.get_output_directory(), "captions")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(output_dir, f"caption_{timestamp}.txt")
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

        print(f"[DEBUG] START - Backend: {backend} | Mode: {mode} | Images: {'Yes' if mode == 'Image Caption' else 'No'}")

        # Calculate total tokens needed (including thinking budget)
        max_tokens_sent = max_tokens + (thinking_budget if enable_thinking else 0)
        reasoning_budget = thinking_budget if enable_thinking else 0

        content = ""

        if backend == "Remote API (llama-server)":
            # Remote API mode using OpenAI-compatible endpoint
            print(f"[DEBUG] Connecting to remote API: {server_url}")
            client = _make_client(server_url)

            if mode == "Image Caption":
                images = self._collect_images(image_1, kwargs)
                if not images:
                    raise ValueError("Connect at least one image in 'Image Caption' mode.")

                content_parts = []
                for label, img_tensor in images.items():
                    pil_img = _tensor_to_pil(img_tensor)
                    b64 = _pil_to_b64(pil_img)
                    content_parts.append({"type": "text", "text": f"[{label}]"})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                content_parts.append({"type": "text", "text": user_prompt})

                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt, enable_thinking, len(images), ", ".join(images.keys()))},
                    {"role": "user", "content": content_parts}
                ]
            else:
                if not (input_text or "").strip():
                    raise ValueError("Enter text in the 'input_text' field.")
                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt, enable_thinking, 0, "(text-only mode)")},
                    {"role": "user", "content": f"{input_text}\n\n{user_prompt}".strip()}
                ]

            # Build API parameters
            api_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens_sent,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed if seed != 0 else None,
            }
            if enable_thinking:
                api_params["reasoning_budget"] = reasoning_budget

            response = client.chat.completions.create(**api_params)
            content = response.choices[0].message.content or ""

        else:  # Local Standalone - YOUR FORK
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python not found. Please install your fork: pip install git+https://github.com/Randy420Marsh/llama-cpp-python.git")

            if self.llm is None:
                print(f"[DEBUG] Loading Gemma-4 with Gemma4ChatHandler → {model_path}")
                flash = (attention_mode == "Flash Attention (recommended)")

                chat_handler = None
                if HAS_GEMMA4_HANDLER:
                    chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)
                    print("[DEBUG] Using Gemma4ChatHandler (from your fork) - Vision ready")
                else:
                    print("[WARNING] Gemma4ChatHandler not found in your llama-cpp-python installation. Vision may not work correctly.")

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
                print("[DEBUG] Gemma-4 + Vision loaded successfully with your fork")

            # Image handling (multi-image support)
            if mode == "Image Caption":
                images = self._collect_images(image_1, kwargs)
                if not images:
                    raise ValueError("Connect at least one image in 'Image Caption' mode.")

                content_parts = []
                for label, img_tensor in images.items():
                    pil_img = _tensor_to_pil(img_tensor)
                    b64 = _pil_to_b64(pil_img)
                    content_parts.append({"type": "text", "text": f"[{label}]"})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

                content_parts.append({"type": "text", "text": user_prompt})

                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt, enable_thinking, len(images), ", ".join(images.keys()))},
                    {"role": "user", "content": content_parts}
                ]
            else:
                if not (input_text or "").strip():
                    raise ValueError("Enter text in the 'input_text' field.")
                messages = [
                    {"role": "system", "content": self._build_system_prompt(system_prompt, enable_thinking, 0, "(text-only mode)")},
                    {"role": "user", "content": f"{input_text}\n\n{user_prompt}".strip()}
                ]

            # Safe call with reasoning_budget
            local_params = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens_sent,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed if seed != 0 else None,
            }
            if enable_thinking:
                local_params["reasoning_budget"] = reasoning_budget

            response = self.llm.create_chat_completion(**local_params)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract clean caption from response
        caption = self._extract_caption(content)

        # Save caption to file if requested
        saved_file_path = self._save_caption(caption) if save_to_file and caption.strip() else ""

        # Build debug output
        debug_str = f"=== BACKEND: {backend} ===\nMode: {mode}\nThinking: {enable_thinking}\n=== RAW ===\n{content}\n\n=== CAPTION ===\n{caption}"

        # Unload model if requested (local mode only)
        if backend == "Local Standalone (llama-cpp-python)" and unload_after_inference:
            self._unload_model()

        return (caption, debug_str, saved_file_path)


# Registration
NODE_CLASS_MAPPINGS = {
    "VisionLLMCaptioner": VisionLLMCaptioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMCaptioner": "VisionLLMCaptioner - Gemma-4 Vision + Prompt Enhancer"
}
