"""
VisionLLMCaptioner - Gemma-4 Vision Captioner + Prompt Enhancer for ComfyUI
FULLY COMPATIBLE with updated Gemma4ChatHandler (structured thinking + native audio)

FIXES vs original:
  - cpu_mode now correctly uses n_gpu_layers=0 (env-var approach doesn't work post-import)
  - thinking_budget is actually passed to local inference (was silently dropped before)
  - enable_thinking guarded against TypeError for non-fork llama-cpp-python builds
  - n_ubatch added so GPU decode respects n_batch (was defaulting to 512)
  - Model cache invalidated when key parameters change (was reusing stale model)
  - presence_penalty moved to top-level OpenAI params (was buried in extra_body)
  - seed=-1 sentinel replaces seed==0 check (0 is a valid seed)
  - _tensor_to_pil forces RGB conversion (RGBA images broke some vision models)
  - mlock failure is caught; node retries with mlock=False instead of crashing
  - Thinking tag regex properly escaped to handle alternate formats
"""

import base64
from io import BytesIO
import re
import gc
import os
import json
from datetime import datetime

import torch
from PIL import Image

try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    folder_paths = None
    COMFYUI_AVAILABLE = False

try:
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        HAS_GEMMA4_HANDLER = True
    except ImportError:
        HAS_GEMMA4_HANDLER = False
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    HAS_GEMMA4_HANDLER = False


def _pil_to_b64(pil_img: Image.Image) -> str:
    # FIX: always convert to RGB so RGBA tensors don't produce 4-channel PNGs
    # that confuse vision models expecting RGB input.
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _tensor_to_pil(tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C) or (H,W,C) to PIL."""
    frame = tensor[0] if tensor.ndim == 4 else tensor
    arr = (frame * 255).clamp(0, 255).byte().cpu().numpy()
    img = Image.fromarray(arr)
    # FIX: force RGB — handled in _pil_to_b64, but do it here too for clarity
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _make_client(server_url: str, max_tokens: int = 8192):
    import httpx
    from openai import OpenAI
    base = server_url.rstrip("/") + "/"
    # Scale read timeout with token budget: ~0.5s per token as a generous estimate
    # for slow models, with a floor of 120s and ceiling of 1800s (30 min).
    read_timeout = min(max(120.0, max_tokens * 0.5), 1800.0)
    return OpenAI(
        base_url=base,
        api_key="lm-studio",
        http_client=httpx.Client(
            base_url=base,
            follow_redirects=True,
            timeout=httpx.Timeout(read_timeout, connect=30.0),
        ),
    )


DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class visual analyst and master prompt engineer specializing in "
    "photorealistic, cinematic image generation. "
    "You have been given {image_count} image(s) labeled {image_labels}. "
    "Always reference images strictly by their exact label in your internal thinking ONLY. "
    "Output the final prompt only, no explanations."
)

# Default built-in presets (cannot be deleted)
DEFAULT_PRESETS = {
    "Image Caption": {
        "Custom": "",
        "Default": DEFAULT_SYSTEM_PROMPT,
    },
    "Text -> Detailed Image Prompt": {
        "Custom": "",
        "Default": (
            "You are a master prompt engineer specializing in detailed, "
            "photorealistic image generation prompts. "
            "Expand the input into a comprehensive, detailed prompt."
        ),
    },
    "Text to Text": {
        "Custom": "",
        "Default": (
            "You are a helpful text assistant. Process and transform the input text "
            "according to the user's instructions."
        ),
    },
}

# Presets file path
def _get_presets_file_path():
    """Get the path to the presets JSON file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "presets.json")

# Load presets from file or use defaults
PRESETS = None

def _load_presets():
    """Load presets from JSON file, falling back to defaults."""
    global PRESETS
    presets_file = _get_presets_file_path()
    if os.path.exists(presets_file):
        try:
            with open(presets_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all modes exist
                result = {mode: dict(presets) for mode, presets in DEFAULT_PRESETS.items()}
                for mode, presets in loaded.items():
                    if mode in result:
                        # Merge, but keep default presets intact
                        result[mode].update(presets)
                    else:
                        result[mode] = presets
                PRESETS = result
                print(f"[VisionLLMCaptioner] Loaded presets from {presets_file}")
        except Exception as e:
            print(f"[VisionLLMCaptioner] Error loading presets: {e}, using defaults")
            PRESETS = {mode: dict(presets) for mode, presets in DEFAULT_PRESETS.items()}
    else:
        PRESETS = {mode: dict(presets) for mode, presets in DEFAULT_PRESETS.items()}
        _save_presets()
    return PRESETS

def _save_presets():
    """Save current presets to JSON file."""
    global PRESETS
    presets_file = _get_presets_file_path()
    try:
        with open(presets_file, 'w', encoding='utf-8') as f:
            json.dump(PRESETS, f, indent=2, ensure_ascii=False)
        print(f"[VisionLLMCaptioner] Saved presets to {presets_file}")
        return True
    except Exception as e:
        print(f"[VisionLLMCaptioner] Error saving presets: {e}")
        return False

# Initialize presets on import
_load_presets()


class VisionLLMCaptioner:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (
                    ["Remote API (llama-server)", "Local Standalone (llama-cpp-python)"],
                    {"default": "Remote API (llama-server)"},
                ),
                "mode": (
                    ["Image Caption", "Text -> Detailed Image Prompt", "Text to Text"],
                    {"default": "Image Caption"},
                ),
                "preset": (
                    ["Custom", "Default"],
                    {"default": "Default"},
                ),
                "server_url":   ("STRING", {"default": "http://127.0.0.1:8080/v1", "multiline": False}),
                "model_name":   ("STRING", {"default": "gemma-4-E4B-it-abliterated.i1-Q4_K_M.gguf", "multiline": False}),
                "model_path":   ("STRING", {"default": "/media/john/A024FBBA24FB9210/LLAMA_GGUF/gemma-4-E4B-it-abliterated.i1-Q4_K_M.gguf", "multiline": False}),
                "mmproj_path":  ("STRING", {"default": "/media/john/A024FBBA24FB9210/LLAMA_GGUF/gemma-4-E4B-it-abliterated.i1-Q4_K_M-mmproj-F16.gguf", "multiline": False}),
                "n_gpu_layers": ("INT",    {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx":        ("INT",    {"default": 32768, "min": 2048, "max": 131072, "step": 512}),
                "n_batch":      ("INT",    {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "attention_mode": (
                    ["Flash Attention (recommended)", "Standard Attention"],
                    {"default": "Flash Attention (recommended)"},
                ),
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True, "label": "System Prompt"}),
                "user_prompt":   ("STRING", {"default": "Describe the scene in extreme detail.", "multiline": True, "label": "Prompt"}),
                "max_tokens":    ("INT",    {"default": 8192, "min": 64, "max": 32768, "step": 64}),
                "enable_thinking": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Thinking ON", "label_off": "Thinking OFF (No Think)"},
                ),
                "thinking_budget": ("INT", {"default": 8192, "min": 256, "max": 32768, "step": 128}),
                "temperature":      ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 1.5, "step": 0.05}),
                "top_p":            ("FLOAT", {"default": 0.95, "min": 0.0,  "max": 1.0, "step": 0.05}),
                "top_k":            ("INT",   {"default": 64,   "min": 1,    "max": 200}),
                "repeat_penalty":   ("FLOAT", {"default": 1.0,  "min": 0.9,  "max": 2.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 1.0,  "min": -2.0, "max": 2.0, "step": 0.05}),
                "min_p":            ("FLOAT", {"default": 0.05, "min": 0.0,  "max": 1.0, "step": 0.01}),
                "unload_after_inference": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Unload after run (recommended)", "label_off": "Keep model in memory"},
                ),
                "save_to_file": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Save caption to file", "label_off": "Don't save"},
                ),
            },
            "optional": {
                "image_1":    ("IMAGE",),
                "input_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Short idea → e.g. 'cyberpunk girl with neon hair'",
                        "label": "Prompt Enhance",
                    },
                ),
                # FIX: seed default changed to -1; -1 means "random" (no seed).
                # The original used 0 as a sentinel, which is also a valid seed value.
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
                "cuda_graphs": (
                    "BOOLEAN",
                    {"default": False, "label_on": "CUDA Graphs (faster on 40/50 series)", "label_off": "No CUDA Graphs"},
                ),
                "mlock": (
                    "BOOLEAN",
                    {"default": True, "label_on": "mlock (prevent swapping)", "label_off": "No mlock"},
                ),
                # FIX: cpu_mode now correctly works — see _load_local_model() for explanation.
                "cpu_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "CPU only (bypass CUDA crash)", "label_off": "GPU (default)"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "full_raw_debug", "saved_file_path")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "vision"

    def __init__(self):
        self.llm = None
        self._model_key = None  # FIX: track loaded model config for cache invalidation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, raw: str, image_count: int, image_labels: str) -> str:
        prompt = (raw or DEFAULT_SYSTEM_PROMPT).strip()
        prompt = prompt.replace("{image_count}", str(image_count))
        prompt = prompt.replace("{image_labels}", image_labels)
        return prompt

    @staticmethod
    def _extract_thinking_and_answer(content: str):
        """
        Split raw model output into (thinking_text, answer_text).

        Handles:
          <|think|>...thinking...<|/think|>answer    ← Unsloth/Gemma format
          <think>...thinking...</think>answer         ← alternate format
          plain text (no tags)                        ← treat entire content as answer

        Returns (thinking_text, answer_text). Either may be empty string.
        """
        thinking = ""
        answer = content or ""

        if not content or not content.strip():
            return ("", "")

        # FIX: pipe characters in tag names must be escaped in regex.
        # Pattern 1: <|think|>...</|think|>  (Unsloth standard)
        m = re.search(r"<\|think\|>(.*?)<\|/think\|>", content, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            answer = (content[: m.start()] + content[m.end() :]).strip()
            return (thinking, answer)

        # Pattern 2: <think>...</think>  (standard XML-style fallback)
        m2 = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if m2:
            thinking = m2.group(1).strip()
            answer = (content[: m2.start()] + content[m2.end() :]).strip()
            return (thinking, answer)

        return (thinking, answer)

    @staticmethod
    def _extract_caption(content: str) -> str:
        """Light cleanup for final caption output."""
        if not content or not content.strip():
            return "[EMPTY RESPONSE]"
        content = re.sub(
            r"<\|?/?channel\|?>.*?<\|?/?channel\|?>", "", content, flags=re.DOTALL | re.IGNORECASE
        )
        return content.strip()

    def _collect_images(self, image_1, kwargs: dict) -> dict:
        images = {}
        if image_1 is not None:
            images["image_1"] = image_1
        for k, v in kwargs.items():
            if k.startswith("image_") and k != "image_1" and v is not None:
                images[k] = v
        sorted_images = dict(
            sorted(images.items(), key=lambda x: int(x[0].split("_")[1]))
        )
        print(f"[VisionLLMCaptioner] Collected {len(sorted_images)} image(s) → {list(sorted_images.keys())}")
        return sorted_images

    def _unload_model(self):
        if self.llm is not None:
            try:
                if hasattr(self.llm, "close"):
                    self.llm.close()
            except Exception:
                pass
            del self.llm
            self.llm = None
            self._model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[VisionLLMCaptioner] Model unloaded and memory freed")

    def _save_caption(self, caption: str) -> str:
        if not COMFYUI_AVAILABLE or folder_paths is None:
            return ""
        output_dir = os.path.join(folder_paths.get_output_directory(), "captions")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(output_dir, f"caption_{timestamp}.txt")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(caption)
        return full_path

    def _load_local_model(
        self,
        model_path: str,
        mmproj_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        n_batch: int,
        flash: bool,
        cuda_graphs: bool,
        mlock: bool,
        cpu_mode: bool,
    ):
        """
        Load (or reuse) the local Llama model.

        FIX – cpu_mode:
          The original code set LLAMA_CUDA / CUDA_VISIBLE_DEVICES env vars here,
          but llama-cpp-python reads those only during its own import (which already
          happened at the top of this file).  Setting them afterward does nothing.
          The correct approach is to pass n_gpu_layers=0, which instructs the already-
          loaded library to run all layers on CPU regardless of CUDA availability.

        FIX – model cache invalidation:
          The original only checked `self.llm is None`.  If the user changes model_path,
          n_ctx, etc., the stale model would keep running.  Now we use a tuple key.

        FIX – n_ubatch:
          llama-cpp-python defaults n_ubatch to 512.  For large n_batch values the
          micro-batch should match or be at most n_batch to avoid decode bottlenecks.
          We set n_ubatch = n_batch.

        FIX – mlock failure:
          mlock requires elevated privileges on some Linux systems.  If the Llama
          constructor raises, we retry with mlock=False before giving up.
        """
        # cpu_mode overrides n_gpu_layers — do NOT touch env vars here
        effective_gpu_layers = 0 if cpu_mode else n_gpu_layers
        if cpu_mode:
            print("[VisionLLMCaptioner] CPU mode → n_gpu_layers forced to 0")

        # Cache key — compare against requested settings for early-exit reuse check.
        # NOTE: model_key is rebuilt after load to record *effective* mlock value
        # (BUG-4 FIX: mlock retry may change what was actually used).
        model_key = (
            model_path, mmproj_path, effective_gpu_layers,
            n_ctx, n_batch, flash, cuda_graphs, mlock,
        )
        if self.llm is not None and self._model_key == model_key:
            print("[VisionLLMCaptioner] Reusing cached model")
            return  # already loaded with identical config

        if self.llm is not None:
            print("[VisionLLMCaptioner] Config changed — reloading model")
            self._unload_model()

        print(f"[VisionLLMCaptioner] Loading model: {model_path}")

        chat_handler = None
        if HAS_GEMMA4_HANDLER:
            chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)
            print("[VisionLLMCaptioner] Gemma4ChatHandler ready")
        else:
            print("[VisionLLMCaptioner] WARNING: Gemma4ChatHandler not found. "
                  "Install from https://github.com/Randy420Marsh/llama-cpp-python")

        constructor_kwargs = dict(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=effective_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            flash_attn=flash,
            use_mlock=mlock,           # FIX: use correct param name (use_mlock, not mlock)
            verbose=False,
        )

        # BUG-3 FIX: n_ubatch and cuda_graphs are only available in newer / fork builds.
        # Add them conditionally so older versions don't TypeError-crash.
        # n_ubatch should equal n_batch for best GPU decode throughput.
        constructor_kwargs["n_ubatch"] = n_batch   # guarded by outer try/except below
        if cuda_graphs:
            constructor_kwargs["cuda_graphs"] = True

        # BUG-3 FIX: strip unknown kwargs one at a time and retry, so n_ubatch or
        # cuda_graphs don't hard-crash older llama-cpp-python builds.
        # BUG-4 FIX: track whether mlock was actually used (may differ from requested)
        #            so _model_key always records what the model was truly loaded with.
        effective_mlock = mlock
        _optional_kwargs = ("n_ubatch", "cuda_graphs")
        _loaded = False
        while not _loaded:
            try:
                self.llm = Llama(**constructor_kwargs)
                _loaded = True
            except TypeError as e:
                stripped = False
                for opt in _optional_kwargs:
                    if opt in str(e) and opt in constructor_kwargs:
                        print(f"[VisionLLMCaptioner] '{opt}' not supported by this build — removing and retrying")
                        del constructor_kwargs[opt]
                        stripped = True
                        break
                if not stripped:
                    raise  # some other TypeError, propagate it
            except Exception as e:
                if effective_mlock and "mlock" in str(e).lower():
                    print(f"[VisionLLMCaptioner] mlock failed ({e}), retrying without mlock")
                    constructor_kwargs["use_mlock"] = False
                    effective_mlock = False
                else:
                    raise

        # Rebuild key with actual values used (effective_mlock may differ from requested)
        model_key = (
            model_path, mmproj_path, effective_gpu_layers,
            n_ctx, n_batch, flash, cuda_graphs, effective_mlock,
        )

        self._model_key = model_key
        print("[VisionLLMCaptioner] Model loaded successfully")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        backend, mode, preset, server_url, model_name, model_path, mmproj_path,
        n_gpu_layers, n_ctx, n_batch, attention_mode,
        system_prompt, user_prompt, max_tokens, enable_thinking, thinking_budget,
        temperature, top_p, top_k, repeat_penalty, presence_penalty, min_p,
        unload_after_inference, save_to_file,
        image_1=None, input_text="", seed=-1,
        cuda_graphs=False, mlock=True, cpu_mode=False,
        **kwargs,
    ):
        print(f"[VisionLLMCaptioner] START — Backend: {backend} | Mode: {mode} | Preset: {preset} | Thinking: {enable_thinking}")

        # Apply preset if not Custom
        if preset != "Custom" and mode in PRESETS and preset in PRESETS[mode]:
            preset_value = PRESETS[mode][preset]
            if preset_value:
                system_prompt = preset_value
                print(f"[VisionLLMCaptioner] Applied preset '{preset}' for mode '{mode}'")

        # FIX: seed=-1 means "no seed"; 0 is now a valid reproducible seed
        resolved_seed = None if seed < 0 else seed

        # Total token budget = output cap + thinking budget (thinking consumed separately)
        max_tokens_sent = max_tokens + (thinking_budget if enable_thinking else 0)

        # Wrap entire inference in try/finally so the model is always unloaded on error.
        # Without this, a crash mid-inference leaks GPU memory until ComfyUI restarts.
        try:
            return self._run_inference(
                backend, mode, server_url, model_name, model_path, mmproj_path,
                n_gpu_layers, n_ctx, n_batch, attention_mode,
                system_prompt, user_prompt, max_tokens_sent, enable_thinking, thinking_budget,
                temperature, top_p, top_k, repeat_penalty, presence_penalty, min_p,
                save_to_file, resolved_seed,
                image_1, input_text, cuda_graphs, mlock, cpu_mode,
                **kwargs,
            )
        except Exception:
            # On any error, unload the model to free GPU memory
            if backend == "Local Standalone (llama-cpp-python)":
                self._unload_model()
            raise
        finally:
            if backend == "Local Standalone (llama-cpp-python)" and unload_after_inference:
                self._unload_model()

    def _run_inference(
        self,
        backend, mode, server_url, model_name, model_path, mmproj_path,
        n_gpu_layers, n_ctx, n_batch, attention_mode,
        system_prompt, user_prompt, max_tokens_sent, enable_thinking, thinking_budget,
        temperature, top_p, top_k, repeat_penalty, presence_penalty, min_p,
        save_to_file, resolved_seed,
        image_1, input_text, cuda_graphs, mlock, cpu_mode,
        **kwargs,
    ):

        thinking_text = ""
        answer_text = ""

        # ── REMOTE API ─────────────────────────────────────────────────────────
        if backend == "Remote API (llama-server)":
            client = _make_client(server_url, max_tokens=max_tokens_sent)

            messages = self._build_messages(
                mode, image_1, kwargs, system_prompt, user_prompt, input_text,
                enable_thinking, inject_thinking_stub=False,  # NOTE-6: remote uses reasoning_budget
            )

            # FIX: presence_penalty belongs in the top-level OpenAI params, not extra_body.
            # repeat_penalty is llama-specific and must stay in extra_body.
            api_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens_sent,
                "top_p": top_p,
                "presence_penalty": presence_penalty,   # standard OpenAI field
                "seed": resolved_seed,
            }

            extra_body = {}
            if top_k and top_k > 0:
                extra_body["top_k"] = top_k
            if min_p is not None:
                extra_body["min_p"] = min_p
            if repeat_penalty is not None:
                extra_body["repeat_penalty"] = repeat_penalty   # send always; 1.0 = llama default
            if enable_thinking:
                extra_body["reasoning_budget"] = thinking_budget

            if extra_body:
                api_params["extra_body"] = extra_body

            try:
                response = client.chat.completions.create(**api_params)
            except Exception as api_err:
                err_name = type(api_err).__name__
                err_msg = str(api_err)
                # Surface connection / timeout / HTTP errors clearly
                if "ConnectError" in err_name or "ConnectionError" in err_name:
                    raise ConnectionError(
                        f"[VisionLLMCaptioner] Cannot connect to server at {server_url}. "
                        f"Is the server running? ({err_name}: {err_msg})"
                    ) from api_err
                if "Timeout" in err_name or "timed out" in err_msg.lower():
                    raise TimeoutError(
                        f"[VisionLLMCaptioner] Request to {server_url} timed out. "
                        f"The model may need more time for {max_tokens_sent} tokens. "
                        f"Try reducing max_tokens or thinking_budget. ({err_name})"
                    ) from api_err
                raise RuntimeError(
                    f"[VisionLLMCaptioner] Remote API error: {err_name}: {err_msg}"
                ) from api_err

            try:
                msg = response.choices[0].message
            except (IndexError, AttributeError) as e:
                raise RuntimeError(
                    f"[VisionLLMCaptioner] Server returned an empty or invalid response. "
                    f"Check server logs. ({e})"
                ) from e

            thinking_text = getattr(msg, "reasoning_content", "") or ""
            answer_text = msg.content or ""
            # BUG-2 FIX: if the server embeds <|think|> tags directly in msg.content
            # (instead of populating reasoning_content), extract them now so raw tags
            # don't leak into the final caption.
            if not thinking_text and answer_text:
                thinking_text, answer_text = self._extract_thinking_and_answer(answer_text)
            # Final fallback: if content is empty but reasoning_content has something, use it
            if not answer_text:
                answer_text = thinking_text

        # ── LOCAL STANDALONE ───────────────────────────────────────────────────
        else:
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError(
                    "llama-cpp-python not found. "
                    "Install from: https://github.com/Randy420Marsh/llama-cpp-python"
                )

            flash = (attention_mode == "Flash Attention (recommended)")
            self._load_local_model(
                model_path, mmproj_path, n_gpu_layers, n_ctx, n_batch,
                flash, cuda_graphs, mlock, cpu_mode,
            )

            messages = self._build_messages(
                mode, image_1, kwargs, system_prompt, user_prompt, input_text,
                enable_thinking, inject_thinking_stub=True,  # NOTE-6: local uses prefill stub
            )

            # FIX: thinking_budget was never included in local_params (silently dropped).
            # FIX: enable_thinking guarded — older or non-fork builds raise TypeError.
            local_params = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens_sent,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "seed": resolved_seed,
            }

            if enable_thinking:
                local_params["enable_thinking"] = True
                local_params["thinking_budget"] = thinking_budget  # FIX: was missing

            print("[VisionLLMCaptioner] Running local inference…")
            try:
                try:
                    result = self.llm.create_chat_completion(**local_params)
                except TypeError as te:
                    if "enable_thinking" in str(te) or "thinking_budget" in str(te):
                        # FIX: fork doesn't support these kwargs — fall back gracefully.
                        # Thinking is instead triggered by the <|think|> prefix we injected
                        # into the message in _build_messages().
                        print(
                            f"[VisionLLMCaptioner] Fork doesn't accept enable_thinking/thinking_budget "
                            f"as kwargs ({te}). Falling back to message-level <|think|> injection."
                        )
                        local_params.pop("enable_thinking", None)
                        local_params.pop("thinking_budget", None)
                        result = self.llm.create_chat_completion(**local_params)
                    else:
                        raise
            except Exception as local_err:
                err_name = type(local_err).__name__
                if isinstance(local_err, TypeError):
                    raise  # already handled above; re-raise for non-thinking TypeErrors
                raise RuntimeError(
                    f"[VisionLLMCaptioner] Local inference failed: {err_name}: {local_err}. "
                    f"Check model path, available VRAM, and n_ctx settings."
                ) from local_err

            try:
                message = result["choices"][0].get("message", {})
                raw_content = message.get("content", "") or ""
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected response structure from local model: {e}") from e

            print(f"[VisionLLMCaptioner] Raw response length: {len(raw_content)}")

            if raw_content.strip():
                thinking_text, answer_text = self._extract_thinking_and_answer(raw_content)
                if not answer_text.strip() and thinking_text.strip():
                    answer_text = thinking_text
            else:
                print(f"[VisionLLMCaptioner] WARNING: Empty content. Message keys: {list(message.keys())}")
                print(f"[VisionLLMCaptioner] Full message: {repr(message)[:800]}")

        # ── FINALISE ───────────────────────────────────────────────────────────
        caption = self._extract_caption(answer_text)
        saved_file_path = self._save_caption(caption) if save_to_file and caption.strip() else ""

        debug_str = (
            f"=== BACKEND: {backend} ===\n"
            f"Mode: {mode} | Thinking: {enable_thinking} | Seed: {resolved_seed}\n"
            f"=== THINKING ===\n{thinking_text}\n\n"
            f"=== CAPTION ===\n{caption}\n\n"
            f"=== RAW CONTENT ===\n{answer_text}"
        )

        return (caption, debug_str, saved_file_path)

    # ------------------------------------------------------------------
    # Message builder (shared between remote and local paths)
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        mode: str,
        image_1,
        kwargs: dict,
        system_prompt: str,
        user_prompt: str,
        input_text: str,
        enable_thinking: bool,
        inject_thinking_stub: bool = False,
    ) -> list:
        """
        Build the messages list for either backend.

        When enable_thinking is True and the fork does not accept enable_thinking
        as a kwarg, the <|think|> token is injected as an assistant prefix message
        so the model starts its response in thinking mode (Unsloth-recommended approach).
        """
        if mode == "Image Caption":
            images = self._collect_images(image_1, kwargs)
            if not images:
                raise ValueError("Connect at least one image in 'Image Caption' mode.")

            content_parts = []
            for label, img_tensor in images.items():
                pil_img = _tensor_to_pil(img_tensor)
                b64 = _pil_to_b64(pil_img)
                content_parts.append({"type": "text", "text": f"[{label}]"})
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )
            content_parts.append({"type": "text", "text": user_prompt})

            messages = [
                {
                    "role": "system",
                    "content": self._build_system_prompt(
                        system_prompt, len(images), ", ".join(images.keys())
                    ),
                },
                {"role": "user", "content": content_parts},
            ]
        else:
            if not (input_text or "").strip():
                raise ValueError("Enter text in the 'input_text' field for Text mode.")

            # Collect any connected reference images (used in "Text -> Detailed Image Prompt")
            ref_images = self._collect_images(image_1, kwargs)

            if ref_images and mode == "Text -> Detailed Image Prompt":
                content_parts = []
                for label, img_tensor in ref_images.items():
                    pil_img = _tensor_to_pil(img_tensor)
                    b64 = _pil_to_b64(pil_img)
                    content_parts.append({"type": "text", "text": f"[{label}]"})
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    )
                content_parts.append({"type": "text", "text": f"{input_text}\n\n{user_prompt}".strip()})

                messages = [
                    {
                        "role": "system",
                        "content": self._build_system_prompt(
                            system_prompt, len(ref_images), ", ".join(ref_images.keys())
                        ),
                    },
                    {"role": "user", "content": content_parts},
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": self._build_system_prompt(system_prompt, 0, "(text-only mode)"),
                    },
                    {"role": "user", "content": f"{input_text}\n\n{user_prompt}".strip()},
                ]

        # NOTE-6 FIX: only inject the <|think|> assistant-prefill stub for local inference.
        # llama-server already honours reasoning_budget in extra_body to trigger thinking.
        # Sending a dangling incomplete assistant turn to strict OpenAI-compatible remotes
        # (OpenRouter, Azure, etc.) causes 400 errors or silent ignoring.
        if enable_thinking and inject_thinking_stub:
            messages.append({"role": "assistant", "content": "<|think|>"})

        return messages


# -------------------------------------------------------------------------
# Preset Management API Endpoints
# -------------------------------------------------------------------------

WEB_DIRECTORY = "./js"


def setup_preset_api(routes):
    """Setup preset management API endpoints on the given RouteDef."""
    from aiohttp import web

    @routes.get("/vision_llmcaptioner/presets")
    async def get_presets(request):
        """Get all presets for all modes."""
        return web.json_response(PRESETS)

    @routes.post("/vision_llmcaptioner/presets/save")
    async def save_preset(request):
        """Save a preset for a specific mode."""
        try:
            data = await request.json()
            mode = data.get("mode")
            name = data.get("name")
            prompt = data.get("prompt")

            if not mode or not name:
                return web.json_response({"error": "mode and name are required"}, status=400)

            if mode not in PRESETS:
                PRESETS[mode] = {}

            # Don't allow overwriting "Custom"
            if name == "Custom":
                return web.json_response({"error": "Cannot overwrite 'Custom' preset"}, status=400)

            PRESETS[mode][name] = prompt
            _save_presets()
            return web.json_response({"success": True, "mode": mode, "name": name})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/vision_llmcaptioner/presets/delete")
    async def delete_preset(request):
        """Delete a preset for a specific mode."""
        try:
            data = await request.json()
            mode = data.get("mode")
            name = data.get("name")

            if not mode or not name:
                return web.json_response({"error": "mode and name are required"}, status=400)

            # Don't allow deleting default presets
            if mode in DEFAULT_PRESETS and name in DEFAULT_PRESETS[mode]:
                return web.json_response({"error": "Cannot delete default preset"}, status=400)

            if mode in PRESETS and name in PRESETS[mode]:
                del PRESETS[mode][name]
                _save_presets()
                return web.json_response({"success": True, "mode": mode, "name": name})
            else:
                return web.json_response({"error": "Preset not found"}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


_preset_api_registered = False

def register_preset_api(server_module):
    """Register preset API with ComfyUI server (idempotent).

    Uses PromptServer.instance.routes which is the RouteDef connected
    to the actual aiohttp router.  The old code used server.routes
    (module-level), which may be a different RouteTableDef that is
    never added to the running router — causing 404s on preset API
    calls and a JSON-parse error in the frontend.
    """
    global _preset_api_registered
    if _preset_api_registered:
        return
    routes = server_module.PromptServer.instance.routes
    setup_preset_api(routes)
    _preset_api_registered = True
    print("[VisionLLMCaptioner] Preset management API registered")


# Registration
NODE_CLASS_MAPPINGS = {
    "VisionLLMCaptioner": VisionLLMCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMCaptioner": "VisionLLMCaptioner - Gemma-4 Vision + Prompt Enhancer (Fixed)",
}
