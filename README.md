# VisionLLMCaptioner – Gemma-4 Vision Captioner + Prompt Enhancer for ComfyUI

A powerful ComfyUI node that leverages **Gemma-4** vision capabilities for image captioning and prompt enhancement. Supports both **Remote API (llama-server)** and **Local Standalone (llama-cpp-python)** modes.

**Optimized for best defaults from [Unsloth Gemma-4 docs](https://unsloth.ai/docs/models/gemma-4)**  
Recommended model: **Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf** + **mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf**

### Features
- Dual Backend Support: Switch between Remote API (OpenAI-compatible) and Local Standalone (your custom llama-cpp-python fork)
- Multi-Image Analysis: Process multiple images simultaneously with labeled references
- Gemma-4 Vision: Full support for Gemma-4's vision capabilities via Gemma4ChatHandler (from your fork)
- Thinking Mode: Enable chain-of-thought reasoning with configurable budget (uses `<|think|>` per Unsloth)
- Prompt Enhancement: Transform simple ideas into detailed, cinematic image generation prompts
- Memory Efficient: Automatic model unloading and GPU cache management
- CUDA Optimized: Built for NVIDIA GPUs with Flash Attention and CUDA Graphs support

## Installation

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git cuda-toolkit-12-0

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows
```

### Step 3: Compile and Install llama-cpp-python from Randy420Marsh Fork

This fork includes the critical `Gemma4ChatHandler` and vision support for Gemma-4 models (including E4B multimodal).

#### Option A: Direct pip install (Simplest)

```bash
# Set CUDA architecture for your GPU (optional but faster compilation)
# RTX 30xx: 8.6, RTX 40xx: 8.9, RTX 50xx: 9.0
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
export FORCE_CMAKE=1

# Install from GitHub
pip install --no-cache-dir git+https://github.com/Randy420Marsh/llama-cpp-python.git
```

#### Option B: Clone and Build Manually (More Control)

```bash
# Clone the repository
git clone https://github.com/Randy420Marsh/llama-cpp-python.git
cd llama-cpp-python

# Set build flags for CUDA
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=off"
export FORCE_CMAKE=1

# Build and install
pip install -e .

# Or build a wheel for distribution
pip wheel . --no-deps
```

#### Common CUDA Architecture Codes

| GPU Series | Example Models | Architecture Code |
|------------|----------------|-------------------|
| RTX 30xx   | 3060, 3070, 3080, 3090 | 86 |
| RTX 40xx   | 4070, 4080, 4090 | 89 |
| RTX 50xx   | 5070, 5080, 5090 | 90 |
| GTX 16xx   | 1650, 1660 | 75 |
| RTX 20xx   | 2060, 2070, 2080 | 75 |

Example for RTX 4090:

```bash
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
```

### Step 4: Verify Installation

```python
python -c "from llama_cpp import Llama; from llama_cpp.llama_chat_format import Gemma4ChatHandler; print('Gemma4ChatHandler available!')"
```

If this runs without errors, your installation is successful.

---

## ComfyUI Node Installation

### Option A: Manual Installation

1. Navigate to your ComfyUI custom_nodes directory:

   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```

2. Clone or copy this repository:

   ```bash
   git clone https://github.com/Randy420Marsh/VisionLLMCaptioner.git
   ```

3. Restart ComfyUI.

### Option B: ComfyUI Manager

1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "VisionLLMCaptioner"
4. Click Install
5. Restart ComfyUI

---

## Usage

### Basic Image Captioning

1. Add the **VisionLLMCaptioner** node to your workflow
2. Connect an image to the image_1 input
3. Set **Backend** to Local Standalone (llama-cpp-python) or Remote API (llama-server)
4. Configure model paths (for local mode) or server URL (for remote mode)
5. Click "Queue Prompt"

### Text-to-Prompt Enhancement

1. Set **Mode** to Text -> Detailed Image Prompt
2. Enter your idea in the input_text field (e.g., "cyberpunk girl with neon hair")
3. Optionally connect reference images for style guidance
4. The node will expand your idea into a detailed, cinematic prompt

### Multi-Image Analysis

The node automatically detects all connected `image_*` inputs:

- Connect image_1, image_2, image_3, etc.
- The system will label them internally as `[image_1]`, `[image_2]`, etc.
- Reference specific images in your prompts for comparative analysis

---

## Configuration (Updated for Gemma-4-E4B Best Defaults)

### Local Standalone Mode (Recommended for your E4B model)

| Parameter | Description | Recommended Value for E4B Q4_K_M |
|-----------|-------------|----------------------------------|
| model_path | Path to Gemma-4 GGUF model | `/path/to/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` |
| mmproj_path | Path to mmproj (vision projector) | `/path/to/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf` |
| n_gpu_layers | Layers offloaded to GPU (-1 = all) | `-1` (full GPU offload) |
| n_ctx | Context window size | `32768` (32K – start here; E4B supports up to 128K) |
| n_batch | Batch size for prompt processing | `2048` |
| attention_mode | Flash Attention for speed | Flash Attention (recommended) |
| cuda_graphs | Enable CUDA Graphs (RTX 40/50 series) | `True` for faster inference on 40/50xx |

### Remote API Mode

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| server_url | llama-server endpoint | `http://127.0.0.1:8080/v1` |
| model_name | Model identifier on server | `gemma-4-E4B` or your custom name |

### Generation Parameters (Aligned with Unsloth Gemma-4 Recommendations)

| Parameter | Description | Recommended Value | Notes |
|-----------|-------------|-------------------|-------|
| enable_thinking | Chain-of-thought reasoning | `True` | Uses `<|think|>` prefix (Unsloth standard) |
| thinking_budget | Max tokens for thinking | `8192` | Adjust based on VRAM; 4K–16K typical |
| temperature | Creativity vs determinism | `1.0` | **Unsloth default** – more natural outputs |
| top_p | Nucleus sampling | `0.95` | Matches Unsloth |
| top_k | Top-K sampling | `64` | Matches Unsloth |
| repeat_penalty | Reduce repetition | `1.0` | **Unsloth: 1.0 or disabled** (unless looping) |
| presence_penalty | Encourage new topics | `1.0` | **Unsloth: 1.0 or disabled** |
| min_p | Minimum probability | `0.05` | Good balance |
| max_tokens | Max output length | `8192` | Increase for long captions/prompts |

**Why these defaults?**  
Per [Unsloth Gemma-4 docs](https://unsloth.ai/docs/models/gemma-4):
- Temperature 1.0 for best quality on E4B
- Penalties at 1.0 to avoid over-penalizing creative/uncensored outputs (your Aggressive variant benefits from this)
- 32K context is responsive starting point
- Full GPU offload + Flash Attention for E4B Q4_K_M speed/quality

**Quantization Note**: Your Q4_K_M is excellent for speed/memory on E4B (Unsloth recommends Q8_0 for max quality, but Q4_K_M is aggressive and works great with your fork).

---

## Troubleshooting

### Issue: "Gemma4ChatHandler not found"

**Solution**: Ensure you're using the Randy420Marsh fork:

```bash
pip uninstall llama-cpp-python
pip install --no-cache-dir git+https://github.com/Randy420Marsh/llama-cpp-python.git
```

### Issue: CUDA compilation errors

**Solution**:

1. Verify CUDA toolkit is installed: `nvcc --version`
2. Set correct architecture: `export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89"`
3. Clean build cache: `pip cache purge`

### Issue: Out of memory (OOM) with E4B Q4_K_M

**Solution**:

- Reduce `n_ctx` (e.g., from 32768 to 16384 or 8192)
- Reduce `thinking_budget` to 4096
- Enable `unload_after_inference`
- Use smaller batch or ensure enough VRAM (E4B Q4_K_M ~6-8GB with full offload)

### Issue: Slow inference or poor vision quality

**Solution**:

- Enable Flash Attention + CUDA Graphs
- Increase `n_gpu_layers` to `-1`
- Use `n_batch=2048` or higher
- Verify mmproj matches your exact model (E4B f16 projector)
- For aggressive uncensored model: temperature 1.0 + thinking ON gives best detailed captions

### Issue: Thinking mode not working as expected

**Solution**: The node automatically prepends `<|think|>` when enabled (Unsloth format). Keep `enable_thinking=True` and adjust budget. Only the final visible answer is kept in history.

---

## Example Workflows

### Workflow 1: Detailed Image Captioning (E4B Optimized)

```
[Load Image] → [VisionLLMCaptioner] → [Save Text File]
                                      ↓
                              [Show Text]
```

**Recommended Settings**:
- Backend: Local Standalone (llama-cpp-python)
- Mode: Image Caption
- model_path: your E4B Q4_K_M path
- mmproj_path: matching E4B f16
- temperature: 1.0
- enable_thinking: True
- thinking_budget: 8192
- User Prompt: "Describe the scene in extreme detail, including lighting, composition, mood, and artistic style."

### Workflow 2: Text-to-Detailed Prompt (Aggressive Creative Mode)

- Mode: Text -> Detailed Image Prompt
- input_text: "epic fantasy warrior in glowing armor"
- temperature: 1.0 (Unsloth default for creativity)
- repeat_penalty: 1.0

---

## Credits & Links

- **Your llama-cpp-python fork**: https://github.com/Randy420Marsh/llama-cpp-python (includes Gemma4ChatHandler)
- **This node**: https://github.com/Randy420Marsh/VisionLLMCaptioner
- **Best Gemma-4 settings**: https://unsloth.ai/docs/models/gemma-4
- **Model source inspiration**: Unsloth Gemma-4 collection (your fine-tuned Aggressive variant)

For issues or contributions, open a GitHub issue on either repo.

**Happy captioning with Gemma-4 E4B!** 🚀
