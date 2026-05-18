# VisionLLMCaptioner – Gemma-4 Vision Captioner + Text Transformer for ComfyUI

A powerful ComfyUI node that leverages **Gemma-4** vision capabilities for image captioning, prompt enhancement, and text transformation. Supports three operation modes and both **Remote API (llama-server)** and **Local Standalone (llama-cpp-python)** backends.

**Optimized for best defaults from [Unsloth Gemma-4 docs](https://unsloth.ai/docs/models/gemma-4)**  
Recommended model: **Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf** + **mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf**

### Features
- **Three Operation Modes**: Image Caption, Text → Detailed Image Prompt, and Text to Text
- **Preset System**: Mode-aware presets for quick system prompt switching
- Dual Backend Support: Switch between Remote API (OpenAI-compatible) and Local Standalone (your custom llama-cpp-python fork)
- Multi-Image Analysis: Process multiple images simultaneously with labeled references
- Gemma-4 Vision: Full support for Gemma-4's vision capabilities via Gemma4ChatHandler (from your fork)
- Thinking Mode: Enable chain-of-thought reasoning with configurable budget (uses `<|think|>` per Unsloth)
- Prompt Enhancement: Transform simple ideas into detailed, cinematic image generation prompts
- Text Transformation: Rewrite, summarize, or transform input text based on instructions
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

### Mode Selection

The node supports three modes:

| Mode | Description | Required Input |
|------|-------------|----------------|
| **Image Caption** | Analyze images and generate detailed captions | One or more connected images |
| **Text → Detailed Image Prompt** | Expand short ideas into detailed image generation prompts | Input text in Prompt Enhance field |
| **Text to Text** | Transform text based on instructions (rewrite, summarize, translate, etc.) | Input text in Prompt Enhance field |

### Field Labels

The node uses clear, descriptive labels:

- **System Prompt** – Controls how the model behaves (use presets for quick switching)
- **Prompt** – Your specific instructions for the task
- **Prompt Enhance** – Input text to be processed (used in text modes)

### Preset System

The node features a **persistent preset system** for quick system prompt switching:

- **Built-in presets**: "Default" preset for each mode (cannot be deleted)
- **Custom preset**: Select "Custom" to write your own system prompt
- **User presets**: Save your own custom presets that persist across ComfyUI sessions

**Preset Management Buttons:**
- **Save Preset** – Saves the current system prompt as a new preset (prompts for a name)
- **Delete Preset** – Deletes the currently selected preset (except "Custom" and "Default")
- **Refresh Presets** – Reloads presets from the server (useful if presets.json was edited externally)

Presets are stored in `presets.json` in the node's directory and persist across ComfyUI restarts.

**Important Notes:**
- "Custom" and "Default" presets cannot be deleted
- Each mode has its own set of presets (switching modes shows only that mode's presets)
- Changing modes automatically switches to the "Default" preset for that mode

### Basic Image Captioning

1. Add the **VisionLLMCaptioner** node to your workflow
2. Connect an image to the image_1 input
3. Set **Mode** to "Image Caption"
4. Set **Backend** to Local Standalone (llama-cpp-python) or Remote API (llama-server)
5. Choose a **Preset** or edit the **System Prompt** directly
6. Enter your instructions in the **Prompt** field
7. Configure model paths (for local mode) or server URL (for remote mode)
8. Click "Queue Prompt"

### Text-to-Prompt Enhancement

1. Set **Mode** to "Text → Detailed Image Prompt"
2. Select a preset or configure the **System Prompt**
3. Enter your idea in the **Prompt Enhance** field (e.g., "cyberpunk girl with neon hair")
4. Enter specific instructions in the **Prompt** field
5. Optionally connect reference images for style guidance
6. The node will expand your idea into a detailed, cinematic prompt

### Text to Text Transformation

1. Set **Mode** to "Text to Text"
2. Select a preset or configure the **System Prompt** (e.g., "You are a helpful text editor.")
3. Enter your source text in the **Prompt Enhance** field
4. Enter transformation instructions in the **Prompt** field (e.g., "Summarize this in 3 bullet points")
5. The node will transform your text based on the instructions

**Example Text to Text use cases:**
- Summarization: "Summarize this text in 2-3 sentences"
- Rewriting: "Rewrite this in a more formal tone"
- Translation: "Translate this to Spanish"
- Formatting: "Format this as a markdown table"

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
- Penalties at 1.0 to avoid over-penalizing creative/uncensored outputs (with abliterated models)
- 32K context is responsive starting point
- Full GPU offload + Flash Attention for E4B Q4_K_M speed/quality

**Quantization Note**: Q4_K_M is excellent for speed/memory on E4B (Unsloth recommends Q8_0 for max quality, but Q4_K_M is aggressive and works great).

---

## Adding Custom Presets

You can add custom presets in two ways:

### Method 1: Using the UI (Recommended)

1. Select the desired **Mode**
2. Edit the **System Prompt** field with your desired content
3. Click the **Save Preset** button
4. Enter a name for your preset
5. The preset is saved and immediately available in the dropdown

### Method 2: Manual Editing (Advanced)

To manually add presets by editing files:

#### 1. Edit `presets.json`

Find the `presets.json` file in the node directory and add your presets:

```json
{
  "Image Caption": {
    "Custom": "",
    "Default": "...",
    "Technical Analysis": "Analyze this image from a technical perspective..."
  },
  "Text -> Detailed Image Prompt": {
    "Custom": "",
    "Default": "...",
    "Cinematic": "Create a cinematic image generation prompt..."
  },
  "Text to Text": {
    "Custom": "",
    "Default": "...",
    "Summarizer": "You are a skilled summarizer..."
  }
}
```

#### 2. Refresh Presets

Click the **Refresh Presets** button in the node UI to reload from the JSON file.

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
- Preset: Default (or Custom)
- model_path: your E4B Q4_K_M path
- mmproj_path: matching E4B f16
- temperature: 1.0
- enable_thinking: True
- thinking_budget: 8192
- Prompt: "Describe the scene in extreme detail, including lighting, composition, mood, and artistic style."

### Workflow 2: Text-to-Detailed Prompt (Aggressive Creative Mode)

```
[Primitive String] → [VisionLLMCaptioner] → [Show Text]
                         ↓
                    (Prompt Enhance input)
```

**Settings**:
- Mode: Text → Detailed Image Prompt
- Prompt Enhance: "epic fantasy warrior in glowing armor"
- Prompt: "Expand into a detailed prompt for image generation"
- temperature: 1.0 (Unsloth default for creativity)
- repeat_penalty: 1.0

### Workflow 3: Text Summarization

```
[Primitive String] → [VisionLLMCaptioner] → [Show Text]
                         ↓
              (Long text to summarize)
```

**Settings**:
- Mode: Text to Text
- Preset: Default (or Custom with: "You are a skilled summarizer.")
- Prompt Enhance: [paste your long text here]
- Prompt: "Summarize this in 3-5 sentences focusing on the main points."
- enable_thinking: True

### Workflow 4: Text Rewriting

**Settings**:
- Mode: Text to Text
- System Prompt: "You are a professional editor specializing in making text more engaging and readable."
- Prompt Enhance: [your original text]
- Prompt: "Rewrite this to be more concise while keeping all important information."

---

## Credits & Links

- **llama-cpp-python fork**: https://github.com/Randy420Marsh/llama-cpp-python (includes Gemma4ChatHandler)
- **This node**: https://github.com/Randy420Marsh/VisionLLMCaptioner
- **Best Gemma-4 settings**: https://unsloth.ai/docs/models/gemma-4
- **Model source inspiration**: Unsloth Gemma-4 collection (your fine-tuned Aggressive variant)

For issues or contributions, open a GitHub issue on either repo.

**Happy captioning with Gemma-4 E4B!** 🚀
