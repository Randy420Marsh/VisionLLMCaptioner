# VisionLLMCaptioner - README

A powerful ComfyUI node that leverages **Gemma-4** vision capabilities for image captioning and prompt enhancement. Supports both **Remote API (llama-server)** and **Local Standalone (llama-cpp-python)** modes.

## ✨ Features

- **Dual Backend Support**: Switch between Remote API (OpenAI-compatible) and Local Standalone (your custom llama-cpp-python fork)
- **Multi-Image Analysis**: Process multiple images simultaneously with labeled references
- **Gemma-4 Vision**: Full support for Gemma-4's vision capabilities via `Gemma4ChatHandler`
- **Thinking Mode**: Enable chain-of-thought reasoning with configurable budget
- **Prompt Enhancement**: Transform simple ideas into detailed, cinematic image generation prompts
- **Memory Efficient**: Automatic model unloading and GPU cache management
- **CUDA Optimized**: Built for NVIDIA GPUs with Flash Attention and CUDA Graphs support

---

## 🚀 Installation

### Step 1: Install System Dependencies

Ensure you have the necessary build tools and CUDA toolkit installed:

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
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Compile and Install llama-cpp-python from Randy420Marsh Fork

This fork includes the critical `Gemma4ChatHandler` and vision support for Gemma-4 models.

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
| RTX 30xx   | 3060, 3070, 3080, 3090 | `86` |
| RTX 40xx   | 4070, 4080, 4090 | `89` |
| RTX 50xx   | 5070, 5080, 5090 | `90` |
| GTX 16xx   | 1650, 1660 | `75` |
| RTX 20xx   | 2060, 2070, 2080 | `75` |

Example for RTX 4090:
```bash
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
```

### Step 4: Verify Installation

```bash
python -c "from llama_cpp import Llama; from llama_cpp.llama_chat_format import Gemma4ChatHandler; print('✅ Gemma4ChatHandler available!')"
```

If this runs without errors, your installation is successful.

---

## 📦 ComfyUI Node Installation

### Option A: Manual Installation

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```

2. Clone or copy this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VisionLLMCaptioner.git
   ```

3. Restart ComfyUI.

### Option B: ComfyUI Manager

1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "VisionLLMCaptioner"
4. Click Install
5. Restart ComfyUI

---

## 🎯 Usage

### Basic Image Captioning

1. Add the **VisionLLMCaptioner** node to your workflow
2. Connect an image to the `image_1` input
3. Set **Backend** to `Local Standalone (llama-cpp-python)` or `Remote API (llama-server)`
4. Configure model paths (for local mode) or server URL (for remote mode)
5. Click "Queue Prompt"

### Text-to-Prompt Enhancement

1. Set **Mode** to `Text -> Detailed Image Prompt`
2. Enter your idea in the `input_text` field (e.g., "cyberpunk girl with neon hair")
3. Optionally connect reference images for style guidance
4. The node will expand your idea into a detailed, cinematic prompt

### Multi-Image Analysis

The node automatically detects all connected `image_*` inputs:
- Connect `image_1`, `image_2`, `image_3`, etc.
- The system will label them internally as `[image_1]`, `[image_2]`, etc.
- Reference specific images in your prompts for comparative analysis

---

## ⚙️ Configuration

### Local Standalone Mode

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `model_path` | Path to Gemma-4 GGUF model | `/path/to/gemma-4-E2B.gguf` |
| `mmproj_path` | Path to mmproj (vision projector) | `/path/to/mmproj-Gemma-4-f16.gguf` |
| `n_gpu_layers` | Layers offloaded to GPU (-1 = all) | `-1` |
| `n_ctx` | Context window size | `32768` |
| `attention_mode` | Flash Attention for speed | `Flash Attention (recommended)` |
| `cuda_graphs` | Enable CUDA Graphs (RTX 40/50 series) | `True` for 40/50xx |

### Remote API Mode

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `server_url` | llama-server endpoint | `http://127.0.0.1:8080/v1` |
| `model_name` | Model identifier on server | `gemma-4-E2B` |

### Generation Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `enable_thinking` | Chain-of-thought reasoning | `True` |
| `thinking_budget` | Max tokens for thinking | `8192` |
| `temperature` | Creativity vs determinism | `0.7` |
| `top_p` | Nucleus sampling | `0.95` |
| `top_k` | Top-K sampling | `64` |
| `repeat_penalty` | Reduce repetition | `1.05` |
| `presence_penalty` | Encourage new topics | `1.30` |

---

## 🔧 Troubleshooting

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

### Issue: Out of memory (OOM)

**Solution**:
- Reduce `n_ctx` (e.g., from 32768 to 16384)
- Reduce `thinking_budget`
- Enable `unload_after_inference`
- Use a smaller quantization (IQ3_M instead of F16)

### Issue: Slow inference

**Solution**:
- Enable `Flash Attention`
- Enable `CUDA Graphs` (RTX 40/50 series)
- Increase `n_gpu_layers` to `-1` (full offload)
- Use `n_batch=2048` or higher

---

## 📝 Example Workflows

### Workflow 1: Detailed Image Captioning

```
[Load Image] → [VisionLLMCaptioner] → [Save Text File]
                                      ↓
                              [Show Text]
```

**Settings**:
- Mode: `Image Caption`
- User Prompt: `Describe the scene in extreme detail, including lighting, composition, mood, and artistic style.`
- Enable Thinking: `True`

### Workflow 2: Prompt Enhancement

```
[Text Input: "fantasy castle"] → [VisionLLMCaptioner] → [CLIP Text Encode] → [KSampler]
```

**Settings**:
- Mode: `Text -> Detailed Image Prompt`
- Input Text: `fantasy castle at sunset`
- User Prompt: `Expand this into a photorealistic, cinematic prompt with detailed lighting, atmosphere, and composition.`

### Workflow 3: Multi-Image Comparison

```
[Load Image A] ─┐
                ├→ [VisionLLMCaptioner] → [Text Output]
[Load Image B] ─┘
```

**Settings**:
- Connect both images to `image_1` and `image_2`
- User Prompt: `Compare these two images. What are the key differences in style, composition, and subject matter?`

---

## 📄 License

This project is provided as-is for educational and research purposes. Please respect the licenses of:
- Gemma-4 model weights (Google)
- llama-cpp-python (MIT License)
- ComfyUI (GPL-3.0)

---

## 🙏 Credits

- **Gemma-4**: Google DeepMind
- **llama-cpp-python**: Original by @abetlen, Vision fork by @Randy420Marsh
- **ComfyUI**: ComfyOrg community

Special thanks to the Randy420Marsh fork for enabling Gemma-4 vision support in llama-cpp-python!

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

For issues related to llama-cpp-python vision support, please report to:
https://github.com/Randy420Marsh/llama-cpp-python

---

## 📬 Support

- **Issues**: Open a GitHub issue
- **Discussions**: GitHub Discussions tab
- **ComfyUI Community**: r/comfyui on Reddit

Happy prompting! 🎨✨
