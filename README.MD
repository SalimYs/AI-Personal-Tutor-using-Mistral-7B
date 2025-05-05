# AI-Powered Personal Tutor

This repository provides a step-by-step guide to set up an AI-powered personal tutor chatbot using the Mistral 7B Instruct v0.3 model, optimized for an RTX  with minimal VRAM usage (\~4 GB).

---

## 📋 Prerequisites

* **Windows 10/11** with WSL 2 installed
* **GPU**: NVIDIA RTX  (with at least 4 GB VRAM)
* **WSL 2** enabled and configured for GPU passthrough
* **Python 3.8+** installed in WSL

---

## ⚙️ Environment Setup

### 1. Install WSL 2 (if not already)

```powershell
# Open PowerShell as Administrator
wsl --install
```

### 2. Launch WSL and Update

```bash
# In WSL Ubuntu terminal
sudo apt update && sudo apt upgrade -y
```

### 3. Create & Activate Python Virtual Environment

```bash
python3 -m venv ~/mistral-tutor-env
source ~/mistral-tutor-env/bin/activate
python -m pip install --upgrade pip
```

### 4. Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
```

### 5. Install Dependencies

```bash
pip install transformers accelerate huggingface_hub einops safetensors bitsandbytes gradio
```

---

## 🧠 Model Setup

### Option 1: Automatic Download

The first time you run the app, the Mistral 7B Instruct v0.3 model will be fetched automatically and cached under `~/.cache/huggingface/`.

### Option 2: Manual Download

```bash
# Authenticate with Hugging Face
huggingface-cli login

git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
```

---

## 🚀 Running the Chatbot

1. Place the `chatbot_app.py` file in your working directory.
2. Activate the virtual environment:

   ```bash
   source ~/mistral-tutor-env/bin/activate
   ```
3. Launch the app:

   ```bash
   python chatbot_app.py
   ```
4. Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser.

---

## 🛠️ Configuration & Tuning

Edit the top of `chatbot_app.py` to adjust hyperparameters:

```python
MAX_NEW_TOKENS = 512         # Maximum tokens per response
MAX_HISTORY_LENGTH = 6       # Conversation turns to retain
TEMPERATURE = 0.7            # Sampling temperature
TOP_P = 0.9                  # Nucleus sampling
```

### GPU Memory Management

* Use 4‑bit quantization via `bitsandbytes` to reduce VRAM.

* Clear cache after each generation:

  ```python
  import torch
  torch.cuda.empty_cache()
  ```

* Monitor GPU usage:

  ```bash
  watch -n 1 nvidia-smi
  ```

---

## 🔧 Troubleshooting

| Issue             | Solution                             |
| ----------------- | ------------------------------------ |
| CUDA not detected | 1. Install NVIDIA drivers in Windows |

2. Ensure WSL 2 GPU support is enabled
3. Run `nvidia-smi` in WSL
   |
   \| Out of memory errors        | 1. Lower `MAX_NEW_TOKENS`
4. Decrease `MAX_HISTORY_LENGTH`
5. Stop other GPU processes
   |
   \| Slow response               | 1. Set `do_sample=False`
6. Reduce `TEMPERATURE` & `TOP_P`
   |

---

## 📖 Model Details

* **Name**: Mistral 7B Instruct v0.3
* **Parameters**: \~7B (quantized to \~2 GB with 4-bit)
* **Context window**: 8,192 tokens
* **Use case**: Instruction-following, conversational AI

---

## 🎨 UI

This project uses [Gradio](https://gradio.app) for the web interface. Modify `chatbot_app.py` to customize the UI:

```python
import gradio as gr
# e.g., change theme or layout
```

---

## 📚 References

* [Mistral AI on Hugging Face](https://huggingface.co/mistralai)
* [BitsAndBytes Documentation](https://github.com/facebookresearch/bitsandbytes)
* [Gradio Documentation](https://gradio.app)

---

## 📝 License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
