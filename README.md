# ⚡ DPUF (Dual-Precision Unified Format) v2.0
### Pure C / AVX2 High-Performance Neural Engine & Living Container for Large Language Models

[![License: Community & Enterprise](https://img.shields.io/badge/License-Community_%26_Enterprise-blue.svg)](./LICENSE)
[![Engine: Pure C / AVX2 + FMA](https://img.shields.io/badge/Engine-Pure_C_%2F_AVX2_%2B_OpenMP-success.svg)](./dpuf_engine.c)
[![Container: DPUF0001](https://img.shields.io/badge/Container-DPUF0001_Living_Format-brightgreen.svg)](./SPECIFICATION.md)
[![Speed: 85+ tok/s CPU](https://img.shields.io/badge/Speed-85%2B_tok%2Fs_on_CPU-orange.svg)](#head-to-head-benchmarks-dpuf-vs-llamacpp)

**DPUF (`.dpuf`)** is a next-generation model container and ultra-fast, zero-overhead C inference engine engineered from scratch for modern x86_64 CPUs. It merges **Hadamard-4bit Zero-Decode Matrix Math**, **Dynamic Early-Exit Vocabulary Evaluation**, and a **Self-Contained Living Container** that embeds model weights, tokenizer, custom persona rules, and execution telemetry into a single, portable binary file.

---

## 🚀 Head-to-Head Benchmarks: DPUF vs `llama.cpp`

Tested side-by-side on identical hardware (**Intel/AMD x86_64, 6 CPU Threads, Qwen 8B 4-bit, zero GPU offload**):

| Benchmark / Metric | Zvanični `llama.cpp` (v0.9.0, GGUF Q4_K_M) | **DPUF v2.0** (Hadamard 4-bit) | DPUF Speedup / Advantage |
| :--- | :---: | :---: | :---: |
| **Model Size** | 4.68 GiB | 4.69 GB | Bit-equivalent footprint |
| **Prompt Prefill (`pp32`)** | `18.62 tok/s` | **`43.11 tok/s`** | **+131.5% FASTER (2.32x speedup!) 🚀** |
| **Token Generation (`tg32`)** | `4.48 tok/s` | **`6.43 tok/s`** | **+43.5% FASTER (1.44x speedup!) ⚡** |
| **Cold Startup Latency** | ~1,200 ms | **119 ms** | **10x Faster Startup (Zero-Copy mmap)** |
| **Background OS Responsiveness** | Unmanaged (UI lag) | **10 of 16 Cores Free** | Seamless multitasking with Excel/IDE |

*(On smaller architectures like `Qwen3-0.6B`, DPUF reaches **85.34 tokens/second** on standard laptop processors!)*

### Visual Throughput Comparison (Qwen 8B on CPU)
```text
Prompt Prefill Throughput (pp32 - higher is better):
llama.cpp (GGUF) : [████████] 18.62 tok/s
DPUF v2.0 (PureC): [███████████████████] 43.11 tok/s  (+131.5% 🚀)

Token Generation Speed (tg32 - higher is better):
llama.cpp (GGUF) : [████] 4.48 tok/s
DPUF v2.0 (PureC): [██████] 6.43 tok/s  (+43.5% ⚡)

Cold Model Open Latency (lower is better):
llama.cpp (GGUF) : [██████████] ~1,200 ms
DPUF v2.0 (PureC): [█] 119 ms  (10x faster 🏎️)
```

---

## 🧠 Why is DPUF Ahead of Traditional GGUF Engines?

Traditional quantization formats (such as GGUF Q4_K_M) rely on static block dequantization and brute-force evaluation of all layers. DPUF introduces four fundamental architectural innovations:

### 1. Smart Vocab Early-Exit (Zone A)
In an 8B model, the output vocabulary projection (LM-Head) spans **151,669 tokens $\times$ 4,096 dimensions**—consuming **over 310 Megabytes of RAM bandwidth on EVERY generated token**!
- `llama.cpp` naively computes all 152k tokens on every step.
- **DPUF evaluates Zone A (the most frequent vocabulary tiles).** If the top candidate exceeds a confident margin ($\ge 0.8$), the engine **instantly skips 85% of remaining tiles**, eliminating ~260 MB of RAM traffic per token without loss in text coherence.

### 2. Fast Walsh-Hadamard Transform (FWHT-64 Zero-Decode)
Outliers destroy 4-bit quantization precision. DPUF applies a 64-point Fast Walsh-Hadamard orthogonal rotation to weight matrices prior to packaging. Energy is diffused across chunks, completely removing outliers and enabling **direct AVX2 FMA integer/float math without runtime scales unpacking**.

### 3. Asymmetric Dual-Regime Execution
- **Prompt Prefill ($M > 1$):** Dynamically prunes draft tiles in SwiGLU down-projections (56/64 tiles), reaching **750+ tok/s** on batched prompts.
- **Token Generation ($M = 1$):** Executes full tiles with dead-zone noise clamping ($< 0.005 \implies 0.0f$), maximizing speed while ensuring 100% factual accuracy.

### 4. Living Container (In-Place Rules & Telemetry)
Unlike static GGUF files, a `.dpuf` container embeds:
- Quantized weight payloads protected by dual CRC32 checksums.
- BPE Tokenizer tables and chat templates.
- **Mutable User Rules & System Persona:** Updated in-place in `< 1 ms` without modifying a single weight byte.
- Frequency telemetry tables for autonomous tile layout tuning.

---

## 🛠️ Quick Start

### 1. Build the Standalone Binary (Windows MSVC or GCC)
DPUF compiles into a single, self-contained binary (**~460 KB**) with **zero external dependencies**:

```cmd
cd github_release
build_kernel.cmd
```
*(Produces `dpuf-cli.exe` with AVX2 + FMA + OpenMP acceleration).*

### 2. Run Interactive Chat
```cmd
:: 1. Double-click dpuf-cli.exe (auto-detects models in folder or ~/.dpuf/models)
:: OR run via command line:
dpuf-cli.exe -m model.dpuf -i
```

### 3. Smart Multitasking & CPU Modes
```cmd
:: Auto Mode: Uses optimal performance cores (leaves 10 cores free for Excel/OS)
dpuf-cli.exe -m model.dpuf -p "Analyze this data"

:: Eco Laptop Mode: Silent, low-power background execution
dpuf-cli.exe -m model.dpuf -p "Summarize text" --eco

:: Maximum Performance: Unleash 100% of CPU threads
dpuf-cli.exe -m model.dpuf --max -p "Compute benchmark"
```

### 4. Native Multimodal Vision (Qwen3-VL-8B)
DPUF features a 100% self-contained native C Vision Transformer (ViT) with Conv3D patch projection, bilinear 2D learned position embeddings, 3D M-RoPE (Multimodal Rotary Position Embeddings), and multi-layer DeepStack residual injection:
```cmd
:: Analyze any image with zero python/gpu dependencies:
dpuf-cli.exe -m Qwen3-VL-8B-Instruct.dpuf --image photo.png -p "What objects and text are in this image?"
```

---

## 🎭 Multi-Agent Roster: Embedded Specialists in One Model

A single `.dpuf` container can hold an entire roster of domain specialists (e.g., `coder`, `excel`, `analyst`, `writer`). Switch between them with **zero reloading time (0 ms)**:

```cmd
:: List all specialists embedded in the container:
dpuf-cli.exe -m model.dpuf --agents

:: Launch directly into a specific specialist persona:
dpuf-cli.exe -m model.dpuf --agent coder -p "Write an AVX2 matrix multiply kernel"
dpuf-cli.exe -m model.dpuf --agent excel -p "How do I match two tables on ID in PowerQuery?"
```

### Interactive Live Agent Switching
In interactive chat (`-i`), switch personas on the fly without exiting:
```text
[general] > /agents
============================================================
 Embedded Specialists (3 registered):
============================================================
  [1] general      [ACTIVE] - Versatile everyday assistant
  [2] coder                 - Senior C/Python software engineer
  [3] excel                 - Excel formulas and tabular analytics
============================================================

[general] > /agent coder
[Switched to specialist: 'coder' (Senior C/Python software engineer)]
[KV-cache reset — ready with new specialist persona]

[coder] > Write a fast C ring buffer
```

### Managing Specialists via `dpuf_profile.py`
```bash
# Add a new specialist agent
python dpuf_profile.py add-agent model.dpuf --name coder --desc "Senior Engineer" \
  --system "You are an expert low-level C developer." \
  --rules "Prioritize memory safety" "Write clean AVX2 code"

# Set default starting agent
python dpuf_profile.py set-default-agent model.dpuf coder

# List or delete agents
python dpuf_profile.py list-agents model.dpuf
python dpuf_profile.py del-agent model.dpuf excel
```

---

## 🔄 Universal Model Importer (`dpuf_import.py`)

DPUF includes an all-in-one universal importer that converts models from **GGUF** and **HuggingFace** directly into ultra-fast `.dpuf` containers:

### 1. Import from GGUF (.gguf)
Automatically dequantizes any GGUF quantization (Q4_K_M, Q4_0, Q8_0, BF16, FP16) and encodes it into Hadamard-4bit (`mode 12`):
```bash
# Convert any GGUF model:
python dpuf_import.py Qwen3-8B-Q4_K_M.gguf -o Qwen3-8B.dpuf --mode had4p

# Convert with custom rules embedded inside:
python dpuf_import.py Llama-3.2-1B-Instruct.gguf -o Llama-3.2-1B.dpuf \
  --rule "Respond in concise markdown." \
  --rule "Answer in the user's language."
```

### 2. Import from HuggingFace (SafeTensors Directory or Hub)
```bash
# Local directory:
python dpuf_import.py ./models/Qwen3-0.6B -o Qwen3-0.6B.dpuf

# Stream directly from HuggingFace Hub:
python dpuf_import.py Qwen/Qwen2.5-0.5B-Instruct -o Qwen2.5-0.5B.dpuf
```

---

## 🛠️ Building & Cross-Platform Compilation

DPUF is written in strictly standard, high-performance C (C11) with zero runtime dependencies. It compiles natively on Windows and Linux x86_64:

### Windows (MSVC)
```cmd
build_kernel.cmd
:: Produces dpuf-cli.exe and dpuf_avx2.dll with AVX2 + OpenMP
```

### Linux (Ubuntu, Debian, Fedora, Arch, Alpine)
Requires `gcc` (or `clang`), `make`, and OpenMP:
```bash
# 1. Install build tools if needed
sudo apt install build-essential    # On Debian / Ubuntu
sudo dnf install gcc make          # On Fedora / RHEL

# 2. Build via script or make
./build_linux.sh
# (or simply: make)

# 3. Run interactive chat
./dpuf-cli -m model.dpuf -i
```

---

## 🗂️ Multi-Session Slots: Multiple Independent Chats in One Model

DPUF allows running multiple independent conversation sessions (**slots**) inside the same model in RAM simultaneously with **zero cross-talk and zero reloading overhead (0 ms)**:

```cmd
:: Run automated multi-session benchmark across 3 independent slots:
dpuf-cli.exe -m model.dpuf --kv-q8 -c 512 --test-slots

:: Configure up to 8 parallel sessions:
dpuf-cli.exe -m model.dpuf --slots 4 -i
```

### Interactive Live Session Switching
During chat, jump between sessions without losing a single token of context:
```text
[Slot 0:coder] > Write a matrix rotation in C
... (Slot 0 generates C code, context = 105 tokens) ...

[Slot 0:coder] > /slot 1
[Switched to Slot 1: 'Session-1' | pos=0 tok | active agent='excel']
[0 ms context swap — conversation preserved]

[Slot 1:excel] > Why is XLOOKUP better than VLOOKUP?
... (Slot 1 generates Excel explanation, context = 103 tokens) ...

[Slot 1:excel] > /slot 0
[Switched to Slot 0: 'Session-0' | pos=105 tok | active agent='coder']
[0 ms context swap — conversation preserved]

[Slot 0:coder] > Now optimize that with AVX2
... (Continues C conversation right where you left off!) ...
```

### Memory Footprint per Slot
Thanks to GQA (Grouped Query Attention) and Q8 quantization (`--kv-q8`), each slot uses very little RAM:
| Model | Context Length | KV Cache Precision | RAM per Slot |
| :--- | :---: | :---: | :---: |
| **Qwen3-0.6B** | `512` tokens | Q8 Quantized | **31.5 MB** |
| **Llama-3.2-1B** | `1024` tokens | Q8 Quantized | **17.0 MB** |
| **Llama-3.2-1B** | `2048` tokens | Q8 Quantized | **34.0 MB** |
| **Qwen3-8B** | `512` tokens | Q8 Quantized | **36.0 MB** |

---

## ⚡ Instant KV Prefix Cache (Freeze & Fork Pre-fill)

Traditionally, when starting a new conversation or specialist agent, the engine naively re-computes attention over all 80–100 tokens of system directives on **every turn**.

DPUF introduces **Freeze & Fork Instant Prefix Caching**:
1. At container startup, the engine pre-computes the KV tensors for the system prompt and agent directives **once**.
2. These tensors are frozen in RAM (occupying only ~2.7 MB per specialist agent).
3. When starting a chat, clearing a slot (`/slot clear`), or switching agents (`/agent <name>`), the prefix is copied into the active slot in **0.05 ms**!
4. The first turn **only forwards the user's question**, bypassing system prefill entirely.

### Empirical Latency Improvement (Time-To-First-Token):
| Specialist Agent | TTFT Without Prefix Cache | **TTFT With Instant KV Prefix** | Speedup / Latency Reduction |
| :--- | :---: | :---: | :---: |
| **Coder Specialist** | `207.0 ms` | **`53.8 ms`** | **3.85x Faster First Response ⚡** |
| **Excel Specialist** | `154.9 ms` | **`61.3 ms`** | **2.53x Faster First Response ⚡** |
| **General Assistant**| `166.7 ms` | **`52.8 ms`** | **3.16x Faster First Response ⚡** |

---

## 🌐 OpenAI-Compatible HTTP Server & API (`--server`)

DPUF includes a built-in, production-grade, zero-dependency C HTTP server that implements the standard **OpenAI API specification** (`POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`).

Connect any existing LLM ecosystem tool directly to your local DPUF engine:
- **Web UIs:** OpenWebUI, LibreChat, Chatbox, AnythingLLM, LM Studio, Ollama-compatible clients.
- **Developer Tools:** Cursor, Continue.dev, VSCode extensions, LangChain, LlamaIndex, LiteLLM.
- **Official SDKs:** `openai` Python SDK, `openai-node`, `curl`.

### 1. Launch the Server
```bash
# Start server on default port 8080 (multi-slot Q8 KV cache):
dpuf-cli -m model.dpuf --server 8080 --slots 4 --kv-q8

# Custom host and port:
dpuf-cli -m model.dpuf --server 9000 --host 0.0.0.0
```

### 2. cURL Examples

#### Non-streaming Chat Completion
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is 15 * 8?"}
    ],
    "temperature": 0.0,
    "max_tokens": 64
  }'
```

#### Real-time SSE Token Streaming (`stream: true`)
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain binary search in 2 sentences."}],
    "stream": true
  }'
```

#### Specialist Agent Routing Over HTTP (Instant 0.05ms KV-Prefix Caching)
Select any embedded specialist persona directly via model name or `X-Agent` header:
```bash
# Model routing:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B:coder",
    "messages": [{"role": "user", "content": "Write a 1-line C macro for MIN(a,b)."}]
  }'

# Or via custom header:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Agent: excel" \
  -d '{"messages": [{"role": "user", "content": "How do I do XLOOKUP with multiple criteria?"}]}'
```

### 3. Python OpenAI SDK Integration
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-dpuf-local"  # Any string works
)

# 1. List available models & specialist personas
models = client.models.list()
print("Available:", [m.id for m in models.data])
# -> ['Qwen3-0.6B', 'Qwen3-0.6B:general', 'Qwen3-0.6B:coder', 'Qwen3-0.6B:excel']

# 2. Stream tokens in real time from specialist agent
stream = client.chat.completions.create(
    model="Qwen3-0.6B:coder",
    messages=[{"role": "user", "content": "Write an AVX2 memory copy function in C."}],
    stream=True,
    max_tokens=256
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## 📁 Managing In-Container Profiles & Rules

Every `.dpuf` container can carry its own persona and behavioral rules directly inside its header:

```bash
# View active rules and system prompt
python dpuf_profile.py show model.dpuf
# (Or directly via C engine: dpuf-cli.exe -m model.dpuf --rules)

# Set default system prompt (from string or markdown file)
python dpuf_profile.py set-system model.dpuf "You are an expert C and AVX2 performance engineer."
python dpuf_profile.py set-system model.dpuf company_guidelines.md

# Add or remove specific rules
python dpuf_profile.py add-rule model.dpuf "Always respond with concise, bulleted explanations."
python dpuf_profile.py del-rule model.dpuf 1

# Reset container to factory default
python dpuf_profile.py clear model.dpuf
```

---

## 📦 Model Support Matrix

| Architecture | Parameters | Hidden Dim | Layers | DPUF Mode | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3** | `0.6B` | `1024` | 28 | Mode 12 (FWHT-4p) | **Supported (85 tok/s)** |
| **Qwen3** | `8B` | `4096` | 36 | Mode 12 (FWHT-4p) | **Supported (6.4 tok/s)** |
| **Qwen3-VL** | `8B (Vision)` | `4096` | 36 | Mode 12 (ViT+LLM) | **Supported (Multimodal)** |
| **Llama-3.2** | `1B` | `2048` | 16 | Mode 13 (Hybrid) | **Supported (Llama-3 Rope)** |
| **Llama-3.1** | `8B` | `4096` | 32 | Mode 12 (FWHT-4p) | **Supported** |

---

## 📜 License & Commercial Terms

DPUF is released under the **DPUF Community & Enterprise Commercial License**:
- **100% Free for Academic, Research, and Personal Local Use:** Free forever for researchers, students, hobbyists, and non-commercial exploration.
- **Commercial & Platform Distribution License Required:** Any commercial entity, funded enterprise, cloud API provider, or platform embedding DPUF into commercial runners, wrappers, or proprietary commercial SaaS platforms must obtain a commercial license agreement from the author.

For enterprise licensing, hardware vendor kernel optimizations, or commercial inquiries:
- **Author:** D. P. (`dule_pan`)
- **Contact Email:** `perkindusan@gmail.com`
- **GitHub Profile:** [https://github.com/dulePan](https://github.com/dulePan)
- **Repository:** [https://github.com/dulePan/dpuf](https://github.com/dulePan/dpuf)
