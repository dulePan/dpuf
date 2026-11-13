#!/usr/bin/env python3
"""
dpuf_import.py — Universal Model Importer for DPUF (Deep Packed Unified Format)

Converts models from multiple source formats into self-contained, ultra-fast .dpuf files:
  1. GGUF models (.gguf) — Supports Q4_K_M, Q4_0, Q8_0, BF16, FP16, etc.
  2. HuggingFace SafeTensors directories — Local directories containing config.json & safetensors
  3. HuggingFace Hub repositories — Streamed directly from huggingface.co

Usage Examples:
  # Import from GGUF file:
  python dpuf_import.py model.gguf -o model.dpuf --mode had4p

  # Import from local HuggingFace directory:
  python dpuf_import.py ./models/Qwen3-0.6B -o Qwen3-0.6B.dpuf

  # Import with custom base rules:
  python dpuf_import.py model.gguf -o model.dpuf --rule "Be concise" --rule "Use markdown tables"
"""

import argparse
import gc
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time
import zlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Ensure local modules can be found
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import dpuf_format as dpuf
import pack_dpuf


def get_gguf_field_val(field: Any) -> Any:
    """Helper to extract native python value from GGUFReader field."""
    if field is None:
        return None
    try:
        # Array of parts or single part
        if hasattr(field, "parts") and hasattr(field, "data"):
            part_idx = field.data[0]
            val = field.parts[part_idx]
            if isinstance(val, (bytes, memoryview)):
                try:
                    return bytes(val).decode("utf-8")
                except Exception:
                    return bytes(val)
            if hasattr(val, "tolist"):
                return val.tolist()
            return val
    except Exception:
        pass
    return getattr(field, "value", None)


def import_from_gguf(
    gguf_path: pathlib.Path,
    out_path: pathlib.Path,
    mode: str = "had4p",
    step_mult: float = 0.255,
    embed_mult: float = 0.75,
    ecc_group: int = 64,
    tile_rows: int = 16,
    base_rules: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Imports any GGUF file directly into a .dpuf model container."""
    try:
        import gguf
    except ImportError:
        raise RuntimeError("Importing GGUF files requires 'gguf' package. Install with: pip install gguf")

    t0 = time.perf_counter()
    print(f"\n============================================================")
    print(f" DPUF Universal Importer: GGUF -> .dpuf")
    print(f" Source: {gguf_path} ({gguf_path.stat().st_size / (1024**2):.1f} MB)")
    print(f" Target: {out_path} [Mode: {mode}]")
    print(f"============================================================")

    reader = gguf.GGUFReader(str(gguf_path), mode="r")

    # Detect architecture and hyperparameters
    arch_field = reader.fields.get("general.architecture")
    arch = "qwen2"
    if arch_field:
        raw_arch = get_gguf_field_val(arch_field)
        if isinstance(raw_arch, str):
            arch = raw_arch.lower()

    print(f"  Detected Architecture: {arch}")

    def f_int(key: str, default: int = 0) -> int:
        f = reader.fields.get(key)
        if f is None:
            return default
        val = get_gguf_field_val(f)
        return int(val) if val is not None else default

    def f_float(key: str, default: float = 0.0) -> float:
        f = reader.fields.get(key)
        if f is None:
            return default
        val = get_gguf_field_val(f)
        return float(val) if val is not None else default

    # Extract architecture dimensions
    hidden_size = f_int(f"{arch}.embedding_length", 1024)
    intermediate_size = f_int(f"{arch}.feed_forward_length", 2816)
    num_layers = f_int(f"{arch}.block_count", 28)
    num_heads = f_int(f"{arch}.attention.head_count", 16)
    num_kv_heads = f_int(f"{arch}.attention.head_count_kv", 8)
    head_dim = f_int(f"{arch}.attention.key_length", hidden_size // max(1, num_heads))
    max_seq_len = f_int(f"{arch}.context_length", 2048)
    rope_theta = f_float(f"{arch}.rope.freq_base", 1000000.0)
    rms_norm_eps = f_float(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)

    # Detect chat format
    chat_format = "chatml"
    if "llama" in arch:
        chat_format = "llama3"

    print(f"  Specs: layers={num_layers}, hidden={hidden_size}, ffn={intermediate_size}, heads={num_heads}/{num_kv_heads}, ctx={max_seq_len}")

    # Build DPUF model config
    model_config = {
        "architecture": "Qwen3ForCausalLM" if "qwen" in arch else "LlamaForCausalLM",
        "chat_format": chat_format,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_seq_len,
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "use_neox_rope": True,
        "has_qk_norm": False,
    }

    # Extract vocabulary tokens
    tokens_list: List[str] = []
    tok_field = reader.fields.get("tokenizer.ggml.tokens")
    if tok_field:
        raw_toks = get_gguf_field_val(tok_field)
        if isinstance(raw_toks, list):
            tokens_list = [t.decode("utf-8", errors="replace") if isinstance(t, bytes) else str(t) for t in raw_toks]

    if not tokens_list:
        # Fallback dummy vocab if missing
        tokens_list = [f"<tok_{i}>" for i in range(151669 if "qwen" in arch else 128256)]

    vocab_size = len(tokens_list)
    model_config["vocab_size"] = vocab_size

    # Check for tie_word_embeddings
    tensor_names = [t.name for t in reader.tensors]
    tie_embd = ("output.weight" not in tensor_names) and ("token_embd.weight" in tensor_names)
    model_config["tie_word_embeddings"] = tie_embd

    # Setup profile metadata
    base_meta: Dict[str, Any] = {
        "format": "DPUF0001",
        "container_version": "2.0",
        "model_architecture": model_config["architecture"],
        "source_format": "GGUF",
        "source_file": gguf_path.name,
        "mode": mode,
        "profile": {
            "name": gguf_path.stem,
            "desc": f"Universal DPUF container converted from {gguf_path.name}",
            "system_prompt": system_prompt or "You are a helpful, precise AI assistant.",
            "rules": base_rules or [
                "Respond accurately and concisely.",
                "Use GitHub-flavored markdown formatting.",
                "Respond in the language used by the user."
            ],
            "default_agent": "general",
            "agents": [
                {
                    "name": "general",
                    "desc": "General Knowledge & Creative Problem Solver",
                    "system_prompt": "You are a knowledgeable and versatile assistant capable of solving wide-ranging tasks.",
                    "rules": ["Be balanced, thorough, and polite."]
                },
                {
                    "name": "coder",
                    "desc": "Elite Software Engineer & Systems Architect",
                    "system_prompt": "You are an expert systems software engineer specializing in clean, high-performance algorithms.",
                    "rules": ["Write idiomatic, bug-free code with clear comments.", "Optimize for time and memory complexity."]
                },
                {
                    "name": "excel",
                    "desc": "Data Analysis & Spreadsheet Expert",
                    "system_prompt": "You are a financial and data analytics expert specializing in Excel, formulas, and data analysis.",
                    "rules": ["Provide exact Excel/Google Sheets formula syntax.", "Explain step-by-step table operations."]
                }
            ]
        },
        "tokenizer": {
            "chat_format": chat_format,
            "vocab_size": vocab_size,
            "tokens": tokens_list,
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>"},
                {"id": 151644, "content": "<|im_start|>"},
                {"id": 151645, "content": "<|im_end|>"},
            ] if "qwen" in arch else [
                {"id": 128000, "content": "<|begin_of_text|>"},
                {"id": 128001, "content": "<|end_of_text|>"},
                {"id": 128009, "content": "<|eot_id|>"},
            ],
            "eos_token_ids": [151645, 151643] if "qwen" in arch else [128001, 128009],
        }
    }

    # Encode tensors into DPUF payload
    hdr_slot = 4 * 1024 * 1024  # 4 MB reserved header
    tensor_index: Dict[str, List[int]] = {}
    offset = 0
    raw_payload_bytes = 0
    total_params = 0
    pay_crc = 0
    has_q_norm = False
    n_expected = len(reader.tensors)

    print(f"\n  Encoding {n_expected} tensors into DPUF {mode} format ...")

    with open(out_path, "wb") as f:
        # Pre-allocate header space
        f.write(b"\x00" * (24 + hdr_slot))

        for idx, t in enumerate(reader.tensors):
            tname = t.name
            if tname == "output.weight" and tie_embd:
                continue
            if "attn_q_norm" in tname:
                has_q_norm = True

            # Dequantize GGUF tensor to float32
            raw_data = t.data
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                f32_arr = np.frombuffer(raw_data, dtype=np.float32)
            elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                f16_arr = np.frombuffer(raw_data, dtype=np.float16)
                f32_arr = f16_arr.astype(np.float32)
            elif t.tensor_type == gguf.GGMLQuantizationType.BF16:
                u16_arr = np.frombuffer(raw_data, dtype=np.uint16)
                f32_arr = (u16_arr.astype(np.uint32) << 16).view(np.float32)
            else:
                # Quantized GGUF tensor (Q4_K, Q4_0, Q8_0, etc.)
                f32_arr = gguf.quants.dequantize(raw_data, t.tensor_type)

            # Reshape based on GGUF shape (reverse for numpy row-major)
            if len(t.shape) == 2:
                f32_arr = f32_arr.reshape((t.shape[1], t.shape[0]))
            elif len(t.shape) == 1:
                f32_arr = f32_arr.reshape((t.shape[0],))

            # Convert to uint16 BF16 representation for DPUF encoders
            u16_bf16 = (f32_arr.view(np.uint32) >> 16).astype(np.uint16)
            total_params += int(u16_bf16.size)

            # Encode tensor into DPUF format
            seed_name = pack_dpuf._hadamard_seed_name(tname)
            blob = pack_dpuf._encode_one_tensor(tname, u16_bf16, mode, seed_name, step_mult, embed_mult)
            del f32_arr
            del u16_bf16

            pad_n = dpuf._pad64(offset)
            if pad_n:
                pad_b = b"\x00" * pad_n
                f.write(pad_b)
                pay_crc = zlib.crc32(pad_b, pay_crc)
                offset += pad_n

            tensor_index[tname] = [offset, len(blob)]
            f.write(blob)
            pay_crc = zlib.crc32(blob, pay_crc)
            offset += len(blob)
            raw_payload_bytes += len(blob)
            del blob

            if (idx + 1) % 30 == 0 or (idx + 1) == n_expected:
                print(f"    [{idx+1}/{n_expected}] {tname} -> {offset / (1024**2):.1f} MiB ({time.perf_counter() - t0:.1f}s)", flush=True)

        pay_crc &= 0xFFFFFFFF
        model_config["has_qk_norm"] = has_q_norm
        bpw = (raw_payload_bytes * 8.0) / float(total_params) if total_params else 0.0
        base_meta["num_tensors"] = len(tensor_index)
        base_meta["payload_bytes"] = offset
        base_meta["total_parameters"] = total_params
        base_meta["bits_per_weight"] = round(bpw, 3)
        base_meta["model_config"] = model_config

        hdr_json_bytes = json.dumps({"__metadata__": base_meta, "__tensors__": tensor_index}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(hdr_json_bytes) > hdr_slot:
            raise RuntimeError(f"Reserved header slot {hdr_slot} too small for {len(hdr_json_bytes)} bytes")
        hdr_json_bytes += b" " * (hdr_slot - len(hdr_json_bytes))
        hdr_crc = zlib.crc32(hdr_json_bytes) & 0xFFFFFFFF
        top_hdr = struct.pack("<8sIIII", dpuf.MAGIC_DPUF, len(tensor_index), hdr_crc, pay_crc, len(hdr_json_bytes))

        f.seek(0)
        f.write(top_hdr)
        f.write(hdr_json_bytes)

    dt = time.perf_counter() - t0
    total_bytes = 24 + len(hdr_json_bytes) + offset
    res = {
        "out_path": str(out_path),
        "mib": total_bytes / (1024.0 * 1024.0),
        "bpw": bpw,
        "total_params": total_params,
        "tensors": len(tensor_index),
        "pack_s": dt,
    }
    print(f"\n============================================================")
    print(f" SUCCESS: Converted GGUF to DPUF Container!")
    print(f"  Container: {out_path} ({res['mib']:.2f} MiB, {res['bpw']:.3f} bits/param)")
    print(f"  Tensors: {res['tensors']} | Total Parameters: {res['total_params']:,}")
    print(f"  Conversion Time: {res['pack_s']:.1f} seconds")
    print(f"============================================================")
    return res


def main():
    ap = argparse.ArgumentParser(
        description="DPUF Universal Model Importer — Convert GGUF or HuggingFace models to .dpuf"
    )
    ap.add_argument("source", type=str, help="Source path (.gguf file or HuggingFace directory or repo ID)")
    ap.add_argument("-o", "--out", type=pathlib.Path, required=True, help="Output .dpuf file path")
    ap.add_argument("--mode", default="had4p", choices=["had4p", "had4p-hybrid", "had4", "split4", "hybrid", "q8"],
                    help="Quantization mode (default: had4p / Mode 12)")
    ap.add_argument("--max-seq-len", default=2048, type=int, help="Override context length")
    ap.add_argument("--step-mult", default=0.255, type=float, help="Step multiplier (default: 0.255)")
    ap.add_argument("--embed-mult", default=0.75, type=float, help="Embedding step multiplier (default: 0.75)")
    ap.add_argument("--system", default=None, type=str, help="Base system prompt")
    ap.add_argument("--rule", action="append", dest="rules", help="Add global rule (repeatable)")
    ap.add_argument("--hf-repo", default=None, type=str, help="HuggingFace repo ID if streaming shards")
    ap.add_argument("--delete-shards", action="store_true", help="Delete downloaded HF shards after packing")
    args = ap.parse_args()

    src_path = pathlib.Path(args.source)

    if src_path.is_file() and src_path.suffix.lower() == ".gguf":
        import_from_gguf(
            src_path,
            args.out,
            mode=args.mode,
            step_mult=args.step_mult,
            embed_mult=args.embed_mult,
            base_rules=args.rules,
            system_prompt=args.system,
        )
    elif src_path.is_dir():
        pack_dpuf.pack_hf_model(
            src_path,
            args.out,
            mode=args.mode,
            max_seq_len=args.max_seq_len,
            step_mult=args.step_mult,
            embed_mult=args.embed_mult,
            hf_repo=args.hf_repo,
            delete_shards=args.delete_shards,
        )
    else:
        # Assume it's a HuggingFace repo ID
        print(f"Streaming from HuggingFace repository: {args.source} ...")
        tmp_dir = pathlib.Path("./_hf_download_tmp")
        tmp_dir.mkdir(exist_ok=True)
        try:
            pack_dpuf.pack_hf_model(
                tmp_dir,
                args.out,
                mode=args.mode,
                max_seq_len=args.max_seq_len,
                step_mult=args.step_mult,
                embed_mult=args.embed_mult,
                hf_repo=args.source,
                delete_shards=args.delete_shards,
            )
        finally:
            if args.delete_shards and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
