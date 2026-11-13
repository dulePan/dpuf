"""
pack_dpuf.py — Standalone Packer from HuggingFace SafeTensors Directory to Self-Contained .dpuf

Usage:
  python pack_dpuf.py --model-dir ../real_models/Qwen3-0.6B --out ../real_models/dpuf/Qwen3-0.6B-standalone-had4.dpuf --mode had4
  python pack_dpuf.py --model-dir ../real_models/Qwen3-0.6B --out ../real_models/dpuf/Qwen3-0.6B-standalone-hybrid.dpuf --mode hybrid
  python pack_dpuf.py --model-dir ../real_models/Llama-3.2-1B-Instruct --out ../real_models/dpuf/Llama-3.2-1B-Instruct-standalone-had4.dpuf --mode had4
"""

import argparse
import gc
import json
import mmap
import os
import pathlib
import shutil
import struct
import subprocess
import time
import zlib
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import dpuf_format as dpuf


def _safetensors_shard_list(model_dir: pathlib.Path) -> List[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        return sorted(set(weight_map.values()))
    return ["model.safetensors"]


def list_hf_safetensors_names(model_dir: pathlib.Path) -> List[str]:
    """Brzo cita samo index/JSON zaglavlja SafeTensors fajlova i vraca listu HF naziva tenzora."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        return list(weight_map.keys())
    names: List[str] = []
    for fname in _safetensors_shard_list(model_dir):
        fpath = model_dir / fname
        with open(fpath, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len).decode("utf-8"))
            for name in hdr.keys():
                if name != "__metadata__":
                    names.append(name)
    return names


def _download_hf_shard(hf_repo: str, fname: str, dst_path: pathlib.Path) -> None:
    url = f"https://huggingface.co/{hf_repo}/resolve/main/{fname}"
    part_path = dst_path.with_suffix(dst_path.suffix + ".part")
    print(f"  [DL] Downloading {fname} from {hf_repo} ...", flush=True)
    t_dl = time.perf_counter()
    curl_exe = shutil.which("curl.exe") or shutil.which("curl")
    if curl_exe:
        cmd = [curl_exe, "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", str(part_path), url]
        subprocess.run(cmd, check=True)
    else:
        import urllib.request
        with urllib.request.urlopen(url) as resp, open(part_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f, length=4 * 1024 * 1024)
    part_path.replace(dst_path)
    sz_gib = dst_path.stat().st_size / (1024.0 ** 3)
    print(f"  [DL] Ready {fname} ({sz_gib:.2f} GiB in {time.perf_counter() - t_dl:.1f}s)", flush=True)


def iter_hf_safetensors_bf16(
    model_dir: pathlib.Path,
    hf_repo: Optional[str] = None,
    delete_downloaded_shards: bool = False,
) -> Iterator[Tuple[str, np.ndarray]]:
    """Zero-dependency mmap SafeTensors reader returning (tensor_name, uint16_bf16_array)."""
    for fname in _safetensors_shard_list(model_dir):
        fpath = model_dir / fname
        downloaded_here = False
        if not fpath.exists():
            if not hf_repo:
                raise FileNotFoundError(f"Missing SafeTensors shard {fpath} (pass --hf-repo to stream)")
            _download_hf_shard(hf_repo, fname, fpath)
            downloaded_here = True
        with open(fpath, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(mm)
            try:
                hdr_len = struct.unpack("<Q", mv[:8])[0]
                hdr = json.loads(bytes(mv[8 : 8 + hdr_len]).decode("utf-8"))
                base = 8 + hdr_len
                for name, meta in hdr.items():
                    if name == "__metadata__":
                        continue
                    dtype = meta["dtype"]
                    shape = tuple(meta["shape"])
                    start, end = meta["data_offsets"]
                    slice_mv = mv[base + start : base + end]
                    try:
                        if dtype == "BF16":
                            arr = np.frombuffer(slice_mv, dtype=np.uint16).reshape(shape).copy()
                        elif dtype == "F16":
                            f32 = np.frombuffer(slice_mv, dtype=np.float16).astype(np.float32).reshape(shape)
                            arr = dpuf.fp32_to_bf16_u16(f32)
                        elif dtype == "F32":
                            f32 = np.frombuffer(slice_mv, dtype=np.float32).reshape(shape)
                            arr = dpuf.fp32_to_bf16_u16(f32)
                        else:
                            raise ValueError(f"Unsupported SafeTensors dtype {dtype} for {name}")
                    finally:
                        slice_mv.release()
                    yield name, arr
            finally:
                mv.release()
                mm.close()
        if delete_downloaded_shards and downloaded_here and fpath.exists():
            fpath.unlink()
            print(f"  [DL] Deleted temporary shard {fname}", flush=True)


def map_hf_tensor_name(hf_name: str) -> str:
    # Normalize Qwen3-VL / multimodal language_model prefix
    lm_name = hf_name
    if lm_name.startswith("model.language_model."):
        lm_name = "model." + lm_name[len("model.language_model."):]

    if lm_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if lm_name == "model.norm.weight":
        return "output_norm.weight"
    if lm_name == "lm_head.weight":
        return "output.weight"
    if lm_name.startswith("model.layers."):
        rest = lm_name[len("model.layers."):]
        layer_idx, suffix = rest.split(".", 1)
        mapping = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        if suffix in mapping:
            return f"blk.{layer_idx}.{mapping[suffix]}"

    # Vision encoder + multimodal projectors (Qwen3-VL)
    if hf_name == "model.visual.patch_embed.proj.weight":
        return "v.patch_embed.weight"
    if hf_name == "model.visual.patch_embed.proj.bias":
        return "v.patch_embed.bias"
    if hf_name == "model.visual.pos_embed.weight":
        return "v.pos_embed.weight"
    if hf_name.startswith("model.visual.blocks."):
        rest = hf_name[len("model.visual.blocks."):]
        layer_idx, suffix = rest.split(".", 1)
        vmapping = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "attn.qkv.weight": "attn_qkv.weight",
            "attn.qkv.bias": "attn_qkv.bias",
            "attn.proj.weight": "attn_proj.weight",
            "attn.proj.bias": "attn_proj.bias",
            "mlp.linear_fc1.weight": "mlp_fc1.weight",
            "mlp.linear_fc1.bias": "mlp_fc1.bias",
            "mlp.linear_fc2.weight": "mlp_fc2.weight",
            "mlp.linear_fc2.bias": "mlp_fc2.bias",
        }
        if suffix in vmapping:
            return f"v.blk.{layer_idx}.{vmapping[suffix]}"
    if hf_name.startswith("model.visual.merger."):
        suffix = hf_name[len("model.visual.merger."):]
        mmapping = {
            "norm.weight": "v.merger.norm.weight",
            "norm.bias": "v.merger.norm.bias",
            "linear_fc1.weight": "v.merger.fc1.weight",
            "linear_fc1.bias": "v.merger.fc1.bias",
            "linear_fc2.weight": "v.merger.fc2.weight",
            "linear_fc2.bias": "v.merger.fc2.bias",
        }
        return mmapping.get(suffix, "")
    if hf_name.startswith("model.visual.deepstack_merger_list."):
        rest = hf_name[len("model.visual.deepstack_merger_list."):]
        ds_idx, suffix = rest.split(".", 1)
        dmapping = {
            "norm.weight": "norm.weight",
            "norm.bias": "norm.bias",
            "linear_fc1.weight": "fc1.weight",
            "linear_fc1.bias": "fc1.bias",
            "linear_fc2.weight": "fc2.weight",
            "linear_fc2.bias": "fc2.bias",
        }
        if suffix in dmapping:
            return f"v.deepstack.{ds_idx}.{dmapping[suffix]}"
    return ""


def _hadamard_seed_name(cname: str) -> str:
    """Share sign_bits across co-input projections (Q/K/V and Gate/Up) so FWHT+Q8 runs once."""
    if cname.startswith("blk."):
        parts = cname.split(".", 2)
        if len(parts) == 3:
            l_idx, suffix = parts[1], parts[2]
            if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
                return f"blk.{l_idx}.qkv_in"
            if suffix in ("ffn_gate.weight", "ffn_up.weight"):
                return f"blk.{l_idx}.mlp_in"
    return cname


def _encode_one_tensor(
    cname: str,
    arr_u16: np.ndarray,
    mode: str,
    seed_name: str,
    step_mult: float,
    embed_mult: float,
) -> bytes:
    # 5D Conv3d patch_embed.proj: (1152, 3, 2, 16, 16) -> (1152, 1536)
    if arr_u16.ndim == 5 and cname == "v.patch_embed.weight":
        arr_u16 = arr_u16.reshape(arr_u16.shape[0], -1)

    # All 1D norms and biases are stored uncompressed (exact 0-loss) in MODE_RAW_TINY
    if arr_u16.ndim == 1:
        return dpuf.encode_tensor_raw_tiny(arr_u16)

    # 2D pos_embed is learned coordinate table (2304, 1152), store in RAW_TINY for bit-exact coordinates
    if cname == "v.pos_embed.weight":
        return dpuf.encode_tensor_raw_tiny(arr_u16)

    # ViT MLP linear_fc2 has shape (1152, 4304). Zero-pad cols 4304 -> 4352 (+48) so cols % 64 == 0,
    # enabling high-performance Mode 12/13/14 AVX2 slab kernels with exact 0-loss on padded zeros.
    if cname.endswith(".mlp_fc2.weight") and arr_u16.ndim == 2 and arr_u16.shape[1] == 4304:
        pad = np.zeros((arr_u16.shape[0], 48), dtype=arr_u16.dtype)
        arr_u16 = np.ascontiguousarray(np.hstack([arr_u16, pad]))

    is_emb = cname in ("token_embd.weight", "output.weight")
    if arr_u16.ndim != 2 or min(arr_u16.shape) < 64:
        return dpuf.encode_tensor_qr_zd4(arr_u16)

    if mode == "had4p-hybrid":
        l_idx = -1
        if cname.startswith("blk."):
            try:
                l_idx = int(cname.split(".", 2)[1])
            except ValueError:
                l_idx = -1
        if (
            cname.endswith(".ffn_down.weight")
            or cname.endswith(".attn_v.weight")
            or cname.endswith(".attn_k.weight")
            or (cname.endswith(".attn_output.weight") and l_idx >= 0 and (l_idx % 2 == 0))
            or cname == "output.weight"
            or cname.endswith(".fc2.weight")
            or cname == "v.patch_embed.weight"
        ):
            return dpuf.encode_tensor_qr_had6(arr_u16, tensor_name=cname, w_pow=0.0)
        return dpuf.encode_tensor_qr_had4_asym(arr_u16, tensor_name=cname, w_pow=0.0)
    if is_emb and mode == "hybrid":
        return dpuf.encode_tensor_qr_q8(arr_u16)
    if mode == "had4p":
        return dpuf.encode_tensor_qr_hadamard4_packed(arr_u16, tensor_name=seed_name)
    if mode == "had4":
        return dpuf.encode_tensor_qr_hadamard4(arr_u16, tensor_name=seed_name)
    if mode in ("split4", "hybrid"):
        return dpuf.encode_tensor_qr_split4(
            arr_u16, step_mult=step_mult, is_embed=is_emb, embed_mult=embed_mult
        )
    if mode == "q8":
        return dpuf.encode_tensor_qr_q8(arr_u16)
    raise ValueError(f"Unknown mode: {mode}")


def pack_hf_model(
    model_dir: pathlib.Path,
    out_path: pathlib.Path,
    mode: str = "had4p",
    max_seq_len: int = 2048,
    step_mult: float = 0.255,
    embed_mult: float = 0.75,
    hf_repo: Optional[str] = None,
    delete_shards: bool = False,
) -> dict:
    """Pakuje HF SafeTensors direktorijum direktno u .dpuf putem streaming upisa (nizak RAM)."""
    t0 = time.perf_counter()
    model_dir = pathlib.Path(model_dir)
    out_path = pathlib.Path(out_path)

    meta_info = dpuf.extract_hf_tokenizer_and_config(model_dir, max_seq_len=max_seq_len)
    tie_embd = bool(meta_info["model_config"].get("tie_word_embeddings", True))

    mode_desc = {
        "had4p": "Mode 12: Pre-Packed 8x32 Walsh-Hadamard FWHT-64 + Pure Dense Blk32 Int4 (Zero-Transpose L2-Tiled)",
        "had4p-hybrid": "Hybrid Mode 13 (LSQ HAD4_ASYM) + Mode 14 (LSQ HAD6 k/v/down/even-o/lm)",
        "had4": "Mode 10: Walsh-Hadamard FWHT-64 + Pure Dense Blk32 Int4 [-8..7] (Q8_32)",
        "split4": f"Mode 9: Branchless Split4 [-8..7] + Sparse CSR (step={step_mult}, emb={embed_mult})",
        "hybrid": f"Hybrid Mode 9 (step={step_mult}) + Mode 11 Q8_32 Embeddings/LM-Head",
        "q8": "Mode 11: QR-Tiled Pure Dense Blk32 Symmetric Int8 [-127..127]",
    }[mode]

    # Prvi brzi prolaz (samo SafeTensors JSON zaglavlja) za rezervaciju fiksno poravnatog DPUF headera.
    mapped_names: List[str] = []
    for hf_name in list_hf_safetensors_names(model_dir):
        cname = map_hf_tensor_name(hf_name)
        if not cname or (cname == "output.weight" and tie_embd):
            continue
        mapped_names.append(cname)

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_meta = {
        "format": f"DPUF / LR-FLAC v{dpuf.DPUF_FORMAT_VERSION}",
        "format_version": dpuf.DPUF_FORMAT_VERSION,
        "format_version_code": dpuf.DPUF_FORMAT_VERSION_CODE,
        "min_reader_version": "1.2.0",
        "codec_version": dpuf.DPUF_CODEC_VERSION,
        "model_revision": "1.0.0",
        "model_revision_code": 10000,
        "supersedes": None,
        "author": "D. P. (dule_pan)",
        "license": dpuf.DEFAULT_LICENSE_NOTICE,
        "created_at": created_at,
        "num_tensors": len(mapped_names),
        "payload_bytes": 9999999999999,
        "model_name": meta_info["model_name"],
        "architecture": meta_info["model_config"]["model_type"],
        "compression_mode": mode_desc,
        "total_parameters": 9999999999999,
        "bits_per_weight": 99.999,
        "model_config": dict(meta_info["model_config"], has_qk_norm=True),
        "tokenizer": meta_info["tokenizer"],
    }
    placeholder_idx = {c: [9999999999999, 99999999999] for c in mapped_names}
    ph_bytes = json.dumps({"__metadata__": base_meta, "__tensors__": placeholder_idx}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    min_hdr_len = len(ph_bytes) + 4096
    hdr_slot = min_hdr_len + dpuf._pad64(24 + min_hdr_len)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_index: Dict[str, List[int]] = {}
    total_params = 0
    raw_payload_bytes = 0
    offset = 0
    pay_crc = 0
    has_q_norm = False
    n_expected = len(mapped_names)
    done_cnt = 0

    with open(out_path, "wb") as f:
        f.write(b"\x00" * (24 + hdr_slot))
        for hf_name, arr_u16 in iter_hf_safetensors_bf16(model_dir, hf_repo=hf_repo, delete_downloaded_shards=delete_shards):
            cname = map_hf_tensor_name(hf_name)
            if not cname or (cname == "output.weight" and tie_embd):
                del arr_u16
                continue
            if "attn_q_norm" in cname:
                has_q_norm = True

            total_params += int(arr_u16.size)
            seed_name = _hadamard_seed_name(cname)
            blob = _encode_one_tensor(cname, arr_u16, mode, seed_name, step_mult, embed_mult)
            del arr_u16

            pad_n = dpuf._pad64(offset)
            if pad_n:
                pad_b = b"\x00" * pad_n
                f.write(pad_b)
                pay_crc = zlib.crc32(pad_b, pay_crc)
                offset += pad_n
            tensor_index[cname] = [offset, len(blob)]
            f.write(blob)
            pay_crc = zlib.crc32(blob, pay_crc)
            offset += len(blob)
            raw_payload_bytes += len(blob)
            del blob

            done_cnt += 1
            if n_expected >= 320 and (done_cnt % 40 == 0 or done_cnt == n_expected):
                print(f"  [{done_cnt}/{n_expected}] {cname} -> {offset / (1024**2):.1f} MiB ({time.perf_counter() - t0:.1f}s)", flush=True)
            if done_cnt % 32 == 0:
                gc.collect()

        pay_crc &= 0xFFFFFFFF
        meta_info["model_config"]["has_qk_norm"] = has_q_norm
        bpw = (raw_payload_bytes * 8.0) / float(total_params) if total_params else 0.0
        base_meta["num_tensors"] = len(tensor_index)
        base_meta["payload_bytes"] = offset
        base_meta["total_parameters"] = total_params
        base_meta["bits_per_weight"] = round(bpw, 3)
        base_meta["model_config"] = meta_info["model_config"]

        hdr_bytes = json.dumps({"__metadata__": base_meta, "__tensors__": tensor_index}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(hdr_bytes) > hdr_slot:
            raise RuntimeError(f"Reserved header slot {hdr_slot} too small for {len(hdr_bytes)} bytes")
        hdr_bytes += b" " * (hdr_slot - len(hdr_bytes))
        hdr_crc = zlib.crc32(hdr_bytes) & 0xFFFFFFFF
        top_hdr = struct.pack("<8sIIII", dpuf.MAGIC_DPUF, len(tensor_index), hdr_crc, pay_crc, len(hdr_bytes))

        f.seek(0)
        f.write(top_hdr)
        f.write(hdr_bytes)

    dt = time.perf_counter() - t0
    total_bytes = 24 + len(hdr_bytes) + offset
    return {
        "out_path": str(out_path),
        "mib": total_bytes / (1024.0 * 1024.0),
        "bpw": bpw,
        "total_params": total_params,
        "tensors": len(tensor_index),
        "pack_s": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--mode", default="had4p", choices=["had4p", "had4p-hybrid", "had4", "split4", "hybrid", "q8"])
    ap.add_argument("--max-seq-len", default=2048, type=int)
    ap.add_argument("--step-mult", default=0.255, type=float)
    ap.add_argument("--embed-mult", default=0.75, type=float)
    ap.add_argument("--hf-repo", default=None, type=str, help="HF repo to stream download missing shards from (e.g. Qwen/Qwen3-VL-8B-Instruct)")
    ap.add_argument("--delete-shards", action="store_true", help="Delete downloaded HF shard after packing to save disk")
    args = ap.parse_args()

    res = pack_hf_model(
        args.model_dir, args.out, mode=args.mode,
        max_seq_len=args.max_seq_len, step_mult=args.step_mult, embed_mult=args.embed_mult,
        hf_repo=args.hf_repo, delete_shards=args.delete_shards,
    )
    print(
        f"Packed {res['out_path']}: {res['mib']:.2f} MiB ({res['bpw']:.3f} b/w, "
        f"{res['tensors']} tensors, {res['pack_s']:.1f}s)"
    )


if __name__ == "__main__":
    main()
