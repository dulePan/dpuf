"""
Benchmark DPUF v1.0.0 (4-bit, BF16 skale) vs v1.1.0 (ALS 5-bit, FP16 skale + brzi matmul).
Meri: velicinu, preciznost (cos_sim / Top-1 / Top-5 na logitima) i brzinu matmul-a.
"""
import time
from typing import Dict, Any
import numpy as np

import dpuf_format as dpuf
import test_qwen_llama_lrflac as tql

# Forward pass iz test_qwen_llama_lrflac prepoznaje ZeroDecodeWeight po klasi -> koristimo klasu iz dpuf_format
tql.ZeroDecodeWeight = dpuf.ZeroDecodeWeight


def metrics(ref_logits: np.ndarray, logits: np.ndarray):
    ref_top1 = np.argmax(ref_logits, axis=-1)
    a = ref_logits.ravel().astype(np.float64)
    b = logits.ravel().astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    top1 = float(np.mean(np.argmax(logits, axis=-1) == ref_top1) * 100.0)
    top5_idx = np.argsort(logits, axis=-1)[..., -5:]
    top5 = float(np.mean(np.any(top5_idx == ref_top1[..., None], axis=-1)) * 100.0)
    return cos, top1, top5


def run_codec(tensors, cfg, ids, ref_logits, label, encoder):
    total_params = sum(v.size for v in tensors.values())
    t0 = time.perf_counter()
    blobs = {k: encoder(k, v) for k, v in tensors.items()}
    enc_s = time.perf_counter() - t0
    size = sum(len(b) for b in blobs.values())

    t0 = time.perf_counter()
    weights = {k: dpuf.unpack_zero_decode_tensor(b) for k, b in blobs.items()}
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    logits = tql.run_decoder_only_forward(weights, cfg, ids)
    fw_s = time.perf_counter() - t0

    cos, t1, t5 = metrics(ref_logits, logits)
    print(
        f"{label:<34} | {size/1024**2:7.2f} MB | {(1 - size/(total_params*2))*100:5.1f}% | "
        f"{size*8/total_params:5.2f} b/w | load {load_s:5.2f}s | fwd {fw_s:5.2f}s | "
        f"cos {cos:.6f} | top1 {t1:5.1f}% | top5 {t5:5.1f}%"
    )
    return weights


def bench_matmul():
    """Mikro-benchmark: stari int16 matmul (v1.0.0) vs kontigualni float32 bafer (v1.1.0)."""
    rng = np.random.default_rng(0)
    rows, cols = 1536, 512
    q = rng.integers(-20, 20, size=(rows, cols), dtype=np.int16)
    rs = rng.random(rows, dtype=np.float32)
    cs = rng.random(cols, dtype=np.float32)
    x = rng.standard_normal((2, 1, cols), dtype=np.float32)

    w = dpuf.ZeroDecodeWeight(q, rs, cs)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        y_old = np.matmul(x * cs, q.T) * rs  # v1.0.0 putanja
    t_old = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        y_new = w.matmul_input_x_wt(x)  # v1.1.0 putanja
    t_new = (time.perf_counter() - t0) / n * 1000
    print(f"\nMatmul 1536x512, batch=2 token: v1.0.0 {t_old:.3f} ms | v1.1.0 {t_new:.3f} ms | "
          f"ubrzanje {t_old/t_new:.1f}x | max|diff|={np.max(np.abs(y_old - y_new)):.2e}")
    mem_i16 = q.nbytes
    mem_f32 = w._qt_f32.nbytes
    print(f"RAM za Q: int16 {mem_i16/1024:.0f} KB | float32 kes {mem_f32/1024:.0f} KB ({mem_f32/mem_i16:.0f}x vise)")


def main():
    print("=" * 150)
    print("DPUF v1.0.0 (4-bit BF16 skale) vs v1.1.0 (ALS FP16 skale + 5-bit + Nibble-Escape)")
    print("=" * 150)
    for name, builder, n_layers in [
        ("Qwen3-0.6B", tql.build_qwen3_06b_tensors_bf16, 28),
        ("Llama-3.2-1B", tql.build_llama32_1b_tensors_bf16, 16),
    ]:
        tensors, cfg = builder(n_layers)
        ref_w = {k: dpuf.bf16_u16_to_fp32(v) for k, v in tensors.items()}
        rng = np.random.default_rng(42)
        ids = rng.integers(10, cfg["vocab_size"] - 10, size=(2, 32), dtype=np.int64)
        t0 = time.perf_counter()
        ref_logits = tql.run_decoder_only_forward(ref_w, cfg, ids)
        ref_s = time.perf_counter() - t0
        print(f"\n[{name}] {sum(v.size for v in tensors.values()):,} parametara | referentni FP32 fwd {ref_s:.2f}s")

        run_codec(
            tensors, cfg, ids, ref_logits, "v1.0.0 4-bit (BF16 skale)",
            lambda k, v: dpuf.encode_tensor_zero_decode_4bit(v, 0.180, is_embed=("embed_tokens" in k)),
        )
        run_codec(
            tensors, cfg, ids, ref_logits, "v1.1.0 ALS 5-bit (FP16 skale)",
            lambda k, v: dpuf.encode_tensor_zero_decode_v11(v, k, base_step=0.112),
        )
        run_codec(
            tensors, cfg, ids, ref_logits, "v1.1.0 ALS 5-bit (finiji, 0.095)",
            lambda k, v: dpuf.encode_tensor_zero_decode_v11(v, k, base_step=0.095),
        )

    bench_matmul()


if __name__ == "__main__":
    main()
