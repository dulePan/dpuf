import time
import json
import pathlib
import numpy as np
import dpuf_format as dpuf


def main():
    # Kreiramo demonstracioni .dpuf fajl sa bogatim __metadata__ zaglavljem
    rng = np.random.default_rng(42)
    w_f32 = (rng.standard_normal((512, 512), dtype=np.float32) * 0.02).astype(np.float32)
    w_u16 = dpuf.fp32_to_bf16_u16(w_f32)

    encoded = {
        "model.layers.0.self_attn.q_proj.weight": dpuf.encode_tensor_zero_decode_4bit(w_u16, step_mult=0.180),
        "model.layers.0.input_layernorm.weight": dpuf.encode_tensor_zero_decode_4bit(
            dpuf.fp32_to_bf16_u16(np.ones(512, dtype=np.float32))
        ),
    }

    custom_meta = {
        "model_name": "Qwen3-0.6B-Demo",
        "architecture": "qwen3",
        "compression_mode": "Direct Zero-Decode 4-bit (2D-RMS + Nibble-Escape)",
        "original_bf16_bytes": int(w_u16.nbytes + 1024),
        "bits_per_weight": round(len(encoded["model.layers.0.self_attn.q_proj.weight"]) * 8.0 / w_u16.size, 2),
        "model_config": {
            "hidden_size": 512,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "rope_theta": 1000000.0,
        },
    }

    out_path = pathlib.Path("sample_model_v1.dpuf")
    total_bytes = dpuf.save_dpuf_container(out_path, encoded, metadata=custom_meta, use_dpuf_magic=True)

    # Simuliramo i buduću unapređenu reviziju modela (v1.1.0) koja nasleđuje v1.0.0
    v2_meta = dict(custom_meta)
    v2_meta["model_revision"] = "1.1.0"
    v2_meta["supersedes"] = "v1.0.0 (CRC32 improved calibration)"
    v2_path = pathlib.Path("sample_model_v1_1.dpuf")
    dpuf.save_dpuf_container(v2_path, encoded, metadata=v2_meta, use_dpuf_magic=True)

    t0 = time.perf_counter()
    meta = dpuf.inspect_dpuf_metadata(out_path)
    inspect_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Uspešno kreiran {out_path.name} ({total_bytes:,} bajtova)!")
    print(f"Vreme čitanja __metadata__ zaglavlja (bez učitavanja tenzora): {inspect_ms:.3f} ms")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    diff_ver = dpuf.compare_dpuf_versions(out_path, v2_path)
    print("\nPoređenje verzija modela iz zaglavlja (v1.0.0 vs v1.1.0):")
    print(json.dumps(diff_ver, indent=2, ensure_ascii=False))

    # Verifikujemo i Zero-Decode matrično množenje
    blobs, _ = dpuf.load_dpuf_container(v2_path, verify_crc=True)
    zd_w = dpuf.unpack_zero_decode_tensor(blobs["model.layers.0.self_attn.q_proj.weight"])
    x = rng.standard_normal((2, 16, 512), dtype=np.float32)
    y_zd = zd_w.matmul_input_x_wt(x)
    y_ref = x @ dpuf.bf16_u16_to_fp32(w_u16).T
    cos = float(np.dot(y_zd.ravel(), y_ref.ravel()) / (np.linalg.norm(y_zd) * np.linalg.norm(y_ref)))
    print(f"\nZero-Decode MatMul CosSim (v1.1.0): {cos:.6f}")

    # Čišćenje demo fajlova
    out_path.unlink(missing_ok=True)
    v2_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
