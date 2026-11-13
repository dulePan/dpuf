# DPUF / LR-FLAC Binary Format Specification (`v1.0`)

**DPUF** (*D.P. Unified Format* / *Dual-Precision Unified Format*, `.dpuf` / `.lrflac`) is a secure, self-describing binary container and compression format for Large Language Model (`BFloat16`) weights. It unifies **100% Bit-Exact Lossless compression** and **Direct Zero-Decode 4-bit execution** within a single malware-safe container.

---

## 1. File Header & Self-Describing Metadata Layout

Unlike legacy formats (`GGML`, `.pt` pickle files), every `.dpuf` / `.lrflac` file begins with a fixed 24-byte security prefix followed by a `CRC32`-verified JSON header containing both rich model metadata (`__metadata__`) and the tensor byte-offset directory (`__tensors__`).

```text
+-----------------------------------------------------------------------+
| Offset 0..7   (8B) : Magic Signature ("DPUF0001" or "LRFLAC03")       |
| Offset 8..11  (4B) : uint32_le  num_tensors                           |
| Offset 12..15 (4B) : uint32_le  header_crc32  (CRC32 of JSON Header)  |
| Offset 16..19 (4B) : uint32_le  payload_crc32 (CRC32 of Binary Data)  |
| Offset 20..23 (4B) : uint32_le  json_header_byte_length               |
+-----------------------------------------------------------------------+
| Offset 24 .. 24 + json_header_byte_length :                           |
|   UTF-8 JSON Object:                                                  |
|   {                                                                   |
|     "__metadata__": {                                                 |
|       "format": "DPUF / LR-FLAC v1.0.0",                              |
|       "format_version": "1.0.0",                                      |
|       "format_version_code": 10000,                                   |
|       "min_reader_version": "1.0.0",                                  |
|       "codec_version": "1.0.0",                                       |
|       "model_revision": "1.0.0",                                      |
|       "model_revision_code": 10000,                                   |
|       "supersedes": null,                                             |
|       "author": "D. P. (dule_pan)",                                   |
|       "license": "Free <= 1M MAU & Research (D. P. / dule_pan)",      |
|       "created_at": "2026-11-11T02:55:00Z",                           |
|       "model_name": "Qwen3-0.6B",                                     |
|       "architecture": "qwen3",                                        |
|       "compression_mode": "zero_decode_4bit",                         |
|       "original_bf16_bytes": 193003456,                               |
|       "compressed_bytes": 65252864,                                   |
|       "bits_per_weight": 5.41,                                        |
|       "model_config": { ... }                                         |
|     },                                                                |
|     "__tensors__": {                                                  |
|       "model.embed_tokens.weight": [0, 8912896],                      |
|       "model.layers.0.self_attn.q_proj.weight": [8912896, 172032],    |
|       ...                                                             |
|     }                                                                 |
|   }                                                                   |
+-----------------------------------------------------------------------+
| Binary Tensor Payload (Offset 24 + json_header_byte_length .. EOF)    |
+-----------------------------------------------------------------------+
```

---

## 2. Built-in Security Guarantees (Anti-Malware & Anti-ZipBomb)

1. **Zero Arbitrary Code Execution:** The parser never invokes `pickle`, `eval`, or `exec`. Only raw numerical arrays (`uint8`, `uint16`, `int16`, `float32`) are mapped.
2. **Dual `CRC32` Integrity Verification:** Both the JSON header (`header_crc32`) and the binary tensor payload (`payload_crc32`) are verified prior to tensor unpacking. Any bit-flip or unauthorized binary modification immediately aborts loading.
3. **Strict Allocation Bounds (`Anti-OOM / Anti-ZipBomb`):**
   - `num_tensors <= 4096`
   - `rows, cols <= 262,144`
   - `tensor_byte_span <= file_payload_size`

---

## 3. Tensor Encoding Modes

Each tensor slice inside the binary payload starts with a 1-byte `mode` tag:

| Mode ID | Name | Description | Typical Size (`b/w`) |
| :---: | :--- | :--- | :---: |
| `0` | `MODE_RAW_TINY` | Raw 1D `BFloat16` vectors (`RMSNorm`, biases `< 64` dims) | `16.00 b/w` |
| `2` | `MODE_LOSSLESS_MAG_LUT` | **100% Bit-Exact Lossless:** 2D L2-norm permutation + 32,768-bit `Magnitude-LUT` (`hi4` entropy-coded + `lo4` nibble-packed + `1b` packed sign) | **`10.73 – 10.93 b/w`** (`-32.7%`) |
| `4` | `MODE_NEAR_LOSSLESS_2D_RMS_BYTE` | **Near-Lossless (`6-bit`..`7-bit`):** 2D Row-Column RMS normalization (`row_rms[i] * col_rms[j] * step_mult`) + ZigZag byte split + `zlib` | **`5.87 – 7.00 b/w`** (`-61%`..`-63%`) |
| `5` | `MODE_NEAR_LOSSLESS_2D_RMS_NIBBLE` | **Near-Lossless (`3.5-bit`..`4-bit`):** 2D Row-Column RMS normalization + 4-bit Nibble-Pair packing (`15` = escape to `uint16` overflow) + `zlib` | **`4.15 – 4.61 b/w`** (`-71%`..`-74%`) |
| `6` | `MODE_DIRECT_MMAP_4BIT` | **Direct Zero-Decode (`No zlib`):** Uncompressed 2D-RMS scales + 4-bit Nibble-Pairs (`15` = escape to `uint8`/`uint16` overflow). Enables instant loading (`~245 MB/s`) and **Direct Zero-Decode MatMul**: $Y = ((X \odot \text{col\_rms}) Q^T) \odot (\text{row\_rms} \cdot \text{step})$ without reconstructing `BFloat16` weight matrices in RAM! | **`5.41 – 5.65 b/w`** (`-65%`..`-68%`) |

| `7` | `MODE_DIRECT_ALS_5BIT` *(v1.1.0)* | **High-precision Zero-Decode (`No zlib`):** `float16` row/col scales refined by Alternating Least-Squares (minimises `‖W − diag(r)·Q·diag(c)‖²`), layer-adaptive step (embeddings / attention finer), 5-bit base codes (`31` = escape) + 4-bit nibble tier (`15` = escape) + `uint16` super-outlier tier. Header: `<BIIIIIII` = mode, rows, cols, n_tier1, len_scales, len_base5, len_tier1, len_tier2. **Requires reader `>= 1.1.0`** (`min_reader_version` is set automatically). | **`6.0 – 6.1 b/w`** (`-62%`) |

> **Compatibility:** files that contain only modes `0..6` keep `min_reader_version = "1.0.0"`. A file with at least one mode-`7` tensor is stamped `"1.1.0"`, and older readers refuse to load it with a clear error instead of mis-decoding it.
