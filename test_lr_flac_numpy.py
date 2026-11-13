import zlib
import numpy as np


def fp32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """IEEE-754 Round-to-Nearest-Even (RNE) konverzija iz float32 u bfloat16 (uint16 zapis)."""
    arr_fp32 = np.asarray(arr, dtype=np.float32)
    u32 = arr_fp32.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
    return ((u32 + rounding_bias) >> 16).astype(np.uint16)


def bf16_u16_to_fp32(u16: np.ndarray) -> np.ndarray:
    """Konverzija iz bfloat16 (uint16 zapis) nazad u float32."""
    u32 = np.asarray(u16, dtype=np.uint32) << 16
    return u32.view(np.float32)


def matmul_bf16(a_u16: np.ndarray, b_u16: np.ndarray) -> np.ndarray:
    """Množenje dve BF16 matrice (akumulacija u FP32 i zaokruživanje izlaza u BF16)."""
    res_fp32 = bf16_u16_to_fp32(a_u16) @ bf16_u16_to_fp32(b_u16)
    return fp32_to_bf16_u16(res_fp32)


def compress_bytes(data: bytes) -> int:
    """Vraća veličinu kompresovanih podataka u bajtovima (zlib level 9)."""
    return len(zlib.compress(data, level=9))


def shannon_entropy_bits(arr: np.ndarray) -> float:
    """Računa teorijski minimum bitova po elementu (Šenonova entropija 0-tog reda)."""
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / arr.size
    return float(-np.sum(probs * np.log2(probs)))


def create_realistic_llm_weights(rows=2048, cols=2048, rank=64, seed=42) -> np.ndarray:
    """
    Kreira realističnu matricu težina kakvu viđamo u MLP/Attention slojevima LLM-a:
    1. Jaka niskorangovna (Low-Rank) struktura nastala tokom treninga.
    2. Specifični 'outlier' kanali (kolone/redovi sa većom amplitudom).
    3. Lokalni šum oko nule u BFloat16 formatu (vraća uint16 bit-exact BF16 zapis).
    """
    rng = np.random.default_rng(seed)
    # 1. Latentni korelisani šabloni (simulacija naučenih koncepata u mreži)
    U = rng.standard_normal((rows, rank), dtype=np.float32) * 0.08
    V = rng.standard_normal((rank, cols), dtype=np.float32) * 0.08
    base_structure = U @ V

    # 2. Dodajemo grupne srodnosti između neurona (ali ih posle nasumično izmešamo)
    smooth_pattern = (
        np.sin(np.linspace(0, 10, rows, dtype=np.float32))[:, None]
        @ np.cos(np.linspace(0, 10, cols, dtype=np.float32))[None, :]
        * 0.03
    )
    perm_r = rng.permutation(rows)
    perm_c = rng.permutation(cols)
    scrambled_pattern = smooth_pattern[perm_r][:, perm_c]

    # 3. Dodajemo finu strukturu / šum
    noise = rng.standard_normal((rows, cols), dtype=np.float32) * 0.004

    # 4. Outlier kanali (2% neurona ima znatno veće težine - poznati LLM fenomen)
    channel_scales = np.ones(cols, dtype=np.float32)
    outlier_idx = rng.permutation(cols)[: int(cols * 0.02)]
    channel_scales[outlier_idx] = 5.0

    W_fp32 = (base_structure + scrambled_pattern + noise) * channel_scales
    return fp32_to_bf16_u16(W_fp32)


# =====================================================================
# TEST 1: DOKAZ 100% IDENTIČNOSTI PRI ZAMENI KOLONA/REDOVA (INVARIANTS)
# =====================================================================
def prove_permutation_invariance():
    print("=" * 72)
    print("TEST 1: DOKAZ PERMUTACIONE INVARIJANTNOSTI (Dve spojene matrice: W1 -> W2)")
    print("=" * 72)

    rng = np.random.default_rng(10)
    X = fp32_to_bf16_u16(rng.standard_normal((1, 512), dtype=np.float32))
    W1 = fp32_to_bf16_u16(rng.standard_normal((512, 1024), dtype=np.float32))
    W2 = fp32_to_bf16_u16(rng.standard_normal((1024, 512), dtype=np.float32))

    # Originalni prolaz kroz mrežu
    Y_orig = matmul_bf16(matmul_bf16(X, W1), W2)

    # Napravimo nasumičnu permutaciju (zamenu mesta za svih 1024 neurona)
    perm = rng.permutation(1024)

    # Zamenimo KOLONE u W1 i istovremeno zamenimo REDOVE u W2
    W1_perm = W1[:, perm]
    W2_perm = W2[perm, :]

    # Varijanta A: Vraćanje redosleda kolona indeksiranjem (100% bit-exact čak i za float sumu)
    hidden_perm = matmul_bf16(X, W1_perm)
    inv_perm = np.argsort(perm)
    hidden_restored = hidden_perm[:, inv_perm]
    Y_restored = matmul_bf16(hidden_restored, W2)

    max_diff = float(np.max(np.abs(bf16_u16_to_fp32(Y_orig) - bf16_u16_to_fp32(Y_restored))))
    bit_identical = bool(np.array_equal(Y_orig, Y_restored))

    # Varijanta B: Direktno množenje hidden_perm @ W2_perm (bez ikakvog indeksiranja u runtime-u!)
    Y_direct_perm = matmul_bf16(hidden_perm, W2_perm)
    max_diff_direct = float(np.max(np.abs(bf16_u16_to_fp32(Y_orig) - bf16_u16_to_fp32(Y_direct_perm))))
    bit_identical_direct = bool(np.array_equal(Y_orig, Y_direct_perm))

    print(f"[Sa vraćanjem indeksa] Maksimalna numerička razlika izlaza: {max_diff:.10f}")
    print(f"[Sa vraćanjem indeksa] Da li je izlaz U BIT IDENTIČAN (100% bit-exact)? -> {bit_identical}")
    print(f"[Direktno W1_perm @ W2_perm] Maksimalna razlika (zbog redosleda FP sabiranja): {max_diff_direct:.10f}")
    print(f"[Direktno W1_perm @ W2_perm] Bit-exact bez vraćanja indeksa: {bit_identical_direct}")
    print()


# =====================================================================
# TEST 2: POREDJENJE METODA KOMPRESIJE ("WAV" vs "DFloat11" vs "LR-FLAC")
# =====================================================================
def run_compression_experiment():
    print("=" * 72)
    print("TEST 2: EKSPERIMENT KOMPRESIJE ('AV1 za LLM' / LR-FLAC Prototip)")
    print("Koder u upotrebi: zlib (Deflate/Huffman)")
    print("=" * 72)

    rows, cols = 2048, 2048
    total_params = rows * cols
    W_u16 = create_realistic_llm_weights(rows, cols, rank=64)

    raw_bytes = W_u16.nbytes
    print(f"Dimenzije matrice: {rows} x {cols} ({total_params:,} parametara)")
    print(f"[0] Originalni BF16 ('WAV'): {raw_bytes / 1024:.1f} KB -> 16.00 bita/težini\n")

    # -----------------------------------------------------------------
    # METOD 1: DFloat11 / ZipNN stil (Bit-Plane razdvajanje sirove matrice)
    # -----------------------------------------------------------------
    sign_bit = ((W_u16 >> 15) & 0x1).astype(np.uint8)
    exponent = ((W_u16 >> 7) & 0xFF).astype(np.uint8)
    mantissa = (W_u16 & 0x7F).astype(np.uint8)

    packed_signs = np.packbits(sign_bit).tobytes()
    comp_exp = compress_bytes(exponent.tobytes())
    comp_man = compress_bytes(mantissa.tobytes())
    comp_sign = compress_bytes(packed_signs)

    dfloat11_total = comp_sign + comp_exp + comp_man
    dfloat11_bits = (dfloat11_total * 8) / total_params
    exp_entropy = shannon_entropy_bits(exponent)
    man_entropy = shannon_entropy_bits(mantissa)

    print("--- METOD 1: Standardni DFloat11 / ZipNN (Bez predikcije šablona) ---")
    print(f"  Entropija eksponenta (od 8 bita): {exp_entropy:.2f} bita")
    print(f"  Entropija mantise (od 7 bita):    {man_entropy:.2f} bita")
    print(f"  Ukupna veličina: {dfloat11_total / 1024:.1f} KB")
    print(f"  Prosečno bitova po težini: {dfloat11_bits:.2f} bita (Ušteda: {(1 - dfloat11_bits/16)*100:.1f}%)\n")

    # -----------------------------------------------------------------
    # METOD 2: Naš "LR-FLAC" (100% BIT-EXACT LOSSLESS)
    # -----------------------------------------------------------------
    W_fp32 = bf16_u16_to_fp32(W_u16)

    # Korak 2.2: PNG Trik (Sortiranje kolona po kosinusnoj sličnosti / profilu)
    col_norms = np.linalg.norm(W_fp32, axis=0) + 1e-8
    W_normed = W_fp32 / col_norms
    u0 = W_normed[:, 0]
    scores = u0 @ W_normed
    perm_cols = np.argsort(scores)
    inv_perm_cols = np.argsort(perm_cols)

    # Sortiramo i redove na isti način za 2D glatkoću!
    row_norms = np.linalg.norm(W_fp32, axis=1) + 1e-8
    W_row_normed = (W_fp32.T / row_norms).T
    v0 = W_row_normed[0, :]
    row_scores = W_row_normed @ v0
    perm_rows = np.argsort(row_scores)
    inv_perm_rows = np.argsort(perm_rows)

    # Primenjujemo permutaciju na našu originalnu BF16 matricu
    W_perm_u16 = W_u16[perm_rows][:, perm_cols]
    W_perm_fp32 = bf16_u16_to_fp32(W_perm_u16)

    # Korak 2.3: Izvlačimo Low-Rank (SVD) predikciju iz permutovane matrice
    rank = 64
    U, S, Vh = np.linalg.svd(W_perm_fp32, full_matrices=False)
    L_u16 = fp32_to_bf16_u16(U[:, :rank] * S[:rank])
    R_u16 = fp32_to_bf16_u16(Vh[:rank, :])

    # Rekonstrukcija predikcije u BF16 formatu
    W_pred_u16 = matmul_bf16(L_u16, R_u16)

    def bf16_u16_to_ordered_i16(u16: np.ndarray) -> np.ndarray:
        i16 = u16.view(np.int16).copy()
        neg_mask = i16 < 0
        i16[neg_mask] ^= 0x7FFF
        return i16

    def ordered_i16_to_bf16_u16(i16: np.ndarray) -> np.ndarray:
        res = i16.copy()
        neg_mask = res < 0
        res[neg_mask] ^= 0x7FFF
        return res.view(np.uint16)

    # Pretvaramo i original i SVD predikciju u ovaj neprekidni celobrojni prostor
    ordered_orig = bf16_u16_to_ordered_i16(W_perm_u16)
    ordered_pred = bf16_u16_to_ordered_i16(W_pred_u16)

    # Oduzimamo šablon! (100% reverzibilno u uint16 aritmetici modula 65536)
    residual_i16 = (ordered_orig.astype(np.int32) - ordered_pred.astype(np.int32)).astype(np.int16)

    # ZigZag kodiranje (0->0, -1->1, 1->2, -2->3...)
    zigzag_res = ((residual_i16 >> 15) ^ (residual_i16 << 1)).view(np.uint16)

    # Razdvajamo viši bajt i niži bajt ostatka
    high_byte = (zigzag_res >> 8).astype(np.uint8)
    low_byte = (zigzag_res & 0xFF).astype(np.uint8)

    # Merimo veličinu svega što moramo da sačuvamo u fajlu:
    overhead_bytes = (
        compress_bytes(L_u16.tobytes())
        + compress_bytes(R_u16.tobytes())
        + compress_bytes(perm_cols.astype(np.uint16).tobytes())
        + compress_bytes(perm_rows.astype(np.uint16).tobytes())
    )

    comp_high = compress_bytes(high_byte.tobytes())
    comp_low = compress_bytes(low_byte.tobytes())

    lr_flac_total = overhead_bytes + comp_high + comp_low
    lr_flac_bits = (lr_flac_total * 8) / total_params

    # PROVERA DEKOMPRESIJE (Da li zaista vraća 100% u bit istu matricu?)
    reconstructed_zigzag = (high_byte.astype(np.uint16) << 8) | low_byte.astype(np.uint16)
    reconstructed_res_i16 = (
        (reconstructed_zigzag >> 1).astype(np.int32) ^ -(reconstructed_zigzag & 1).astype(np.int32)
    ).astype(np.int16)
    reconstructed_ordered = (ordered_pred.astype(np.int32) + reconstructed_res_i16.astype(np.int32)).astype(np.int16)
    reconstructed_perm_u16 = ordered_i16_to_bf16_u16(reconstructed_ordered)
    W_reconstructed_u16 = reconstructed_perm_u16[inv_perm_rows][:, inv_perm_cols]

    is_exact = bool(np.array_equal(W_u16, W_reconstructed_u16))

    print("--- METOD 2: Naš 'LR-FLAC' (100% BIT-EXACT LOSSLESS) ---")
    print(f"  Entropija višeg bajta ostatka (od 8 bita): {shannon_entropy_bits(high_byte):.2f} bita!")
    print(f"  Entropija nižeg bajta ostatka (od 8 bita): {shannon_entropy_bits(low_byte):.2f} bita!")
    print(f"  Veličina šablona (L, R + Permutacija): {overhead_bytes / 1024:.1f} KB")
    print(f"  Veličina kompresovanog ostatka: {(comp_high + comp_low) / 1024:.1f} KB (high: {comp_high/1024:.1f} KB, low: {comp_low/1024:.1f} KB)")
    print(f"  UKUPNA VELIČINA: {lr_flac_total / 1024:.1f} KB")
    print(f"  Prosečno bitova po težini: {lr_flac_bits:.2f} bita (Ušteda: {(1 - lr_flac_bits/16)*100:.1f}%)")
    print(f"  100% U BIT IDENTIČNA REKONSTRUKCIJA: {is_exact}\n")

    # -----------------------------------------------------------------
    # METOD 3: "AV1 Near-Lossless" (Odbacivanje samo poslednja 3 bita šuma u ostatku)
    # -----------------------------------------------------------------
    noise_bits_to_drop = 3
    near_lossless_zigzag = zigzag_res >> noise_bits_to_drop
    nl_high = (near_lossless_zigzag >> 8).astype(np.uint8)
    nl_low = (near_lossless_zigzag & 0xFF).astype(np.uint8)

    nl_total = overhead_bytes + compress_bytes(nl_high.tobytes()) + compress_bytes(nl_low.tobytes())
    nl_bits = (nl_total * 8) / total_params

    # Rekonstrukcija Near-Lossless matrice da izmerimo tačnost
    rec_nl_zz = near_lossless_zigzag << noise_bits_to_drop
    rec_nl_res = ((rec_nl_zz >> 1).astype(np.int32) ^ -(rec_nl_zz & 1).astype(np.int32)).astype(np.int16)
    rec_nl_ord = (ordered_pred.astype(np.int32) + rec_nl_res.astype(np.int32)).astype(np.int16)
    rec_nl_u16 = ordered_i16_to_bf16_u16(rec_nl_ord)
    W_nl_u16 = rec_nl_u16[inv_perm_rows][:, inv_perm_cols]

    # Kosinusna sličnost izlaza sloja na nasumičnom ulazu X
    rng_test = np.random.default_rng(123)
    X_test_u16 = fp32_to_bf16_u16(rng_test.standard_normal((16, rows), dtype=np.float32))
    Y_true = bf16_u16_to_fp32(matmul_bf16(X_test_u16, W_u16)).flatten()
    Y_nl = bf16_u16_to_fp32(matmul_bf16(X_test_u16, W_nl_u16)).flatten()
    cos_sim = float(np.dot(Y_true, Y_nl) / (np.linalg.norm(Y_true) * np.linalg.norm(Y_nl)))

    print("--- METOD 3: Naš 'AV1 Near-Lossless' (Sa filtriranjem 3 LSB bita šuma) ---")
    print(f"  UKUPNA VELIČINA: {nl_total / 1024:.1f} KB")
    print(f"  Prosečno bitova po težini: {nl_bits:.2f} bita (Ušteda: {(1 - nl_bits/16)*100:.1f}%)")
    print(f"  Kosinusna sličnost izlaza neurona: {cos_sim:.7f} (Praktično identično originalu!)")
    print("=" * 72)


if __name__ == "__main__":
    prove_permutation_invariance()
    run_compression_experiment()
