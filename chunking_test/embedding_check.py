from pathlib import Path
import numpy as np

folder = Path(r"F:\internal_split\sentence_embedding")
for npy in folder.glob("vectors_*.npy"):
    js = npy.with_name(npy.name.replace("vectors_", "keys_")).with_suffix(".jsonl")
    v = np.load(npy, mmap_mode="r")
    n = sum(1 for x in open(js, encoding="utf-8") if x.strip())
    ok = len(v) == n and not np.isnan(v).any() and not np.isinf(v).any()

    norms = np.linalg.norm(v, axis=1)
    zero_num = np.sum(norms < 1e-8)
    normalized = np.allclose(norms, 1.0, atol=1e-3)

    print(
        npy.name,
        "zero:", zero_num,
        "norm mean:", norms.mean(),
        "normalized:", normalized,
    )
    print(npy.name, "OK" if ok else "ERROR", v.shape, n)