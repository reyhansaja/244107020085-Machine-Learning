"""Cek notebook: jumlah sel error, jumlah output PNG, dan cetak output teks penting.

Pakai: python tools/check_nb.py <nb.ipynb> [filter]
"""
import json
import sys
from pathlib import Path

nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_filter = sys.argv[2] if len(sys.argv) > 2 else None

errors, pngs, texts = 0, 0, []
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            errors += 1
            print(f"ERROR sel {i}: {o.get('ename')}: {o.get('evalue')}")
        if o.get("output_type") in ("display_data", "execute_result"):
            if "image/png" in o.get("data", {}):
                pngs += 1
        if o.get("output_type") == "stream":
            t = o.get("text", []); texts.append("".join(t) if isinstance(t, list) else str(t))
        if "text/plain" in o.get("data", {}):
            t = o["data"]["text/plain"]; texts.append("".join(t) if isinstance(t, list) else str(t))

body = "\n".join(texts)
print(f"\n== {Path(sys.argv[1]).name}: sel={len(nb['cells'])} error={errors} png={pngs} ==")
if out_filter:
    press = [ln for ln in body.splitlines() if out_filter in ln]
    print("\n".join(press[:60]) if press else "(tidak ada baris cocok)")
