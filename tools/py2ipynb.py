"""Konversi .py ber-marker sel (# %% / # %% [markdown]) menjadi .ipynb nbformat 4.

Pakai: python tools/py2ipynb.py <in.py> <out.ipynb>
"""
import json
import sys
from pathlib import Path

MARK = "# %%"


def cells(src: str):
    out, kind, buf = [], "code", []
    for line in src.splitlines():
        if line.startswith(MARK):
            out.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    out.append((kind, "\n".join(buf).strip("\n")))
    return [(k, v) for k, v in out if v.strip()]


def main(src, dst):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, txt in cells(Path(src).read_text(encoding="utf-8")):
        lines = txt.splitlines(keepends=True)
        if kind == "markdown":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            nb["cells"].append({
                "cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": lines,
            })
    Path(dst).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{dst}: {len(nb['cells'])} sel")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
