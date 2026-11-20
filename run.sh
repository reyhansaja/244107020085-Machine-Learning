#!/usr/bin/env bash
# Eksekusi semua notebook (output tersimpan di dalam .ipynb).
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
[ -x "$PY" ] || PY=./.venv/bin/python
fail=0
for nb in $(ls -1 js*/*/*.ipynb | sort); do
  echo "== $nb"
  if "$PY" -m nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=2400 "$nb" > /dev/null 2>&1; then
    echo "   ok"
  else
    echo "   GAGAL"
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "SEMUA NOTEBOOK OK" || echo "ADA YANG GAGAL"
exit $fail
