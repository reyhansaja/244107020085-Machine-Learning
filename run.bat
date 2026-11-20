@echo off
cd /d "%~dp0"
for /r %%f in (*.ipynb) do (
  echo == %%f
  .venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace "%%f"
)
