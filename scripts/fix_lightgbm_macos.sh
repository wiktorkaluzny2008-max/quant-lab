#!/bin/sh
# macOS only. LightGBM's wheel expects libomp from Homebrew. This points it at the
# libomp that scikit-learn already ships inside the project's .venv instead.
# Only touches .venv (undo with: uv sync --reinstall-package lightgbm). Re-run after a re-install.
set -e
SP=$(uv run python -c "import site; print(site.getsitepackages()[0])")
L="$SP/lightgbm/lib/lib_lightgbm.dylib"
install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs "$L" 2>/dev/null || true
codesign --force --sign - "$L"
uv run python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'OK')"
