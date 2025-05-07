#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Cleaning up repository..."

# 1. Remove local Python virtualenv
echo "- Removing venv/"
rm -rf venv/

# 2. Remove zipped environments
echo "- Removing any .zip files in envs/"
rm -f envs/*.zip

# 3. Remove extraneous README files
echo "- Removing subfolder README.md files"
rm -f envs/README.md models/README.md notebooks/README.md

# 4. Remove log files
echo "- Deleting all .log files"
find . -type f -name "*.log" -delete

# 5. Remove __pycache__ directories
echo "- Deleting all __pycache__ directories"
find . -type d -name "__pycache__" -exec rm -rf {} +

# 6. Remove stray notebooks and scripts not in rubric
echo "- Removing Crawler.ipynb, train_faster*, run_fast_training.sh"
rm -f notebooks/Crawler.ipynb \
      train_faster.py \
      src/train_faster.py \
      run_fast_training.sh

# 7. (Optional) Remove any other temp/unneeded files
echo "- Cleaning up leftover temporary files"
find . -type f -name "*~" -delete

echo "✅ Cleanup complete."

