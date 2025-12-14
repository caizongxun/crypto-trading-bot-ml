#!/bin/bash
# Cleanup old versions and keep only V8

echo "Cleaning up old versions..."

# Remove old model versions
rm -f train_v*.py
rm -f train_lstm*.py
rm -f train_tft*.py
rm -f visualize_*.py
rm -f predict_*.py

# Remove only old versions (NOT _v8)
ls train_*.py 2>/dev/null | grep -v 'v8' | xargs rm -f
ls visualize_*.py 2>/dev/null | grep -v 'v8' | xargs rm -f
ls predict_*.py 2>/dev/null | grep -v 'v8' | xargs rm -f

# Remove test and debug files
rm -f test_*.py
rm -f debug_*.py
rm -f *_test.py
rm -f *.backup

# Keep diagnostic tools
echo "Keeping V8 files and diagnostic tools..."

# List remaining Python files
echo ""
echo "Remaining Python files:"
ls -1 *.py | grep -E '(train|visualize|predict|diagnose|detect|bot|hf)'

echo ""
echo "Cleanup complete!"
