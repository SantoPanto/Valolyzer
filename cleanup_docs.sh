#!/bin/bash

# Valolyzer Documentation Cleanup Script
# This script organizes auxiliary documentation files into a docs/ folder
# Keeps README.md and requirements.txt in the root directory

set -e  # Exit on error

echo "🚀 Starting Valolyzer documentation cleanup..."

# Create docs folder if it doesn't exist
if [ ! -d "docs" ]; then
    mkdir -p docs
    echo "✅ Created docs/ folder"
else
    echo "✅ docs/ folder already exists"
fi

# Move all .md files EXCEPT README.md
echo "📄 Moving markdown files to docs/..."
for file in *.md; do
    if [ "$file" != "README.md" ]; then
        if [ -f "$file" ]; then
            mv "$file" docs/
            echo "  → Moved $file"
        fi
    fi
done

# Move all .ipynb files (Jupyter notebooks)
echo "📓 Moving Jupyter notebooks to docs/..."
for file in *.ipynb; do
    if [ -f "$file" ]; then
        mv "$file" docs/
        echo "  → Moved $file"
    fi
done

# Move .txt files EXCEPT requirements.txt
echo "📝 Moving text files to docs/..."
for file in *.txt; do
    if [ "$file" != "requirements.txt" ]; then
        if [ -f "$file" ]; then
            mv "$file" docs/
            echo "  → Moved $file"
        fi
    fi
done

echo ""
echo "✨ Cleanup complete! Root directory is now organized."
echo "📁 All auxiliary documentation is in docs/ folder"
echo "📌 Essential files remain in root: README.md, requirements.txt"
