#!/bin/bash
# postCreateCommand script for devcontainer initialization
# Requirements: 2.3, 2.5, 3.5, 4.1, 4.2, 4.3, 5.6, 6.6, 9.4

set +e  # Continue on errors

echo "===== Starting postCreateCommand ====="

# 1. uv sync (Req 2.3)
echo ""
echo "Running uv sync..."
if uv sync; then
    echo "✓ uv sync completed successfully"
else
    echo "⚠️  Warning: uv sync failed"
fi

# 2. GitHub CLI credential helper setup (Req 4.1, 4.2)
echo ""
echo "Setting up GitHub CLI credential helper..."
if gh auth setup-git 2>&1; then
    echo "✓ GitHub CLI credential helper configured"
else
    echo "⚠️  Warning: gh auth setup-git failed. Run 'gh auth login' on host machine first, then restart container."
fi

# 3. Git submodule initialization (Req 4.1, 4.2, 4.3)
echo ""
echo "Initializing Git submodules..."
if git submodule update --init --recursive; then
    echo "✓ Git submodules initialized successfully"
else
    echo "❌ Error: Git submodule initialization failed (Req 4.3). PSDS_Eval and sebbs may be unavailable."
    echo "   Retry manually: git submodule update --init --recursive"
    echo "   Check GitHub authentication: gh auth status"
fi

# 4. pre-commit install (Req 5.6)
echo ""
echo "Installing pre-commit hooks..."
if pre-commit install; then
    echo "✓ pre-commit hooks installed successfully"
else
    echo "⚠️  Warning: pre-commit install failed. Git hooks not activated. Install manually if needed."
fi

# 5. System libraries verification (Req 3.5)
echo ""
echo "Verifying system libraries..."
libs_ok=true
for lib in sox ffmpeg git gh; do
    if which "$lib" > /dev/null 2>&1; then
        echo "  ✓ $lib found"
    else
        echo "  ❌ $lib not found"
        libs_ok=false
    fi
done

if $libs_ok; then
    echo "✓ All system libraries verified"
else
    echo "⚠️  Warning: System libraries missing"
fi

# 6. Directory write permissions verification (Req 6.6)
echo ""
echo "Verifying directory write permissions..."
dirs_ok=true

# Determine base path (prefer /workspace if exists, otherwise current directory)
if [ -d "/workspace" ]; then
    base_path="/workspace"
else
    base_path="."
fi

for dir in data embeddings exp wandb; do
    full_path="$base_path/$dir"
    if [ -d "$full_path" ]; then
        if touch "$full_path/.test_write" 2>/dev/null && rm "$full_path/.test_write" 2>/dev/null; then
            echo "  ✓ $dir writable"
        else
            echo "  ⚠️  Warning: $dir not writable. Data persistence may fail. Check volume mounts."
            dirs_ok=false
        fi
    else
        echo "  ℹ️  Info: $dir directory not found. Creating..."
        if mkdir -p "$full_path" 2>/dev/null; then
            if touch "$full_path/.test_write" 2>/dev/null && rm "$full_path/.test_write" 2>/dev/null; then
                echo "  ✓ $dir created and writable"
            else
                echo "  ⚠️  Warning: $dir not writable after creation"
                dirs_ok=false
            fi
        else
            echo "  ⚠️  Warning: Cannot create $dir directory"
            dirs_ok=false
        fi
    fi
done

if $dirs_ok; then
    echo "✓ All directories writable"
fi

# 7. Resource checks (Req 9.4)
echo ""
echo "Checking system resources..."

# Memory check
if command -v free > /dev/null 2>&1; then
    available_mem=$(free -m | awk '/^Mem:/{print $7}')
    if [ -n "$available_mem" ] && [ "$available_mem" -lt 2048 ]; then
        echo "⚠️  Warning: Low memory detected ($available_mem MB available). Training may fail. Increase Docker memory limit."
    else
        echo "✓ Memory: $available_mem MB available"
    fi
else
    echo "ℹ️  Info: 'free' command not available (likely macOS host). Memory check skipped."
fi

# Disk space check
if df -k /workspace > /dev/null 2>&1; then
    # Convert KB to GB
    available_disk=$(df -k /workspace | awk 'NR==2{print int($4/1024/1024)}')
    if [ -n "$available_disk" ] && [ "$available_disk" -lt 10 ]; then
        echo "⚠️  Warning: Low disk space ($available_disk GB available). Data storage may fail. Clean up volumes or increase disk."
    else
        echo "✓ Disk: $available_disk GB available"
    fi
else
    echo "ℹ️  Info: /workspace not accessible. Disk check skipped."
fi

echo ""
echo "===== postCreateCommand completed ====="
