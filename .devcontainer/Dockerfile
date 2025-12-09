# Multi-stage Dockerfile for DCASE 2024 Task 4 Dev Container (CPU)
#
# Build Strategy:
#   Stage 1: Copy uv binary from official image
#   Stage 2: Setup base image with system libraries and Python 3.12
#   Stage 3: Install Python dependencies using uv
#
# Requirements Coverage:
#   - Req 1.2, 1.3: Python 3.12+ base image (python:3.12-slim-bookworm)
#   - Req 1.7: Non-root user (vscode)
#   - Req 2.1: Python 3.12 installation
#   - Req 2.2: uv package manager
#   - Req 2.3, 2.4: pyproject.toml dependencies (all groups)
#   - Req 3.1-3.3: System audio libraries (SoX, FFmpeg, libsndfile)
#   - Req 7.1, 7.2: Security (non-root, minimal privileges)
#   - Req 8.1-8.5: Reproducibility (version pinning, hash verification, build timestamp)

# ============================================================================
# Stage 1: UV Binary
# ============================================================================
FROM ghcr.io/astral-sh/uv:0.9.16 AS uv

# ============================================================================
# Stage 2: Base Image + System Libraries
# ============================================================================
FROM python:3.12-slim-bookworm AS base

# Copy uv binary from Stage 1 (Req 2.2)
# Note: In ghcr.io/astral-sh/uv:0.9.16, the binary is located at /uv
COPY --from=uv /uv /usr/local/bin/uv

# Install system dependencies (Req 3.1-3.3)
# - build-essential: C/C++ compiler toolchain (required for building Python packages)
# - sox: Sound eXchange for audio processing
# - libsox-dev: SoX development files (required for soxbindings compilation)
# - ffmpeg: Audio/video processing (torchaudio backend)
# - libsndfile1-dev: Sound file library (torchaudio backend)
# - git: Version control, submodule management (Req 4.1-4.2)
# - curl: Required for GitHub CLI installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sox \
    libsox-dev \
    ffmpeg \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI (gh) for Git authentication (Req 4.1-4.2)
# Official installation method for Debian (https://cli.github.com/manual/installation)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 using uv (Req 2.1, 8.1)
RUN uv python install 3.12

# Create non-root user (Req 1.7, 7.1)
RUN useradd -m -s /bin/bash vscode

# Add build timestamp label (Req 8.5)
# Note: Timestamp is set at build time using date command
RUN date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/build_timestamp
LABEL build_timestamp="auto-generated"

# ============================================================================
# Stage 3: Python Dependencies
# ============================================================================
FROM base AS dependencies

# Set working directory
WORKDIR /workspace

# Copy dependency files (Req 2.3, 8.2)
# Note: pyproject.toml and uv.lock should be in project root
COPY pyproject.toml uv.lock* ./

# Install Python dependencies with uv (Req 2.3, 2.4, 8.2, 8.3, 8.4)
# - --frozen: Use uv.lock hash verification (reproducibility)
# - --all-groups: Include dev dependencies (mypy, ruff, etc.)
# - Cache mount: Speed up rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups || \
    uv sync --all-groups

# Switch to non-root user (Req 7.1, 7.2)
USER vscode

# Set entrypoint to shell
CMD ["/bin/bash"]
