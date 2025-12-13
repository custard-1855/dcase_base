  echo "=== OS ===" && cat /etc/os-release | grep -E "^(NAME|VERSION)=" && \
  echo -e "\n=== Kernel ===" && uname -r && \
  echo -e "\n=== CPU ===" && lscpu | grep -E "^(Modelname|Architecture|CPU\(s\)|Thread)" && \
  echo -e "\n=== Memory ===" && free -h | grep "Mem:" && \
  echo -e "\n=== GPU ===" && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv && \
  echo -e "\n=== CUDA ===" && nvcc --version | grep "release" && \
  echo -e "\n=== Python ===" && uv run python --version && \
  echo -e "\n=== Key Packages ===" && uv pip list | grep -E "(torch|tensorflow|numpy|pandas|librosa|scipy|sed-eval|psds-eval|soundfile)"
