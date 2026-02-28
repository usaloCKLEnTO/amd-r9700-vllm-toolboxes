# Persist kernel compilation caches across container restarts.
# Toolbox mounts /home/numble from the host, so caches stored there
# survive container recreation. Without this, HIPRTC/Triton/comgr
# recompile FP8 kernels on every cold start (~3 min penalty).

_VLLM_CACHE_BASE="/home/numble/.cache/vllm-kernels"
mkdir -p "${_VLLM_CACHE_BASE}/triton" \
         "${_VLLM_CACHE_BASE}/torchinductor" \
         "${_VLLM_CACHE_BASE}/comgr" \
         "${_VLLM_CACHE_BASE}/miopen" 2>/dev/null

# Triton JIT kernel cache
export TRITON_CACHE_DIR="${_VLLM_CACHE_BASE}/triton"

# Torch inductor compiled kernels (used even with --enforce-eager
# for Triton attention kernels)
export TORCHINDUCTOR_CACHE_DIR="${_VLLM_CACHE_BASE}/torchinductor"

# AMD Code Object Manager (HIPRTC compiled GPU code)
export AMD_COMGR_CACHE="${_VLLM_CACHE_BASE}/comgr"

# MIOpen convolution/GEMM kernel cache
export MIOPEN_USER_DB_PATH="${_VLLM_CACHE_BASE}/miopen"

unset _VLLM_CACHE_BASE
