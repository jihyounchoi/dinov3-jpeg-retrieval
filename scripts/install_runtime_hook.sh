#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dinov3-jpeg-retrieval}"

if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${ENV_NAME}" ]]; then
  TARGET_PREFIX="${CONDA_PREFIX}"
else
  TARGET_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
fi

STATIC_ITT="${TARGET_PREFIX}/lib/libittnotify.a"
PRELOAD_SO="${TARGET_PREFIX}/lib/libittnotify_preload.so"
ACTIVATE_DIR="${TARGET_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${TARGET_PREFIX}/etc/conda/deactivate.d"

if [[ ! -f "${STATIC_ITT}" ]]; then
  echo "Missing ${STATIC_ITT}. Install ittapi in ${ENV_NAME} first." >&2
  exit 1
fi

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"
gcc -shared -o "${PRELOAD_SO}" -Wl,--whole-archive "${STATIC_ITT}" -Wl,--no-whole-archive

cat > "${ACTIVATE_DIR}/ads_dinov3_itt_preload.sh" <<'HOOK'
export _ADS_DINOV3_OLD_LD_PRELOAD="${LD_PRELOAD:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify_preload.so${LD_PRELOAD:+:${LD_PRELOAD}}"
HOOK

cat > "${DEACTIVATE_DIR}/ads_dinov3_itt_preload.sh" <<'HOOK'
if [ -n "${_ADS_DINOV3_OLD_LD_PRELOAD+x}" ]; then
  export LD_PRELOAD="${_ADS_DINOV3_OLD_LD_PRELOAD}"
  unset _ADS_DINOV3_OLD_LD_PRELOAD
else
  unset LD_PRELOAD
fi
HOOK

chmod 755 "${ACTIVATE_DIR}/ads_dinov3_itt_preload.sh" "${DEACTIVATE_DIR}/ads_dinov3_itt_preload.sh"
echo "Installed ITT preload hook in ${TARGET_PREFIX}"
