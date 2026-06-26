#!/usr/bin/env bash
# Download the SkyView dataset from Kaggle and install class folders under data/.
#
# Prerequisites: curl, unzip
#
# Usage (from anywhere):
#   bash scripts/download_dataset.sh
#   bash scripts/download_dataset.sh --force   # re-download even if data/ looks complete

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
ZIP_PATH="${HOME}/Downloads/skyview-an-aerial-landscape-dataset.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/ankit1743/skyview-an-aerial-landscape-dataset"
EXPECTED_CLASSES=15

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

dataset_looks_complete() {
  [[ -d "${DATA_DIR}/Agriculture" && -d "${DATA_DIR}/River" ]]
}

install_extracted_dataset() {
  local extract_dir="$1"
  local content_dir="${extract_dir}"

  if [[ ! -d "${content_dir}/Agriculture" ]]; then
    local candidates=()
    while IFS= read -r -d '' entry; do
      candidates+=("${entry}")
    done < <(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d -print0)

    for candidate in "${candidates[@]}"; do
      if [[ -d "${candidate}/Agriculture" ]]; then
        content_dir="${candidate}"
        break
      fi
    done
  fi

  if [[ ! -d "${content_dir}/Agriculture" ]]; then
    echo "Error: could not find class folders (e.g. Agriculture/) in the archive." >&2
    exit 1
  fi

  mkdir -p "${DATA_DIR}"
  shopt -s nullglob
  for class_dir in "${content_dir}"/*/; do
    class_name="$(basename "${class_dir}")"
    rm -rf "${DATA_DIR}/${class_name}"
    mv "${class_dir}" "${DATA_DIR}/${class_name}"
  done
  shopt -u nullglob

  local class_count
  class_count="$(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  echo "Installed ${class_count} class folders under ${DATA_DIR}"
}

if dataset_looks_complete && [[ "${FORCE}" -eq 0 ]]; then
  echo "Dataset already present in ${DATA_DIR} (use --force to re-download)."
  exit 0
fi

command -v curl >/dev/null 2>&1 || { echo "Error: curl is required." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "Error: unzip is required." >&2; exit 1; }

mkdir -p "${HOME}/Downloads"
echo "Downloading dataset to ${ZIP_PATH} ..."
curl -L -o "${ZIP_PATH}" "${DATASET_URL}"

EXTRACT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skyview-extract.XXXXXX")"
cleanup() {
  rm -rf "${EXTRACT_DIR}"
}
trap cleanup EXIT

echo "Extracting archive ..."
unzip -q "${ZIP_PATH}" -d "${EXTRACT_DIR}"

install_extracted_dataset "${EXTRACT_DIR}"
class_count="$(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"

if [[ "${class_count}" -lt "${EXPECTED_CLASSES}" ]]; then
  echo "Warning: expected ${EXPECTED_CLASSES} class folders; found ${class_count}." >&2
fi

echo "Done. Dataset ready at: ${DATA_DIR}"
