#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ontology_file="$repo_root/onotology.md"

if [[ ! -f "$ontology_file" ]]; then
  echo "No onotology.md file found at $ontology_file"
  exit 0
fi

echo "===== BEGIN FUNCTION ONTOLOGY ====="
sed -n '1,240p' "$ontology_file"
echo "===== END FUNCTION ONTOLOGY ====="
