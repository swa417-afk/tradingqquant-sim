#!/usr/bin/env bash
set -euo pipefail

# Export Astrid assets into a standalone folder that can be initialized
# as a separate repository (e.g., Astrid_Hackathon).

OUTPUT_DIR="${1:-../Astrid_Hackathon}"

mkdir -p "$OUTPUT_DIR"

cp -f ASTRID.pdf "$OUTPUT_DIR/"
cp -f astrid_docs_bundle.zip "$OUTPUT_DIR/"
cp -f astrid_eval_skeleton.csv "$OUTPUT_DIR/"

cat > "$OUTPUT_DIR/README.md" <<'EOF'
# Astrid_Hackathon

This repository contains Astrid-specific artifacts extracted from the original
tradingqquant-sim repository.

## Included files
- ASTRID.pdf
- astrid_docs_bundle.zip
- astrid_eval_skeleton.csv
EOF

echo "Astrid bundle exported to: $OUTPUT_DIR"
echo "Next steps:"
echo "  cd $OUTPUT_DIR"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial import of Astrid artifacts'"
echo "  git remote add origin <new-repo-url>"
echo "  git push -u origin main"
