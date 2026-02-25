#!/bin/bash
set -e

# Ensure we're at the repository root
cd "$(git rev-parse --show-toplevel)"

# Create a temporary directory for cloning the upstream repo
TEMP_DIR=$(mktemp -d)

# Clone the upstream repository into the temporary directory (shallow clone)
git clone --depth 1 --branch v3.0.0 https://github.com/sigstore/gh-action-sigstore-python.git "$TEMP_DIR"

# Sync the Python action script
rsync -av \
  "$TEMP_DIR/action.py" .github/actions/sigstore/action.py

# Sync templates
rsync -av \
  --exclude='.*' \
  "$TEMP_DIR/templates/" .github/actions/sigstore/templates/

# Sync requirements
rsync -av \
  --exclude='.*' \
  "$TEMP_DIR/requirements/" .github/actions/sigstore/requirements/

# Remove the temporary directory
rm -rf "$TEMP_DIR"

echo "Files in .github/actions/sigstore updated from upstream v3.0.0."
echo ""
echo "Next steps:"
echo "  1. Review the changed code for security vulnerabilities"
echo "  2. Review action.yml — it must NOT reference any external actions"
echo "     (softprops/action-gh-release should be ./.github/actions/gh-release)"
