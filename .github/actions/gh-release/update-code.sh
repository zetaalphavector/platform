#!/bin/bash
set -e

# Ensure we're at the repository root
cd "$(git rev-parse --show-toplevel)"

# Create a temporary directory for cloning the upstream repo
TEMP_DIR=$(mktemp -d)

# Clone the upstream repository into the temporary directory (shallow clone)
git clone --depth 1 --branch v2.5.0 https://github.com/softprops/action-gh-release.git "$TEMP_DIR"

# Sync TypeScript source from upstream
rsync -av \
  --exclude='.*' \
  "$TEMP_DIR/src/" .github/actions/gh-release/src/

# Sync build tooling from upstream
rsync -av \
  "$TEMP_DIR/package.json" .github/actions/gh-release/package.json
rsync -av \
  "$TEMP_DIR/tsconfig.json" .github/actions/gh-release/tsconfig.json

# Remove the temporary directory
rm -rf "$TEMP_DIR"

echo "Files in .github/actions/gh-release updated."
echo ""
echo "Next steps:"
echo "  1. Review the changed code for security vulnerabilities"
echo "  2. Rebuild dist/ by running:"
echo "     cd .github/actions/gh-release && nvm use v20 && yarn && yarn build && yarn package && rm -fr node_modules && rm -fr lib"
