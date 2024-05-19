#!/bin/bash

# Define the path to the pre-commit hook
HOOK_NAME=pre-commit
HOOKS_DIR=$(git rev-parse --show-toplevel)/.git/hooks
HOOK_SCRIPT_PATH=$HOOKS_DIR/$HOOK_NAME

# Copy the pre-commit script to the .git/hooks directory
cp ./hooks/pre-commit-script $HOOK_SCRIPT_PATH

# Make sure the hook script is executable
chmod +x $HOOK_SCRIPT_PATH

echo "Pre-commit hook installed successfully."
