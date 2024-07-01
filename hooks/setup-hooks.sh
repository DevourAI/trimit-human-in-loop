#!/bin/bash

# Define the path to the pre-commit hook
PRECOMMIT_HOOK_NAME=pre-commit
PREPUSH_HOOK_NAME=pre-push
HOOKS_DIR=$(git rev-parse --show-toplevel)/.git/hooks
PRECOMMIT_HOOK_SCRIPT_PATH=$HOOKS_DIR/$PRECOMMIT_HOOK_NAME
PREPUSH_HOOK_SCRIPT_PATH=$HOOKS_DIR/$PREPUSH_HOOK_NAME

# Copy the pre-commit script to the .git/hooks directory
cp ./hooks/pre-commit-script $PRECOMMIT_HOOK_SCRIPT_PATH
cp ./hooks/pre-push-script $PREPUSH_HOOK_SCRIPT_PATH

# Make sure the hook script is executable
chmod +x $PRECOMMIT_HOOK_SCRIPT_PATH
chmod +x $PREPUSH_HOOK_SCRIPT_PATH

echo "Hooks installed successfully."
