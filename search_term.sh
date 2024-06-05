#!/bin/bash

# Search term
SEARCH_TERM="and_de"
IGNORE_FILE="search_term.sh"

# Loop through each entry in the reflog
git reflog | head -n 100 | while read line; do
  # Extract the commit hash
  COMMIT_HASH=$(echo $line | awk '{print $1}')

  # Check out the code at that commit in a detached state
  git checkout $COMMIT_HASH &>/dev/null 2>&1
  SEARCH_OUTPUT=$(rg -i "$SEARCH_TERM" . --glob "!$IGNORE_FILE")

  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Found '$SEARCH_TERM' in commit $COMMIT_HASH"
    echo "$SEARCH_OUTPUT"
    echo "Associated reflog entry: $line"
    break
  fi

done

# Checkout back to the original branch
git checkout main

