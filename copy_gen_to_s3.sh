#!/bin/bash
export GEN_S3_BUCKET="s3://trimit-generated/frontend"
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
echo "Uploading files with prefix $GIT_COMMIT_HASH"
pids=""
find frontend/gen -type f -name '*.ts' | while read file; do
  aws s3 cp "$file" "$GEN_S3_BUCKET/$GIT_COMMIT_HASH/${file#frontend/gen/}" &
  pids="$pids $!"
done
for pid in $pids; do
    wait $pid || exit 1
done
