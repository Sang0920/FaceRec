#!/bin/bash
set -e

echo "Checking Git status..."
if ! git diff-index --quiet HEAD --; then
    read -p "There are uncommitted changes. Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

echo "Adding files..."
git add .

echo "Committing changes..."
git commit -m "Backup $(date '+%Y-%m-%d %H:%M:%S')"

echo "Pushing to remote..."
git push origin main --force

echo "Backup complete!"