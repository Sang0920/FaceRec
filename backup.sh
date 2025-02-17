#!/bin/bash
set -e

# Environment setup
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/home/teamdev

# Change to project directory
cd /home/teamdev/FaceRec/testing_app

# Set log file with absolute path
LOG_FILE="/home/teamdev/FaceRec/testing_app/logs/backup.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Redirect output with timestamps
exec 1> >(while read line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line" > "$LOG_FILE"; done)
exec 2>&1

echo "Starting backup script..."

echo "Adding files..."
/usr/bin/git add .

echo "Committing changes..."
/usr/bin/git commit -m "Backup $(date '+%Y-%m-%d %H:%M:%S')" || true

echo "Pushing to remote..."
/usr/bin/git push origin main --force

echo "Backup complete!"