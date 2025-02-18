#!/bin/bash

# Default values
DURATION=60
TYPE="IN"

# Parse command line arguments
while getopts "d:t:" opt; do
  case $opt in
    d) DURATION="$OPTARG";;
    t) TYPE="$OPTARG";;
    *) echo "Invalid option"; exit 1;;
  esac
done

# Kill any existing instances of main.py
echo "Checking for existing main.py processes..."
if pgrep -f "python3 main.py" > /dev/null; then
    echo "Killing existing main.py processes..."
    pkill -9 -f "python3 main.py"
    pkill -9 -f "python main.py"
    sleep 1  # Give processes time to clean up
fi

# Get this path 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ACTIVATION_DIR="$(dirname $DIR)"

cd $DIR
source $ACTIVATION_DIR/venv/bin/activate
exec python3 main.py --process_duration "$DURATION" --checkin_type "$TYPE"
