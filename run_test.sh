#!/bin/bash

# Get this path 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ACTIVATION_DIR="$(dirname $DIR)"

cd $DIR
source $ACTIVATION_DIR/venv/bin/activate
python3 tracking.test.py

#  /home/teamdev/FaceRec/testing_app/run_test.sh > /home/teamdev/FaceRec/testing_app/logs/tracking.log 2>&1