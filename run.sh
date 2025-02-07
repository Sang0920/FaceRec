#!/bin/bash

# Assign permissions for ./run_main.sh & ./setup.py & ./main.py 
chmod +x ./run_main.sh
chmod +x ./setup.py
chmod +x ./main.py
chmod +x ./run.sh

python ./setup.py
echo "Setup complete"