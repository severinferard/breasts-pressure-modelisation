#!/bin/bash

export QT_QPA_PLATFORM=xcb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


if [ -z "$DEV" ]; then
    if nc -zw1 google.com 443; then
        git -C $SCRIPT_DIR fetch origin
        git -C $SCRIPT_DIR reset --hard origin/main
    fi
fi


source "$SCRIPT_DIR/.venv/bin/activate"

# Kill all processes that could have the serial port opened
fuser -k /dev/ttyACM*

python3 "$SCRIPT_DIR/live-rpi.py"