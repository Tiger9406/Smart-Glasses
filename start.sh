#!/bin/bash

PYTHON=$(command -v python || command -v python3)
DIR="$(cd "$(dirname "$0")" && pwd)"

# Open main.py in a new Terminal tab
osascript <<EOF
tell application "Terminal"
    activate
    tell application "System Events" to keystroke "t" using command down
    delay 0.5
    do script "cd '$DIR' && $PYTHON main.py" in front window
end tell
EOF

# Wait for server to come up, then open the browser
sleep 3
open http://localhost:8765

# Wait a bit then open simulator in another Terminal tab
sleep 2
osascript <<EOF
tell application "Terminal"
    activate
    tell application "System Events" to keystroke "t" using command down
    delay 0.5
    do script "cd '$DIR' && $PYTHON api/simulator.py" in front window
end tell
EOF
