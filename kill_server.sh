#!/bin/bash
# Kill all running console-wars server instances

pids=$(pgrep -x "console-wars" 2>/dev/null)

if [ -z "$pids" ]; then
    echo "No console-wars server processes found."
    exit 0
fi

echo "Killing console-wars processes: $pids"
kill $pids
echo "Done."
