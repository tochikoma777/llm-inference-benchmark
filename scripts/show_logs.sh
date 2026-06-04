#!/bin/bash
LOG_DIR="logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "No logs directory found."
    exit 1
fi

echo "Available log files:"
echo "-------------------"
ls -lt "$LOG_DIR" | awk '{print $6, $7, $8, "\t", $9}'
echo ""
echo "To view a log: cat logs/<filename>"
echo "To tail a running log: tail -f logs/<filename>"
