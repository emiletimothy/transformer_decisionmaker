#!/bin/bash
# Monitor think1024 progress, kill after weather completes
cd /home/jupyter-xcao/mwu

while true; do
    # Count weather completed
    weather_done=$(find exp_results/weather/qwen3_14b_think1024 -name "result.json" 2>/dev/null | wc -l)
    echo "[$(date)] think1024 weather: $weather_done/20"

    if [ "$weather_done" -ge 20 ]; then
        echo "[$(date)] Weather complete! Killing think1024..."
        ps aux | grep "interactive_api_run.*think1024" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null
        # Also kill the parent sequence script
        ps aux | grep "run_qwen3_all" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null
        echo "[$(date)] Done."
        break
    fi

    sleep 60
done
