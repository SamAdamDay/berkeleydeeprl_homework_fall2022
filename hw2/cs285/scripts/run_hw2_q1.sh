#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python "$SCRIPT_DIR/run_hw2.py" --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa || exit 1
python "$SCRIPT_DIR/run_hw2.py" --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa || exit 1
python "$SCRIPT_DIR/run_hw2.py" --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na || exit 1
python "$SCRIPT_DIR/run_hw2.py" --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa || exit 1
python "$SCRIPT_DIR/run_hw2.py" --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa || exit 1
python "$SCRIPT_DIR/run_hw2.py" --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na || exit 1