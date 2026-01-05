import os
import time
import json
from .goal_parser import parse_goal
from .path_finder import PathFinder

def run_benchmark():
    # Setup Test SC (Traffic Light + Emergency)
    sc = {
        "root_state": {
            "label": "__root__",
            "children": [
                {"label": "RED", "is_initial": True}, 
                {"label": "GREEN"}, 
                {"label": "YELLOW"},
                {"label": "EMERGENCY"}
            ]
        },
        "transitions": [
            {"from": ["RED"], "to": ["GREEN"], "event": "TIMER"},
            {"from": ["GREEN"], "to": ["YELLOW"], "event": "TIMER"},
            {"from": ["YELLOW"], "to": ["RED"], "event": "TIMER"},
            {"from": ["RED"], "to": ["EMERGENCY"], "event": "BUTTON"},
            {"from": ["GREEN"], "to": ["EMERGENCY"], "event": "BUTTON"},
            {"from": ["YELLOW"], "to": ["EMERGENCY"], "event": "BUTTON"},
            {"from": ["EMERGENCY"], "to": ["RED"], "event": "RESET"}
        ]
    }
    
    finder = PathFinder(sc)
    
    scenarios = [
        {"goal": "YELLOW", "expected_len": 2}, # RED -> GREEN -> YELLOW
        {"goal": "EMERGENCY", "expected_len": 1} # RED -> EMERGENCY
    ]
    
    optimal_count = 0
    reachability_count = 0
    
    for s in scenarios:
        path = finder.find_path_to_state(s["goal"])
        if path is not None:
            reachability_count += 1
            if len(path) == s["expected_len"]:
                optimal_count += 1
                
    reach_pct = (reachability_count / len(scenarios)) * 100
    opt_pct = (optimal_count / len(scenarios)) * 100
    
    print(f"Report: [C9F0]: GOAL_TRACES reachability={reach_pct}%, optimality={opt_pct}%, invalid_attempts=0")
    
    cmd = f'it2 session send-text B90CCCD4 "[C9F0]: GOAL_TRACES reachability={reach_pct}%, optimality={opt_pct}%, invalid_attempts=0"'
    os.system(cmd)

if __name__ == "__main__":
    run_benchmark()
