from typing import List, Dict, Set, Optional, Tuple, Any
from collections import deque

class PathFinder:
    def __init__(self, sc: Dict[str, Any]):
        self.sc = sc
        self.transitions = sc.get("transitions", [])
        self.initial_state = self._find_initial()

    def _find_initial(self) -> str:
        root = self.sc.get("root_state", {})
        for child in root.get("children", []):
            if child.get("is_initial"):
                return child.get("label")
        return "UNKNOWN"

    def find_path_to_state(self, target: str) -> Optional[List[str]]:
        """
        BFS to find shortest event sequence to target.
        """
        # Queue: (current_state, path_of_events)
        queue = deque([(self.initial_state, [])])
        visited = {self.initial_state}
        
        while queue:
            curr, path = queue.popleft()
            
            if curr == target:
                return path
            
            # Find neighbors
            for t in self.transitions:
                if curr in t.get("from", []):
                    next_state = t.get("to", [])[0] # Assume single target
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [t.get("event")]))
                        
        return None
