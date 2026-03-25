package analysis

import "sort"

// Reachable returns all states reachable from the given start states.
// Uses BFS to traverse the transition graph.
func (g *StateGraph) Reachable(startStates []string) []string {
	if len(startStates) == 0 {
		return nil
	}

	visited := make(map[string]bool)
	queue := make([]string, 0, len(startStates))

	// Initialize with start states
	for _, s := range startStates {
		if _, ok := g.States[s]; ok {
			visited[s] = true
			queue = append(queue, s)
		}
	}

	// BFS
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		node := g.States[current]
		if node == nil {
			continue
		}

		// Follow outgoing transitions
		for _, edge := range node.Outgoing {
			if !visited[edge.To] {
				visited[edge.To] = true
				queue = append(queue, edge.To)
			}
		}

		// Also include children of composite states (default entries)
		for _, child := range node.Children {
			if !visited[child] {
				visited[child] = true
				queue = append(queue, child)
			}
		}
	}

	// Convert to sorted slice
	var result []string
	for s := range visited {
		if s != "__root__" {
			result = append(result, s)
		}
	}
	sort.Strings(result)
	return result
}

// ReachableFromInitial returns states reachable from the initial configuration.
func (g *StateGraph) ReachableFromInitial() []string {
	// Find all initial states (could be multiple in parallel regions)
	var initials []string
	for label, node := range g.States {
		if node.IsInitial {
			initials = append(initials, label)
		}
	}

	// If no explicit initial, use root's first child
	if len(initials) == 0 && g.initialState != "" {
		initials = []string{g.initialState}
	}

	// If still nothing, try to find it from root
	if len(initials) == 0 {
		if root, ok := g.States["__root__"]; ok {
			for _, child := range root.Children {
				if childNode, ok := g.States[child]; ok && childNode.IsInitial {
					initials = append(initials, child)
				}
			}
			// If no initial marked, take first child
			if len(initials) == 0 && len(root.Children) > 0 {
				initials = append(initials, root.Children[0])
			}
		}
	}

	return g.Reachable(initials)
}

// Unreachable returns states not reachable from the initial configuration.
func (g *StateGraph) Unreachable() []string {
	reachable := make(map[string]bool)
	for _, s := range g.ReachableFromInitial() {
		reachable[s] = true
	}

	var unreachable []string
	for label := range g.States {
		if label == "__root__" {
			continue
		}
		if !reachable[label] {
			unreachable = append(unreachable, label)
		}
	}
	sort.Strings(unreachable)
	return unreachable
}

// CanReachFinal returns states that can reach at least one final state.
// Uses reverse BFS from final states.
func (g *StateGraph) CanReachFinal() []string {
	finals := g.FinalStates()
	if len(finals) == 0 {
		return nil
	}

	visited := make(map[string]bool)
	queue := make([]string, 0, len(finals))

	// Initialize with final states
	for _, s := range finals {
		visited[s] = true
		queue = append(queue, s)
	}

	// Reverse BFS (follow incoming edges)
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		node := g.States[current]
		if node == nil {
			continue
		}

		// Follow incoming transitions
		for _, edge := range node.Incoming {
			if !visited[edge.From] {
				visited[edge.From] = true
				queue = append(queue, edge.From)
			}
		}

		// Also include parent (can enter child via default)
		if node.Parent != "" && node.Parent != "__root__" && !visited[node.Parent] {
			visited[node.Parent] = true
			queue = append(queue, node.Parent)
		}
	}

	var result []string
	for s := range visited {
		if s != "__root__" {
			result = append(result, s)
		}
	}
	sort.Strings(result)
	return result
}

// DeadEnds returns states that cannot reach any final state.
func (g *StateGraph) DeadEnds() []string {
	canReach := make(map[string]bool)
	for _, s := range g.CanReachFinal() {
		canReach[s] = true
	}

	// Only consider reachable states as dead ends
	reachable := make(map[string]bool)
	for _, s := range g.ReachableFromInitial() {
		reachable[s] = true
	}

	var deadEnds []string
	for label := range g.States {
		if label == "__root__" {
			continue
		}
		// Dead end = reachable but can't reach final
		if reachable[label] && !canReach[label] {
			deadEnds = append(deadEnds, label)
		}
	}
	sort.Strings(deadEnds)
	return deadEnds
}
