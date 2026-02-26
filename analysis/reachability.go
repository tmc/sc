package analysis

import (
	"sort"

	sc "github.com/tmc/sc/gen/statecharts/v1"
)

// Reachable returns all states reachable from the given start states.
// Uses BFS to traverse the transition graph.
func (g *StateGraph) Reachable(startStates []string) []string {
	if len(startStates) == 0 {
		return nil
	}

	visited := make(map[string]bool)
	queue := make([]string, 0, len(startStates))

	// markState marks a state reachable and applies default-entry semantics.
	var markState func(label string)
	markState = func(label string) {
		node, ok := g.States[label]
		if !ok || visited[label] {
			return
		}
		visited[label] = true
		queue = append(queue, label)

		// On entering composites:
		// - OR states enter exactly one default child.
		// - AND states enter all region children.
		for _, child := range g.defaultEntryChildren(node) {
			markState(child)
		}
	}

	// Initialize with start states.
	for _, s := range startStates {
		markState(s)
	}

	// BFS over transition graph.
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		node := g.States[current]
		if node == nil {
			continue
		}

		// Follow outgoing transitions.
		for _, edge := range node.Outgoing {
			markState(edge.To)
		}
	}

	// Convert to sorted slice.
	var result []string
	for s := range visited {
		if s != "__root__" && s != g.rootLabel {
			result = append(result, s)
		}
	}
	sort.Strings(result)
	return result
}

// ReachableFromInitial returns states reachable from the initial configuration.
func (g *StateGraph) ReachableFromInitial() []string {
	// Enter from root so default-completion rules are applied hierarchically.
	if g.rootLabel != "" {
		if _, ok := g.States[g.rootLabel]; ok {
			return g.Reachable([]string{g.rootLabel})
		}
	}
	if _, ok := g.States["__root__"]; ok {
		return g.Reachable([]string{"__root__"})
	}
	if g.initialState != "" {
		return g.Reachable([]string{g.initialState})
	}
	return nil
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

func (g *StateGraph) defaultEntryChildren(node *StateNode) []string {
	if node == nil || len(node.Children) == 0 {
		return nil
	}
	if node.Type == sc.StateType_STATE_TYPE_AND {
		return g.existingChildren(node.Children)
	}

	// OR semantics: enter one initial child, else first child.
	for _, child := range node.Children {
		childNode, ok := g.States[child]
		if ok && childNode.IsInitial {
			return []string{child}
		}
	}
	for _, child := range node.Children {
		if _, ok := g.States[child]; ok {
			return []string{child}
		}
	}
	return nil
}

func (g *StateGraph) existingChildren(children []string) []string {
	out := make([]string, 0, len(children))
	for _, child := range children {
		if _, ok := g.States[child]; ok {
			out = append(out, child)
		}
	}
	return out
}
