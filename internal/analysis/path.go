package analysis

// Path represents a sequence of transitions from source to target.
type Path struct {
	States      []string
	Transitions []*TransitionEdge
}

// bfsNode tracks parent info during BFS path search.
type bfsNode struct {
	state string
	edge  *TransitionEdge
}

// Length returns the number of transitions in the path.
func (p *Path) Length() int {
	return len(p.Transitions)
}

// FindPath finds a path from source to target using BFS (shortest path).
// Returns nil if no path exists.
func (g *StateGraph) FindPath(from, to string) *Path {
	if from == to {
		return &Path{States: []string{from}}
	}

	// BFS with parent tracking
	visited := make(map[string]bool)
	parent := make(map[string]bfsNode)
	queue := []string{from}
	visited[from] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		node := g.States[current]
		if node == nil {
			continue
		}

		for _, edge := range node.Outgoing {
			if !visited[edge.To] {
				visited[edge.To] = true
				parent[edge.To] = bfsNode{state: current, edge: edge}
				queue = append(queue, edge.To)

				if edge.To == to {
					// Reconstruct path
					return g.reconstructPath(from, to, parent)
				}
			}
		}
	}

	return nil // No path found
}

func (g *StateGraph) reconstructPath(from, to string, parent map[string]bfsNode) *Path {
	path := &Path{}

	// Build path in reverse
	var states []string
	var transitions []*TransitionEdge

	current := to
	for current != from {
		states = append([]string{current}, states...)
		p := parent[current]
		transitions = append([]*TransitionEdge{p.edge}, transitions...)
		current = p.state
	}
	states = append([]string{from}, states...)

	path.States = states
	path.Transitions = transitions
	return path
}

// FindAllPaths finds all paths from source to target up to maxDepth.
// If maxDepth <= 0, uses a default limit of 10.
func (g *StateGraph) FindAllPaths(from, to string, maxDepth int) []*Path {
	if maxDepth <= 0 {
		maxDepth = 10
	}

	var paths []*Path
	visited := make(map[string]bool)

	var dfs func(current string, path []string, edges []*TransitionEdge, depth int)
	dfs = func(current string, pathStates []string, pathEdges []*TransitionEdge, depth int) {
		if depth > maxDepth {
			return
		}

		if current == to {
			// Found a path
			p := &Path{
				States:      make([]string, len(pathStates)),
				Transitions: make([]*TransitionEdge, len(pathEdges)),
			}
			copy(p.States, pathStates)
			copy(p.Transitions, pathEdges)
			paths = append(paths, p)
			return
		}

		node := g.States[current]
		if node == nil {
			return
		}

		visited[current] = true
		for _, edge := range node.Outgoing {
			if !visited[edge.To] {
				newStates := append(pathStates, edge.To)
				newEdges := append(pathEdges, edge)
				dfs(edge.To, newStates, newEdges, depth+1)
			}
		}
		visited[current] = false
	}

	dfs(from, []string{from}, nil, 0)
	return paths
}

// ShortestPathLength returns the length of the shortest path between two states.
// Returns -1 if no path exists.
func (g *StateGraph) ShortestPathLength(from, to string) int {
	path := g.FindPath(from, to)
	if path == nil {
		return -1
	}
	return path.Length()
}
