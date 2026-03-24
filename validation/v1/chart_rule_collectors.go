package validation

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/sc"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
)

type validationIssue struct {
	message string
	xpath   []string
}

type statePathRecord struct {
	path   string
	parent string
}

type eventBroadcastEdge struct {
	from           string
	to             string
	transitionPath string
}

func issuef(xpath []string, format string, args ...interface{}) validationIssue {
	return validationIssue{
		message: fmt.Sprintf(format, args...),
		xpath:   uniqueNonEmptyStrings(xpath),
	}
}

func firstIssueError(issues []validationIssue) error {
	if len(issues) == 0 {
		return nil
	}
	return errors.New(issues[0].message)
}

func issuesToViolations(rule validationv1.RuleId, severity validationv1.Severity, prefix string, issues []validationIssue) []*validationv1.Violation {
	violations := make([]*validationv1.Violation, 0, len(issues))
	for _, issue := range issues {
		message := issue.message
		if prefix != "" {
			message = prefix + issue.message
		}
		violations = append(violations, &validationv1.Violation{
			Rule:     rule,
			Severity: severity,
			Message:  message,
			Xpath:    append([]string(nil), issue.xpath...),
		})
	}
	return violations
}

func rootStatePath() string {
	return "/root_state"
}

func childStatePath(parentPath string, index int) string {
	return fmt.Sprintf("%s/children[%d]", parentPath, index)
}

func transitionXPath(index int) string {
	return fmt.Sprintf("/transitions[%d]", index)
}

func transitionActionXPath(transitionIndex, actionIndex int) string {
	return fmt.Sprintf("%s/actions[%d]", transitionXPath(transitionIndex), actionIndex)
}

func eventXPath(index int) string {
	return fmt.Sprintf("/events[%d]", index)
}

func walkStates(root *sc.State, visit func(state *sc.State, path string, parent *sc.State, parentPath string)) {
	var walk func(state *sc.State, path string, parent *sc.State, parentPath string)
	walk = func(state *sc.State, path string, parent *sc.State, parentPath string) {
		if state == nil {
			return
		}
		visit(state, path, parent, parentPath)
		for i, child := range state.Children {
			walk(child, childStatePath(path, i), state, path)
		}
	}
	walk(root, rootStatePath(), nil, "")
}

func collectStateRecords(root *sc.State) map[string][]statePathRecord {
	records := make(map[string][]statePathRecord)
	if root == nil {
		return records
	}
	walkStates(root, func(state *sc.State, path string, parent *sc.State, parentPath string) {
		records[state.Label] = append(records[state.Label], statePathRecord{
			path:   path,
			parent: parentPath,
		})
	})
	return records
}

func collectDescendantPaths(state *sc.State, path string, descendants map[string]string) {
	if state == nil {
		return
	}
	descendants[state.Label] = path
	for i, child := range state.Children {
		collectDescendantPaths(child, childStatePath(path, i), descendants)
	}
}

func generatedEventsForValidation(transition *sc.Transition) []string {
	if transition == nil {
		return nil
	}

	set := make(map[string]bool)
	for _, action := range transition.Actions {
		if action == nil {
			continue
		}
		label := action.GetLabel()
		for _, prefix := range []string{"raise:", "emit:", "send:"} {
			if !strings.HasPrefix(label, prefix) {
				continue
			}
			event := strings.TrimSpace(strings.TrimPrefix(label, prefix))
			if event != "" {
				set[event] = true
			}
			break
		}
	}

	var events []string
	for event := range set {
		events = append(events, event)
	}
	sort.Strings(events)
	return events
}

func transitionNameAt(transition *sc.Transition, index int) string {
	if transition == nil || transition.Label == "" {
		return fmt.Sprintf("transition[%d]", index)
	}
	return transition.Label
}

func uniqueNonEmptyStrings(items []string) []string {
	seen := make(map[string]bool)
	out := make([]string, 0, len(items))
	for _, item := range items {
		if item == "" || seen[item] {
			continue
		}
		seen[item] = true
		out = append(out, item)
	}
	return out
}

func canonicalCycleKey(events, paths []string) string {
	if len(events) == 0 {
		return strings.Join(sortedCopy(paths), "|")
	}

	nodes := append([]string(nil), events...)
	if len(nodes) > 1 && nodes[0] == nodes[len(nodes)-1] {
		nodes = nodes[:len(nodes)-1]
	}
	if len(nodes) == 0 {
		return strings.Join(sortedCopy(paths), "|")
	}

	best := ""
	for i := range nodes {
		rotation := append(append([]string(nil), nodes[i:]...), nodes[:i]...)
		candidate := strings.Join(rotation, "->")
		if best == "" || candidate < best {
			best = candidate
		}
	}

	return best + "|" + strings.Join(sortedCopy(paths), "|")
}

func collectUniqueStateLabelIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "root state is nil")}
	}

	var issues []validationIssue
	labels := make(map[string]string)
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if firstPath, ok := labels[state.Label]; ok {
			issues = append(issues, issuef([]string{firstPath, path}, "duplicate state label: %s", state.Label))
			return
		}
		labels[state.Label] = path
	})

	return issues
}

func collectSingleDefaultChildIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "root state is nil")}
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if state.Type != sc.StateTypeNormal {
			return
		}

		defaultCount := 0
		xpath := []string{path}
		for i, child := range state.Children {
			if child != nil && child.IsInitial {
				defaultCount++
				xpath = append(xpath, childStatePath(path, i))
			}
		}
		if defaultCount != 1 {
			issues = append(issues, issuef(xpath,
				"state %s has %d default states, should have exactly 1", state.Label, defaultCount))
		}
	})

	return issues
}

func collectBasicHasNoChildrenIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "root state is nil")}
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if state.Type != sc.StateTypeBasic || len(state.Children) == 0 {
			return
		}
		xpath := []string{path}
		for i := range state.Children {
			xpath = append(xpath, childStatePath(path, i))
		}
		issues = append(issues, issuef(xpath, "basic state %s has children", state.Label))
	})

	return issues
}

func collectCompoundHasChildrenIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "root state is nil")}
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if (state.Type == sc.StateTypeNormal || state.Type == sc.StateTypeParallel) && len(state.Children) == 0 {
			issues = append(issues, issuef([]string{path}, "compound state %s has no children", state.Label))
		}
	})

	return issues
}

func collectDeterministicTransitionSelectionIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil {
		return nil
	}

	var issues []validationIssue
	eventTransitions := make(map[string]map[string]string)
	for i, transition := range statechart.Transitions {
		if transition == nil || transition.Event == "" {
			continue
		}

		path := transitionXPath(i)
		for _, source := range transition.From {
			if eventTransitions[transition.Event] == nil {
				eventTransitions[transition.Event] = make(map[string]string)
			}
			if firstPath, ok := eventTransitions[transition.Event][source]; ok {
				issues = append(issues, issuef([]string{firstPath, path},
					"non-deterministic transitions: multiple transitions from state '%s' on event '%s'",
					source, transition.Event))
				continue
			}
			eventTransitions[transition.Event][source] = path
		}
	}

	return issues
}

func collectNoEventBroadcastCycleIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil {
		return nil
	}

	graph := make(map[string][]eventBroadcastEdge)
	for i, transition := range statechart.Transitions {
		if transition == nil || transition.Event == "" {
			continue
		}
		for _, generated := range generatedEventsForValidation(transition) {
			graph[transition.Event] = append(graph[transition.Event], eventBroadcastEdge{
				from:           transition.Event,
				to:             generated,
				transitionPath: transitionXPath(i),
			})
		}
	}

	var issues []validationIssue
	visited := make(map[string]bool)
	stackIndex := make(map[string]int)
	pathEvents := []string{}
	pathEdges := []eventBroadcastEdge{}
	seen := make(map[string]bool)

	var dfs func(event string)
	dfs = func(event string) {
		visited[event] = true
		stackIndex[event] = len(pathEvents)
		pathEvents = append(pathEvents, event)
		defer func() {
			delete(stackIndex, event)
			pathEvents = pathEvents[:len(pathEvents)-1]
		}()

		for _, edge := range graph[event] {
			if idx, ok := stackIndex[edge.to]; ok {
				cycleEvents := append(append([]string(nil), pathEvents[idx:]...), edge.to)
				cycleEdges := append(append([]eventBroadcastEdge(nil), pathEdges[idx:]...), edge)
				paths := make([]string, 0, len(cycleEdges))
				for _, cycleEdge := range cycleEdges {
					paths = append(paths, cycleEdge.transitionPath)
				}
				key := canonicalCycleKey(cycleEvents, paths)
				if seen[key] {
					continue
				}
				seen[key] = true
				issues = append(issues, issuef(paths,
					"event broadcast cycle detected: %s", strings.Join(cycleEvents, " -> ")))
				continue
			}
			if visited[edge.to] {
				continue
			}
			pathEdges = append(pathEdges, edge)
			dfs(edge.to)
			pathEdges = pathEdges[:len(pathEdges)-1]
		}
	}

	var events []string
	for event := range graph {
		events = append(events, event)
	}
	sort.Strings(events)
	for _, event := range events {
		if !visited[event] {
			dfs(event)
		}
	}

	return issues
}

func collectStateHierarchyWellFormedIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "statechart has no root state")}
	}

	var issues []validationIssue
	seen := make(map[string]statePathRecord)
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, parentPath string) {
		if state.Label == "" {
			issues = append(issues, issuef([]string{path}, "state has empty label"))
			return
		}
		if existing, ok := seen[state.Label]; ok {
			issues = append(issues, issuef([]string{existing.path, path},
				"state appears multiple times in hierarchy (state_label=%s, existing_parent=%s, new_parent=%s)",
				state.Label, existing.parent, parentPath))
			return
		}
		seen[state.Label] = statePathRecord{path: path, parent: parentPath}
	})

	return issues
}

func collectOrthogonalStatesDisjointIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return nil
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if state.Type != sc.StateTypeParallel {
			return
		}

		regions := make([]map[string]string, len(state.Children))
		for i, child := range state.Children {
			regions[i] = make(map[string]string)
			collectDescendantPaths(child, childStatePath(path, i), regions[i])
		}
		for i := 0; i < len(regions); i++ {
			for j := i + 1; j < len(regions); j++ {
				for label, leftPath := range regions[i] {
					rightPath, ok := regions[j][label]
					if !ok {
						continue
					}
					issues = append(issues, issuef([]string{path, leftPath, rightPath},
						"orthogonal regions overlap (parent_state=%s, overlapping_state=%s, region_1=%d, region_2=%d)",
						state.Label, label, i, j))
				}
			}
		}
	})

	return issues
}

func collectTransitionSourceTargetValidityIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil {
		return nil
	}

	stateRecords := collectStateRecords(statechart.RootState)
	var issues []validationIssue
	for i, transition := range statechart.Transitions {
		if transition == nil {
			continue
		}
		path := transitionXPath(i)
		if transition.Label == "" {
			issues = append(issues, issuef([]string{path}, "transition has empty label"))
		}
		for _, source := range transition.From {
			if len(stateRecords[source]) == 0 {
				issues = append(issues, issuef([]string{path},
					"transition references invalid source state (transition=%s, invalid_source=%s)",
					transitionNameAt(transition, i), source))
			}
		}
		for _, target := range transition.To {
			if len(stateRecords[target]) == 0 {
				issues = append(issues, issuef([]string{path},
					"transition references invalid target state (transition=%s, invalid_target=%s)",
					transitionNameAt(transition, i), target))
			}
		}
	}

	return issues
}

func collectEventConsistencyIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil {
		return nil
	}

	declared := make(map[string]string)
	var issues []validationIssue
	for i, event := range statechart.Events {
		path := eventXPath(i)
		if event == nil || event.Label == "" {
			issues = append(issues, issuef([]string{path}, "event has empty label"))
			continue
		}
		if _, ok := declared[event.Label]; !ok {
			declared[event.Label] = path
		}
	}

	for i, transition := range statechart.Transitions {
		if transition == nil || transition.Event == "" {
			continue
		}
		if _, ok := declared[transition.Event]; ok {
			continue
		}
		issues = append(issues, issuef([]string{transitionXPath(i)},
			"transition references undeclared event (transition=%s, undeclared_event=%s)",
			transitionNameAt(transition, i), transition.Event))
	}

	return issues
}

func collectConfigurationConsistencyIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "statechart has no root state")}
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		switch state.Type {
		case sc.StateTypeBasic:
			if len(state.Children) > 0 {
				xpath := []string{path}
				for i := range state.Children {
					xpath = append(xpath, childStatePath(path, i))
				}
				issues = append(issues, issuef(xpath,
					"basic state cannot have children (state_label=%s, child_count=%d)",
					state.Label, len(state.Children)))
			}
		case sc.StateTypeNormal:
			if len(state.Children) == 0 {
				issues = append(issues, issuef([]string{path},
					"normal (XOR) state must have children (state_label=%s)", state.Label))
				return
			}
			initialCount := 0
			xpath := []string{path}
			for i, child := range state.Children {
				if child != nil && child.IsInitial {
					initialCount++
					xpath = append(xpath, childStatePath(path, i))
				}
			}
			if initialCount != 1 {
				issues = append(issues, issuef(xpath,
					"normal state must have exactly one initial child (state_label=%s, initial_count=%d)",
					state.Label, initialCount))
			}
		case sc.StateTypeParallel:
			if len(state.Children) < 2 {
				issues = append(issues, issuef([]string{path},
					"parallel (AND) state must have at least 2 children (state_label=%s, child_count=%d)",
					state.Label, len(state.Children)))
			}
			for i, child := range state.Children {
				if child == nil || !child.IsInitial {
					continue
				}
				issues = append(issues, issuef([]string{path, childStatePath(path, i)},
					"children of parallel state cannot be marked as initial (parent_state=%s, child_state=%s)",
					state.Label, child.Label))
			}
		}
	})

	return issues
}

func collectTransitionWellFormednessIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil {
		return nil
	}

	var issues []validationIssue
	for i, transition := range statechart.Transitions {
		if transition == nil {
			continue
		}
		path := transitionXPath(i)

		if transition.Label == "" {
			issues = append(issues, issuef([]string{path}, "transition has empty label"))
		}
		if len(transition.From) == 0 {
			issues = append(issues, issuef([]string{path},
				"transition has no source states (transition=%s)", transitionNameAt(transition, i)))
		}
		if len(transition.To) == 0 {
			issues = append(issues, issuef([]string{path},
				"transition has no target states (transition=%s)", transitionNameAt(transition, i)))
		}

		if transition.Guard != nil && transition.Guard.Condition == nil {
			if err := validateGuardExpression(transition.Guard.Expression); err != nil {
				issues = append(issues, issuef([]string{path},
					"transition %s has invalid guard: %s", transitionNameAt(transition, i), err.Error()))
			}
		}

		for j, action := range transition.Actions {
			if action != nil && action.Label != "" {
				continue
			}
			issues = append(issues, issuef([]string{transitionActionXPath(i, j)},
				"transition action has empty label (transition=%s, action_index=%d)",
				transitionNameAt(transition, i), j))
		}
	}

	return issues
}

func collectInitialStateExistsIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "statechart has no root state")}
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if state.Type != sc.StateTypeNormal || len(state.Children) == 0 {
			return
		}
		for _, child := range state.Children {
			if child != nil && child.IsInitial {
				return
			}
		}
		issues = append(issues, issuef([]string{path},
			"compound state has no initial state (state_label=%s)", state.Label))
	})

	return issues
}

func collectNoSelfContainmentIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "statechart has no root state")}
	}

	var issues []validationIssue
	var walk func(state *sc.State, path string, ancestors map[string]string)
	walk = func(state *sc.State, path string, ancestors map[string]string) {
		if state == nil {
			return
		}
		if ancestorPath, ok := ancestors[state.Label]; ok {
			issues = append(issues, issuef([]string{ancestorPath, path},
				"circular containment detected (state_label=%s)", state.Label))
			return
		}

		nextAncestors := make(map[string]string, len(ancestors)+1)
		for label, ancestorPath := range ancestors {
			nextAncestors[label] = ancestorPath
		}
		nextAncestors[state.Label] = path

		for i, child := range state.Children {
			walk(child, childStatePath(path, i), nextAncestors)
		}
	}

	walk(statechart.RootState, rootStatePath(), make(map[string]string))
	return issues
}

func collectFinalStatePropertiesIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil {
		return nil
	}

	stateRecords := collectStateRecords(statechart.RootState)
	finalStates := make(map[string][]string)
	if statechart.RootState != nil {
		walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
			if state.IsFinal {
				finalStates[state.Label] = append(finalStates[state.Label], path)
			}
		})
	}

	var issues []validationIssue
	for i, transition := range statechart.Transitions {
		if transition == nil {
			continue
		}
		transitionPath := transitionXPath(i)
		for _, source := range transition.From {
			paths := finalStates[source]
			if len(paths) == 0 {
				continue
			}
			xpath := append(append([]string(nil), paths...), transitionPath)
			if len(stateRecords[source]) == 0 {
				xpath = []string{transitionPath}
			}
			issues = append(issues, issuef(xpath, "final state %s cannot have outgoing transitions", source))
		}
	}

	return issues
}

func collectHistoryStatePropertyIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "statechart has no root state")}
	}

	var issues []validationIssue
	historyCounts := make(map[string]map[sc.HistoryType]int)

	walkStates(statechart.RootState, func(state *sc.State, path string, parent *sc.State, parentPath string) {
		if !state.IsHistory {
			return
		}

		if len(state.Children) > 0 {
			issues = append(issues, issuef([]string{path}, "history state %s cannot have children", state.Label))
		}
		if state.IsInitial {
			issues = append(issues, issuef([]string{path}, "history state %s cannot be an initial state", state.Label))
		}
		if state.IsFinal {
			issues = append(issues, issuef([]string{path}, "history state %s cannot be a final state", state.Label))
		}
		if parent == nil {
			issues = append(issues, issuef([]string{path}, "history state %s must have a composite parent", state.Label))
			return
		}
		if parent.Type != sc.StateTypeNormal {
			issues = append(issues, issuef([]string{parentPath, path},
				"history state %s must be contained in an OR state, found parent %s of type %v",
				state.Label, parent.Label, parent.Type))
		}

		historyType := state.HistoryType
		if historyType == sc.HistoryType_HISTORY_TYPE_UNSPECIFIED {
			historyType = sc.HistoryType_HISTORY_TYPE_SHALLOW
		}
		if historyCounts[parentPath] == nil {
			historyCounts[parentPath] = make(map[sc.HistoryType]int)
		}
		historyCounts[parentPath][historyType]++
		if historyCounts[parentPath][historyType] > 1 {
			issues = append(issues, issuef([]string{parentPath, path},
				"state %s has multiple %s history pseudostates", parent.Label, historyTypeName(historyType)))
		}

		nonHistoryChildren := 0
		initialNonHistoryChildren := 0
		nestedNonHistoryChildren := 0
		xpath := []string{parentPath, path}
		for i, sibling := range parent.Children {
			if sibling == nil || sibling.IsHistory {
				continue
			}
			siblingPath := childStatePath(parentPath, i)
			xpath = append(xpath, siblingPath)
			nonHistoryChildren++
			if sibling.IsInitial {
				initialNonHistoryChildren++
			}
			if len(sibling.Children) > 0 {
				nestedNonHistoryChildren++
			}
		}

		if nonHistoryChildren == 0 {
			issues = append(issues, issuef([]string{parentPath, path},
				"history state %s has no non-history siblings under parent %s", state.Label, parent.Label))
		}
		if initialNonHistoryChildren != 1 {
			issues = append(issues, issuef(xpath,
				"history state %s requires parent %s to have exactly one non-history initial child, found %d",
				state.Label, parent.Label, initialNonHistoryChildren))
		}
		if historyType == sc.HistoryType_HISTORY_TYPE_DEEP && nestedNonHistoryChildren == 0 {
			issues = append(issues, issuef(xpath,
				"deep history state %s requires nested substates under parent %s", state.Label, parent.Label))
		}
	})

	return issues
}

func collectParallelStateSemanticsIssues(statechart *sc.Statechart) []validationIssue {
	if statechart == nil || statechart.RootState == nil {
		return []validationIssue{issuef([]string{rootStatePath()}, "statechart has no root state")}
	}

	var issues []validationIssue
	walkStates(statechart.RootState, func(state *sc.State, path string, _ *sc.State, _ string) {
		if state.Type != sc.StateTypeParallel {
			return
		}
		if len(state.Children) < 2 {
			issues = append(issues, issuef([]string{path},
				"parallel state %s must have at least 2 orthogonal regions, found %d",
				state.Label, len(state.Children)))
		}
	})

	return issues
}
