# Statechart Graph Analysis Tooling

**Status:** Draft
**Author:** sc maintainers
**Created:** 2024-12-25

## Problem Statement

Statecharts encode complex behavioral models with hierarchical states, parallel regions, and conditional transitions. Developers need tools to answer questions like:

- Which states are reachable from the initial configuration?
- What path does the machine take from state A to state B?
- Are there deadlock states with no outgoing transitions?
- Which transitions are never enabled (dead code)?
- What is the scope of a transition (LCA of source/target)?

Currently, these analyses require writing custom code. We need a `digraph`-inspired CLI tool that provides standard graph operations adapted for statechart semantics.

## Goals

- Provide `digraph`-style operations for statechart analysis
- Support both hierarchy graph and transition graph views
- Handle parallel regions and orthogonal states correctly
- Enable pipeline composition with UNIX tools
- Output formats: text (default), JSON, DOT/Mermaid

## Non-Goals

- Runtime simulation (use `sc step` instead)
- Visual editing (use web-visualizer)
- Model checking / temporal logic (future work)

## Reference: golang.org/x/tools/cmd/digraph

The `digraph` tool operates on directed graphs via stdin:

| Command | Description |
|---------|-------------|
| `nodes` | List all nodes |
| `degree` | In-degree and out-degree of each node |
| `succs <node>` | Immediate successors |
| `preds <node>` | Immediate predecessors |
| `forward <node>` | Transitive closure (reachable from) |
| `reverse <node>` | Transitive closure (reaches to) |
| `somepath <a> <b>` | Some path from a to b |
| `allpaths <a> <b>` | All edges on paths from a to b |
| `sccs` | Strongly connected components |
| `focus <node>` | Subgraph through node |

## Proposed Solution

### Dual Graph Model

Statecharts have two distinct graph structures:

1. **Hierarchy Graph (H):** Parent → Child relationships (tree/forest)
2. **Transition Graph (T):** State → State via events (directed multigraph)

Operations specify which graph via `--graph={hierarchy,transition,both}`.

### Command Structure

```
sc graph [command] [flags] [args...]

Flags:
  --input, -i     Input statechart file (JSON/proto)
  --graph, -g     Graph type: hierarchy, transition, both (default: transition)
  --format, -f    Output format: text, json, dot, mermaid (default: text)
  --events        Filter transitions by event(s)
  --guards        Include guard conditions in output
  --with-actions  Include actions in output
```

## Operations

### Basic Queries

#### `states` - List all states

```bash
# All states (flat list)
$ sc graph states -i chart.json
Off
On
On.Idle
On.Running

# With hierarchy depth
$ sc graph states --depth -i chart.json
0 __root__
1 Off
1 On
2 On.Idle
2 On.Running

# Only leaf states (BASIC type)
$ sc graph states --leaves -i chart.json
Off
On.Idle
On.Running

# Only composite states (OR/AND type)
$ sc graph states --composite -i chart.json
On

# Only parallel states
$ sc graph states --parallel -i chart.json
Dashboard
```

#### `transitions` - List all transitions

```bash
# All transitions
$ sc graph transitions -i chart.json
Off -> On [TURN_ON]
On.Idle -> On.Running [START]
On.Running -> On.Idle [STOP]
On -> Off [TURN_OFF]

# Filter by event
$ sc graph transitions --event=START -i chart.json
On.Idle -> On.Running [START]

# With guards
$ sc graph transitions --guards -i chart.json
On.Idle -> On.Running [START] {fuel > 0}
```

#### `events` - List all events

```bash
$ sc graph events -i chart.json
START
STOP
TURN_OFF
TURN_ON
```

#### `degree` - State connectivity

```bash
# Transition graph degree
$ sc graph degree -i chart.json
Off         in=1 out=1
On          in=1 out=1
On.Idle     in=1 out=1
On.Running  in=1 out=1

# Hierarchy graph degree (children count)
$ sc graph degree --graph=hierarchy -i chart.json
__root__  children=2
Off       children=0
On        children=2
On.Idle   children=0
On.Running children=0
```

### Hierarchy Operations

#### `parent` - Get parent state

```bash
$ sc graph parent On.Idle -i chart.json
On

$ sc graph parent On -i chart.json
__root__
```

#### `children` - Get child states

```bash
# Immediate children
$ sc graph children On -i chart.json
On.Idle
On.Running

# All descendants (transitive)
$ sc graph children --transitive On -i chart.json
On.Idle
On.Running

# Including self (reflexive-transitive)
$ sc graph children --reflexive On -i chart.json
On
On.Idle
On.Running
```

#### `ancestors` - Get ancestor states

```bash
$ sc graph ancestors On.Idle -i chart.json
On
__root__

# Including self
$ sc graph ancestors --reflexive On.Idle -i chart.json
On.Idle
On
__root__
```

#### `lca` - Least common ancestor

```bash
$ sc graph lca On.Idle On.Running -i chart.json
On

$ sc graph lca Off On.Idle -i chart.json
__root__
```

#### `depth` - Hierarchy depth

```bash
$ sc graph depth On.Running -i chart.json
2

$ sc graph depth --all -i chart.json
__root__    0
Off         1
On          1
On.Idle     2
On.Running  2
```

### Transition Graph Operations

#### `succs` / `preds` - Immediate successors/predecessors

```bash
# States reachable in one transition
$ sc graph succs On.Idle -i chart.json
On.Running

# States that can reach in one transition
$ sc graph preds On.Running -i chart.json
On.Idle

# With events
$ sc graph succs --with-events On.Idle -i chart.json
On.Running [START]
```

#### `forward` / `reverse` - Transitive closure

```bash
# All states reachable from Off
$ sc graph forward Off -i chart.json
On
On.Idle
On.Running

# All states that can reach Off
$ sc graph reverse Off -i chart.json
On
On.Idle
On.Running
```

#### `somepath` - Find a path

```bash
$ sc graph somepath Off On.Running -i chart.json
Off -> On [TURN_ON]
On -> On.Idle [default]
On.Idle -> On.Running [START]

# As state sequence
$ sc graph somepath --states Off On.Running -i chart.json
Off
On
On.Idle
On.Running
```

#### `allpaths` - All paths between states

```bash
$ sc graph allpaths Off On.Running -i chart.json
# Returns all transitions on any path

# Limit path length
$ sc graph allpaths --max-length=5 Off On.Running -i chart.json
```

#### `reachable` - Reachability from initial configuration

```bash
# All reachable states from initial config
$ sc graph reachable -i chart.json
Off
On
On.Idle
On.Running

# Unreachable states (dead states)
$ sc graph reachable --unreachable -i chart.json
DeadState
```

### Structural Analysis

#### `sccs` - Strongly connected components

```bash
$ sc graph sccs -i chart.json
[Off On On.Idle On.Running]

# Multiple SCCs (one per line)
$ sc graph sccs -i complex.json
[Idle Working]
[Error]
[Shutdown]
```

#### `cycles` - Find cycles

```bash
$ sc graph cycles -i chart.json
On.Idle -> On.Running -> On.Idle

# Just detect presence
$ sc graph cycles --exists -i chart.json
true
```

#### `deadlocks` - States with no outgoing transitions

```bash
$ sc graph deadlocks -i chart.json
FinalState
ErrorState
```

#### `sources` / `sinks` - Entry/exit points

```bash
# States with no incoming transitions
$ sc graph sources -i chart.json
Off

# States with no outgoing transitions
$ sc graph sinks -i chart.json
FinalState
```

### Parallel Region Operations

#### `regions` - List orthogonal regions

```bash
$ sc graph regions -i chart.json
Dashboard.Temperature
Dashboard.Pressure
Dashboard.Status

# For a specific parallel state
$ sc graph regions Dashboard -i chart.json
Dashboard.Temperature
Dashboard.Pressure
Dashboard.Status
```

#### `orthogonal` - Check orthogonality

```bash
$ sc graph orthogonal Temperature.High Pressure.Low -i chart.json
true

$ sc graph orthogonal On.Idle On.Running -i chart.json
false
```

#### `consistent` - Check configuration consistency

```bash
$ sc graph consistent On On.Idle -i chart.json
true

$ sc graph consistent On.Idle On.Running -i chart.json
false  # XOR violation
```

### Subgraph Extraction

#### `focus` - Subgraph through a state

```bash
# All states/transitions passing through On
$ sc graph focus On -i chart.json --format=dot | dot -Tpng -o focus.png
```

#### `scope` - Transition scope (LCA-based)

```bash
# What composite state "owns" this transition?
$ sc graph scope "On.Idle -> On.Running" -i chart.json
On

# Transitions that exit/enter a state
$ sc graph scope --enters On -i chart.json
Off -> On [TURN_ON]

$ sc graph scope --exits On -i chart.json
On -> Off [TURN_OFF]
```

### History State Operations

#### `history` - List history states

```bash
$ sc graph history -i chart.json
On.H (shallow)
Settings.H* (deep)
```

#### `history-scope` - What a history state remembers

```bash
$ sc graph history-scope On.H -i chart.json
On.Idle
On.Running
# (shallow: only immediate children)

$ sc graph history-scope Settings.H* -i chart.json
Settings.Display
Settings.Display.Brightness
Settings.Display.Contrast
Settings.Audio
Settings.Audio.Volume
# (deep: full subtree)
```

### Output Formats

#### Text (default)
```
Off -> On [TURN_ON]
On.Idle -> On.Running [START]
```

#### JSON
```json
{
  "transitions": [
    {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
    {"from": ["On.Idle"], "to": ["On.Running"], "event": "START"}
  ]
}
```

#### DOT (Graphviz)
```bash
$ sc graph transitions -i chart.json --format=dot
digraph {
  "Off" -> "On" [label="TURN_ON"];
  "On.Idle" -> "On.Running" [label="START"];
}
```

#### Mermaid
```bash
$ sc graph transitions -i chart.json --format=mermaid
stateDiagram-v2
  Off --> On : TURN_ON
  On.Idle --> On.Running : START
```

### Pipeline Integration

Designed for UNIX pipeline composition:

```bash
# Find all states reachable from Off, then get their children
$ sc graph forward Off -i chart.json | xargs -I{} sc graph children {} -i chart.json

# Find transitions from all leaf states
$ sc graph states --leaves -i chart.json | xargs -I{} sc graph succs {} -i chart.json

# Visualize reachable subgraph
$ sc graph forward Off -i chart.json | sc graph focus --stdin -i chart.json --format=dot | dot -Tpng -o reachable.png

# Find paths avoiding certain states
$ sc graph allpaths Off On.Running -i chart.json | grep -v "Error"
```

## Guard-Aware Analysis

Guards affect reachability. Optional `--eval-guards` mode:

```bash
# Without guard evaluation (structural only)
$ sc graph forward Off -i chart.json
On
On.Idle
On.Running

# With guard evaluation (requires context)
$ sc graph forward Off -i chart.json --eval-guards --context='{"fuel": 0}'
On
On.Idle
# On.Running not reachable because guard "fuel > 0" is false
```

**Note:** Guard evaluation requires expression support (see expressions.md design doc).

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Define `sc graph` subcommand structure
- [ ] Implement statechart loading from JSON/proto
- [ ] Build internal graph representations (hierarchy + transition)
- [ ] Implement output formatters (text, JSON, DOT, Mermaid)

### Phase 2: Basic Operations (Week 2)
- [ ] `states`, `transitions`, `events` listing
- [ ] `parent`, `children`, `ancestors` hierarchy ops
- [ ] `succs`, `preds` immediate neighbor ops
- [ ] `degree` connectivity metrics

### Phase 3: Traversal Operations (Week 3)
- [ ] `forward`, `reverse` transitive closure
- [ ] `somepath`, `allpaths` path finding
- [ ] `reachable` from initial configuration
- [ ] `lca` least common ancestor

### Phase 4: Structural Analysis (Week 4)
- [ ] `sccs` strongly connected components (Tarjan's algorithm)
- [ ] `cycles` detection
- [ ] `deadlocks`, `sources`, `sinks`
- [ ] `focus` subgraph extraction

### Phase 5: Parallel & History (Week 5)
- [ ] `regions`, `orthogonal`, `consistent` parallel ops
- [ ] `history`, `history-scope` history state ops
- [ ] `scope` transition scope analysis

### Phase 6: Advanced Features (Week 6)
- [ ] `--eval-guards` guard-aware analysis
- [ ] Pipeline mode (`--stdin` for state lists)
- [ ] Performance optimization for large statecharts

## Data Structures

### Internal Graph Representation

```go
// HierarchyGraph represents parent-child relationships
type HierarchyGraph struct {
    States   map[string]*StateNode
    Root     string
}

type StateNode struct {
    Label    string
    Type     StateType  // BASIC, OR, AND
    Parent   string
    Children []string
    Depth    int
}

// TransitionGraph represents state-to-state transitions
type TransitionGraph struct {
    States      map[string]bool
    Transitions []*TransitionEdge
    Outgoing    map[string][]*TransitionEdge  // adjacency list
    Incoming    map[string][]*TransitionEdge  // reverse adjacency
}

type TransitionEdge struct {
    From   []string
    To     []string
    Event  string
    Guard  string
    Label  string
}
```

### Algorithm Complexity

| Operation | Algorithm | Complexity |
|-----------|-----------|------------|
| `states` | Tree traversal | O(n) |
| `children --transitive` | DFS | O(n) |
| `succs`/`preds` | Adjacency lookup | O(degree) |
| `forward`/`reverse` | BFS/DFS | O(n + e) |
| `somepath` | BFS | O(n + e) |
| `allpaths` | DFS with memoization | O(n * e) |
| `sccs` | Tarjan's algorithm | O(n + e) |
| `lca` | Ancestor intersection | O(depth) |
| `reachable` | BFS from initial | O(n + e) |

## Security Considerations

- Input validation: Reject malformed statecharts before analysis
- Cycle detection: Prevent infinite loops in path enumeration
- Memory limits: Cap `allpaths` results for dense graphs
- Guard evaluation: Sandboxed (see expressions.md security model)

## Open Questions

1. **Should `forward`/`reverse` cross hierarchy boundaries?**
   - Option A: Only consider explicit transitions
   - Option B: Include implicit default entries
   - Recommendation: Default to A, flag `--include-defaults` for B

2. **How to handle compound transitions (multiple sources/targets)?**
   - Option A: Expand to individual edges
   - Option B: Treat as hyperedge
   - Recommendation: A for most ops, preserve structure in JSON output

3. **Should parallel regions be separate subgraphs?**
   - For `sccs`: Yes, analyze regions independently
   - For `reachable`: No, consider cross-region transitions

4. **Event wildcards in queries?**
   - `--event="BUTTON_*"` for glob matching
   - Useful for large event alphabets

## CLI Tool Structure

All graph operations are available as standalone CLI commands:

### Primary Entry Points

```bash
# Unified command
sc graph <operation> [args] [flags]

# Or standalone tools (symlinks to sc)
sc-states [flags]
sc-transitions [flags]
sc-reachable [flags]
sc-paths [flags]
sc-analyze [flags]
```

### Command Registration

```go
// cmd/sc/graph.go
var graphCmd = &cobra.Command{
    Use:   "graph",
    Short: "Statechart graph analysis operations",
}

func init() {
    rootCmd.AddCommand(graphCmd)

    // Basic queries
    graphCmd.AddCommand(statesCmd)
    graphCmd.AddCommand(transitionsCmd)
    graphCmd.AddCommand(eventsCmd)
    graphCmd.AddCommand(degreeCmd)

    // Hierarchy operations
    graphCmd.AddCommand(parentCmd)
    graphCmd.AddCommand(childrenCmd)
    graphCmd.AddCommand(ancestorsCmd)
    graphCmd.AddCommand(lcaCmd)
    graphCmd.AddCommand(depthCmd)

    // Transition graph operations
    graphCmd.AddCommand(succsCmd)
    graphCmd.AddCommand(predsCmd)
    graphCmd.AddCommand(forwardCmd)
    graphCmd.AddCommand(reverseCmd)
    graphCmd.AddCommand(somepathCmd)
    graphCmd.AddCommand(allpathsCmd)
    graphCmd.AddCommand(reachableCmd)

    // Structural analysis
    graphCmd.AddCommand(sccsCmd)
    graphCmd.AddCommand(cyclesCmd)
    graphCmd.AddCommand(deadlocksCmd)
    graphCmd.AddCommand(sourcesCmd)
    graphCmd.AddCommand(sinksCmd)
    graphCmd.AddCommand(focusCmd)
    graphCmd.AddCommand(scopeCmd)

    // Parallel region operations
    graphCmd.AddCommand(regionsCmd)
    graphCmd.AddCommand(orthogonalCmd)
    graphCmd.AddCommand(consistentCmd)

    // History operations
    graphCmd.AddCommand(historyCmd)
    graphCmd.AddCommand(historyScopeCmd)

    // Intelligent analysis
    graphCmd.AddCommand(analyzeCmd)
    graphCmd.AddCommand(simulateCmd)
    graphCmd.AddCommand(coverageCmd)
}
```

### Global Flags

```go
var (
    inputFile   string  // -i, --input
    graphType   string  // -g, --graph (hierarchy|transition|both)
    outputFmt   string  // -f, --format (text|json|dot|mermaid)
    contextFile string  // -c, --context (for guard evaluation)
    verbose     bool    // -v, --verbose
    quiet       bool    // -q, --quiet
)
```

## Intelligent Analysis

### History-Aware Reachability

History states create implicit edges that affect reachability:

```bash
# Standard reachability (ignores history)
$ sc graph reachable -i chart.json
Off
On
On.Idle
On.Running

# History-aware reachability
$ sc graph reachable --history-aware -i chart.json
Off
On
On.Idle
On.Running
On.H          # History pseudostate
On.LastActive # States remembered by history
```

#### History Edge Semantics

```go
// HistoryEdge represents implicit transition via history
type HistoryEdge struct {
    HistoryState string      // e.g., "On.H"
    Type         HistoryType // SHALLOW or DEEP
    Remembers    []string    // States that can be restored
    DefaultTarget string     // If no history saved
}

// ExpandHistoryEdges adds implicit edges for analysis
func ExpandHistoryEdges(g *TransitionGraph, sc *Statechart) {
    for _, h := range sc.HistoryStates() {
        if h.Type == SHALLOW {
            // Edges to all immediate children of parent
            for _, child := range sc.Children(h.Parent) {
                g.AddEdge(h.Label, child, "history-restore")
            }
        } else { // DEEP
            // Edges to all leaf descendants
            for _, leaf := range sc.Leaves(h.Parent) {
                g.AddEdge(h.Label, leaf, "history-restore")
            }
        }
    }
}
```

#### History Path Analysis

```bash
# Find paths considering history restoration
$ sc graph somepath Off On.Running --history-aware -i chart.json
Off -> On [TURN_ON]
On -> On.H [default-to-history]
On.H -> On.Running [history-restore]  # If previously in Running

# All possible history restorations
$ sc graph history-paths On.H -i chart.json
On.H -> On.Idle [history-restore, probability=0.5]
On.H -> On.Running [history-restore, probability=0.5]
On.H -> On.Idle [default, if-no-history]
```

### Temporal Analysis (Delays & Timeouts)

Statecharts with `after(duration)` events require temporal reasoning:

```bash
# List all timed transitions
$ sc graph transitions --timed -i chart.json
On.Idle -> On.Timeout [after(5000ms)]
On.Running -> On.Idle [after(30s)]
Connecting -> Error [after(10s)]

# Temporal reachability (what's reachable within time budget)
$ sc graph reachable --time-budget=10s -i chart.json
Off
On
On.Idle
On.Timeout  # Reachable after 5s idle
# On.Running -> On.Idle not shown (requires 30s)

# Critical path timing
$ sc graph somepath Off Error --timing -i chart.json
Off -> Connecting [CONNECT, t=0]
Connecting -> Error [after(10s), t=10s]
Total: 10s minimum
```

#### Temporal Edge Types

```protobuf
message TemporalTransition {
  Transition base = 1;

  oneof timing {
    Duration after = 2;      // after(duration)
    Duration delay = 3;      // delay before action
    TimeRange window = 4;    // valid time window
  }

  bool cancellable = 5;      // Can be preempted?
}

message TimeRange {
  Duration min = 1;
  Duration max = 2;
}
```

#### Timing Analysis Operations

```bash
# Minimum time to reach state
$ sc graph min-time Off On.Running -i chart.json
0ms  # Instantaneous if events available

# Maximum time (considering timeouts)
$ sc graph max-time Off Error -i chart.json
10s  # Timeout path

# Time-bounded model checking
$ sc graph reachable --within=1s -i chart.json
Off
On
On.Idle
# States requiring >1s excluded

# Timeout hazards (states that auto-transition)
$ sc graph timeouts -i chart.json
On.Idle: -> On.Timeout after 5000ms
Connecting: -> Error after 10s
Session: -> Expired after 1h
```

### Comprehensive Analysis Command

```bash
$ sc graph analyze -i chart.json

=== Statechart Analysis Report ===

Structure:
  States: 12 (8 basic, 3 OR-composite, 1 AND-composite)
  Transitions: 18
  Events: 7
  Max depth: 4

Hierarchy:
  Parallel regions: 2 (Dashboard.Left, Dashboard.Right)
  History states: 1 shallow, 1 deep

Reachability:
  Reachable from initial: 11/12 (91.7%)
  Unreachable states: [DeadState]
  Dead transitions: 1 (DeadState -> X, unreachable source)

Connectivity:
  Strongly connected components: 3
  Largest SCC: [Idle, Working, Paused] (3 states)
  Terminal states: [Shutdown, Error]
  Source states: [Initial]

Temporal:
  Timed transitions: 4
  Shortest timeout: 1s (Idle -> Timeout)
  Longest timeout: 1h (Session -> Expired)
  Timeout chains: Idle -> Timeout -> Error (total: 1s + 5s = 6s)

Potential Issues:
  [WARN] Unreachable state: DeadState
  [WARN] Non-determinism: State 'Menu' has 2 transitions on 'SELECT'
  [INFO] Deep history in 'Settings' remembers 5 substates

Recommendations:
  - Remove or connect DeadState
  - Add guards to disambiguate Menu/SELECT transitions
```

### Simulation with History Tracking

```bash
# Simulate with history state tracking
$ sc graph simulate -i chart.json --trace

Initial: [Off]
> TURN_ON
  Exit: Off
  Enter: On, On.Idle
  Config: [On, On.Idle]
  History: {}

> START
  Exit: On.Idle
  Enter: On.Running
  Config: [On, On.Running]
  History: {On.H: [On.Running]}  # Shallow history saved

> TURN_OFF
  Exit: On.Running, On
  Enter: Off
  Config: [Off]
  History: {On.H: [On.Running]}  # History preserved

> TURN_ON
  Exit: Off
  Enter: On
  Enter: On.Running (via On.H history restore)
  Config: [On, On.Running]
  History: {On.H: [On.Running]}
```

### Coverage Analysis

```bash
# State coverage from trace
$ sc graph coverage -i chart.json --trace=execution.log
State Coverage: 10/12 (83.3%)
  Covered: Off, On, On.Idle, On.Running, ...
  Uncovered: DeadState, RareError

Transition Coverage: 15/18 (83.3%)
  Covered: Off->On, On.Idle->On.Running, ...
  Uncovered: RareError->Recovery, ...

Event Coverage: 6/7 (85.7%)
  Covered: TURN_ON, TURN_OFF, START, STOP, ...
  Uncovered: EMERGENCY

# Generate test cases to improve coverage
$ sc graph coverage --generate-tests -i chart.json
Test 1: [TURN_ON, ERROR] -> covers RareError
Test 2: [TURN_ON, ERROR, RECOVER] -> covers RareError->Recovery
Test 3: [EMERGENCY] -> covers EMERGENCY event
```

### Path Analysis with Constraints

```bash
# Paths avoiding certain states
$ sc graph allpaths Off Shutdown --avoid=Error -i chart.json

# Paths that must pass through a state
$ sc graph allpaths Off Shutdown --through=Confirmation -i chart.json

# Paths with specific event sequence
$ sc graph allpaths Off Shutdown --events="INIT,PROCESS,COMPLETE" -i chart.json

# Shortest path by transition count
$ sc graph somepath Off Shutdown --shortest -i chart.json

# Fastest path by timing
$ sc graph somepath Off Shutdown --fastest -i chart.json
```

### Guard Satisfaction Analysis

```bash
# Which guards block reachability?
$ sc graph reachable --explain-guards -i chart.json --context='{"level": 1}'
On.Idle -> On.Advanced: BLOCKED by guard "level >= 5"
On.Running -> On.Turbo: BLOCKED by guard "fuel > 100 && speed > 50"

# Find contexts that enable a path
$ sc graph somepath Off On.Turbo --solve-guards -i chart.json
Required context:
  level >= 5 (for Idle -> Advanced)
  fuel > 100 (for Running -> Turbo)
  speed > 50 (for Running -> Turbo)
Minimal solution: {"level": 5, "fuel": 101, "speed": 51}
```

## Comparison & Equivalence

### `diff` - Compare statechart versions

```bash
$ sc graph diff chart-v1.json chart-v2.json

=== Statechart Diff ===

States:
  + NewFeature           (added)
  + NewFeature.Active    (added)
  - DeprecatedState      (removed)
  ~ Settings             (modified: added child)

Transitions:
  + NewFeature.Active -> Done [COMPLETE]     (added)
  - OldPath -> Deprecated [LEGACY]           (removed)
  ~ Idle -> Working: guard changed           (modified)
    was: "level > 0"
    now: "level > 0 && enabled"

Events:
  + COMPLETE  (added)
  - LEGACY    (removed, now unused)

Summary: +2 states, -1 state, +1 transition, -1 transition, 3 guards modified
```

```bash
# Structural diff only (ignore labels/metadata)
$ sc graph diff --structural chart-v1.json chart-v2.json

# Output as JSON for tooling
$ sc graph diff --format=json chart-v1.json chart-v2.json
```

### `equiv` - Bisimulation equivalence checking

```bash
# Are these charts behaviorally equivalent?
$ sc graph equiv chart1.json chart2.json
EQUIVALENT: Charts are trace-equivalent

$ sc graph equiv chart1.json chart2.json
NOT EQUIVALENT:
  Distinguishing trace: [INIT, START, PAUSE]
  chart1: reaches [Paused]
  chart2: reaches [Stopped]
```

```bash
# Weak bisimulation (ignores internal transitions)
$ sc graph equiv --weak chart1.json chart2.json

# Strong bisimulation (exact structural match)
$ sc graph equiv --strong chart1.json chart2.json

# Check if chart2 simulates chart1 (refinement)
$ sc graph equiv --refines chart1.json chart2.json
```

### `--watch` - Live analysis mode

```bash
# Re-run analysis on file changes
$ sc graph analyze -i chart.json --watch
Watching chart.json for changes...

[12:34:56] File changed, re-analyzing...
=== Statechart Analysis Report ===
...

[12:35:10] File changed, re-analyzing...
=== Statechart Analysis Report ===
...
^C
```

```bash
# Watch with specific command
$ sc graph states --leaves -i chart.json --watch

# Watch with webhook notification
$ sc graph analyze -i chart.json --watch --webhook=http://localhost:8080/notify
```

## Enhanced Analysis Checks

The `sc graph analyze` command includes these additional checks:

### Determinism Analysis

```bash
$ sc graph analyze --check=determinism -i chart.json

Determinism Check:
  [WARN] State 'Menu' has 2 transitions on event 'SELECT':
    -> MenuItem1 [guard: selected == 1]
    -> MenuItem2 [guard: selected == 2]
    Status: GUARDED (guards are mutually exclusive)

  [ERROR] State 'Input' has 2 transitions on event 'KEY':
    -> Processing [no guard]
    -> Error [no guard]
    Status: NON-DETERMINISTIC (ambiguous, first wins by priority)

  [OK] State 'Idle': all events have unique handlers
```

### Completeness Analysis

```bash
$ sc graph analyze --check=completeness -i chart.json

Completeness Check:
  Event 'EMERGENCY' not handled in states:
    - Processing
    - Uploading
    - Downloading

  Event 'CANCEL' not handled in states:
    - Idle (intentional? no cancel from idle)

  Recommendation: Add explicit ignore transitions or handle events
```

### Priority Conflict Detection

```bash
$ sc graph analyze --check=priorities -i chart.json

Priority Conflicts:
  [WARN] Transitions from 'Active' have same priority (0):
    Active -> Paused [PAUSE]
    Active -> Stopped [STOP]
    (OK if events differ, conflict if same event)

  [ERROR] Overlapping transitions from 'Menu.Item':
    Menu.Item -> Menu.SubItem [SELECT, priority=1]
    Menu -> Home [SELECT, priority=1]
    Conflict: parent and child both handle SELECT at same priority
    Resolution: Child wins (inner-to-outer precedence)
```

### All Checks Combined

```bash
$ sc graph analyze -i chart.json

=== Statechart Analysis Report ===

Structure:
  States: 12 (8 basic, 3 OR-composite, 1 AND-composite)
  Transitions: 18
  Events: 7
  Max depth: 4

Hierarchy:
  Parallel regions: 2 (Dashboard.Left, Dashboard.Right)
  History states: 1 shallow, 1 deep

Reachability:
  Reachable from initial: 11/12 (91.7%)
  Unreachable states: [DeadState]
  Dead transitions: 1 (DeadState -> X, unreachable source)

Connectivity:
  Strongly connected components: 3
  Largest SCC: [Idle, Working, Paused] (3 states)
  Terminal states: [Shutdown, Error]
  Source states: [Initial]

Temporal:
  Timed transitions: 4
  Shortest timeout: 1s (Idle -> Timeout)
  Longest timeout: 1h (Session -> Expired)
  Timeout chains: Idle -> Timeout -> Error (total: 1s + 5s = 6s)

Determinism:
  Non-deterministic states: 1 [Input]
  Guarded ambiguity: 2 states (resolved by guards)

Completeness:
  Unhandled event/state pairs: 5
  Events with incomplete coverage: [EMERGENCY, CANCEL]

Priority:
  Conflicting priorities: 0
  Inner-to-outer resolutions: 3

Potential Issues:
  [ERROR] Non-determinism in 'Input' on event 'KEY'
  [WARN] Unreachable state: DeadState
  [WARN] Event 'EMERGENCY' not handled in 3 states
  [INFO] Deep history in 'Settings' remembers 5 substates

Recommendations:
  - Add guards to disambiguate Input/KEY transitions
  - Remove or connect DeadState
  - Add EMERGENCY handlers or document intentional omission
```

## MCP Server Integration

Expose graph operations via MCP for IDE/editor integration:

```typescript
// Available MCP tools
sc_graph_states      // List states with filters
sc_graph_transitions // List transitions
sc_graph_reachable   // Reachability analysis
sc_graph_somepath    // Path finding
sc_graph_analyze     // Comprehensive analysis
sc_graph_simulate    // Step-by-step simulation
sc_graph_coverage    // Coverage analysis
```

```bash
# MCP server provides these as tools
$ sc mcp serve

# Example MCP tool call
{
  "tool": "sc_graph_reachable",
  "arguments": {
    "chart": "{...}",
    "history_aware": true,
    "time_budget": "10s"
  }
}
```

## References

- [golang.org/x/tools/cmd/digraph](https://pkg.go.dev/golang.org/x/tools/cmd/digraph)
- [Tarjan's SCC Algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
- [Harel Statecharts](https://www.sciencedirect.com/science/article/pii/0167642387900359)
- [UML State Machine Semantics](https://www.omg.org/spec/UML/)
- [Timed Automata](https://www.it.uu.se/research/group/darts/uppaal/about.shtml) (UPPAAL)
- [History State Semantics](https://statecharts.dev/glossary/history-state.html)
