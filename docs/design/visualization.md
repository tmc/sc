# Visualization Extensions for Statecharts Proto

## Status: PROPOSED
## Author: Claude Code / tmc
## Date: 2025-12-25

---

## 1. Abstract

This document specifies **visualization extensions** for the sc statecharts
protocol buffer schema. While `layout.proto` handles static positioning
(where elements are placed), visualization extensions handle dynamic display
(how elements are rendered at runtime).

**Key capabilities:**
- **Semantic zoom**: Progressive disclosure based on zoom level
- **Machine views**: Named camera positions with filters
- **Trace visualization**: Rendering execution history on the diagram

**Motivating example**: The Zelda3 statechart has 786 states. At full zoom-out,
showing all labels creates visual noise. Semantic zoom collapses regions into
summaries ("Dungeon Logic: 127 states") until the user zooms in.

### Proto Split (coordination with B300)

```
+------------------------------------------------------------------+
|                    EXTENSION ARCHITECTURE                         |
+------------------------------------------------------------------+
|                                                                   |
|  layout.proto (B300)              visualization.proto (this doc)  |
|  ==================              ==============================   |
|  STATIC POSITIONING              DYNAMIC/RUNTIME DISPLAY          |
|                                                                   |
|  - State positions (x, y)        - Semantic zoom thresholds       |
|  - State sizes (w, h)            - Machine views (camera+filter)  |
|  - Transition waypoints          - Trace rendering modes          |
|  - Anchor points                 - Playback/animation config      |
|  - Z-ordering                    - Metrics overlays               |
|  - Visual style hints            - Alerting/threshold indicators  |
|  - Canvas settings               - Real-time streaming config     |
|                                                                   |
|  "WHERE things are"              "HOW things appear at runtime"   |
|                                                                   |
+------------------------------------------------------------------+
```

**Key distinction:**
- `layout.proto` answers: "At what coordinates should State X be drawn?"
- `visualization.proto` answers: "Should State X be visible at this zoom level?"

Both live in `proto/extensions/v1/` and are consumed by renderers together.

---

## 2. Scope

### In Scope
- Zoom-aware state display (collapse/expand thresholds)
- Saved view configurations (camera + filters)
- Execution trace rendering (breadcrumbs, heatmaps, timelines)
- Playback/animation controls for traces
- Real-time streaming (WebSocket/SSE, rate limiting, buffering)
- Accessibility (color-blind palettes, screen readers, keyboard nav)
- Alerting overlays (SLA breach, anomaly detection, thresholds)

### Out of Scope (handled by other protos)
- Static layout (`extensions/v1/layout.proto`): positions, sizes, anchors
- Trace data (`statecharts/v1/execution.proto`): log entries, checkpoints
- XState metadata (`xstate/v1/xstate.proto`): editor-specific properties

---

## 3. Semantic Zoom

### 3.1 Problem Statement

Large statecharts become unreadable when fully expanded. Users need:
1. Overview at low zoom (forest view)
2. Detail at high zoom (tree view)
3. Smooth transition between levels

### 3.2 StateDisplayConfig

```protobuf
// StateDisplayConfig controls zoom-aware rendering for a state.
// Stored in State.extensions.
message StateDisplayConfig {
  // Zoom threshold below which state collapses to summary.
  // Value: 0.0-1.0 where 1.0 = 100% zoom.
  // Example: 0.5 means collapse when zoomed out to 50% or less.
  double collapse_below_zoom = 1;

  // Zoom threshold above which children are shown.
  // Example: 0.8 means show children at 80%+ zoom.
  double expand_above_zoom = 2;

  // Summary label shown when collapsed.
  // Supports template variables: {state_count}, {transition_count}, {label}.
  // Example: "{label} ({state_count} states)"
  string collapsed_summary = 3;

  // Whether to show transition count badge when collapsed.
  bool show_transition_count = 4;

  // Icon or glyph for collapsed state (for visual categorization).
  string collapsed_icon = 5;

  // Level-of-detail presets: "auto", "always_expanded", "always_collapsed".
  string lod_preset = 6;
}
```

### 3.3 Level-of-Detail Rendering

| Zoom Level | Display Mode | Example |
|------------|--------------|---------|
| < 25% | Icons only | State shown as colored dot |
| 25-50% | Collapsed summaries | "Overworld: 45 states" |
| 50-75% | Top-level children visible | Regions shown, sub-regions collapsed |
| 75-100% | Most states visible | Only deep nesting collapsed |
| > 100% | Full detail | All states, all labels, all transitions |

### 3.4 Hierarchical Summaries

For composite states, generate automatic summaries:

```
Dungeon_Logic (collapsed at 40% zoom)
+-- "Dungeon Logic" badge
+-- "127 states, 89 transitions" subtitle
+-- Entry points: ENTER_DUNGEON, BOSS_DEFEATED
+-- Visual: Aggregated transition arrows in/out
```

### 3.5 Use Case: Zelda3 (786 states)

```
At 20% zoom:
+---------------------------------------------+
|  [*] Overworld (312)                        |
|  [*] Dungeon Logic (127)                    |
|  [*] Link State (89)                        |
|  [*] Sprite System (156)                    |
|  [*] Menus (42)                             |
|  [*] Intro/Outro (60)                       |
+---------------------------------------------+

At 60% zoom (focused on Dungeon Logic):
+---------------------------------------------+
|  Dungeon Logic                              |
|  +-- Room State (23)                        |
|  +-- Door Logic (18)                        |
|  +-- Puzzle State (31)                      |
|  +-- Boss Encounters (12)                   |
|  +-- Item Rooms (43)                        |
+---------------------------------------------+
```

---

## 4. Machine Views

### 4.1 Problem Statement

Users need to save and recall specific viewpoints:
- "Login Flow" - focused on authentication states
- "Error States" - filtered to show only error handling
- "Happy Path" - specific transitions highlighted

### 4.2 MachineView

```protobuf
// MachineView represents a saved camera position with filters.
// Stored in Statechart-level extensions or a dedicated view registry.
message MachineView {
  // Unique view identifier.
  string id = 1;

  // Human-readable view name.
  string name = 2;

  // Optional description.
  string description = 3;

  // Camera position (canvas center point).
  Point camera_position = 4;

  // Zoom level: 1.0 = 100%.
  double zoom_level = 5;

  // States to highlight (emphasize visually).
  repeated string highlighted_states = 6;

  // States to hide (filter out of view).
  repeated string hidden_states = 7;

  // States to focus (dim everything else).
  repeated string focused_states = 8;

  // Event filter: only show transitions for these events.
  repeated string event_filter = 9;

  // Tag filter: only show states with these tags.
  repeated string tag_filter = 10;

  // Whether this is the default view.
  bool is_default = 11;

  // Keyboard shortcut (e.g., "Cmd+1").
  string shortcut = 12;

  // Thumbnail for view picker (base64 PNG).
  string thumbnail = 13;
}
```

### 4.3 View Registry

```protobuf
// ViewRegistry maintains all saved views for a statechart.
message ViewRegistry {
  // All saved views.
  repeated MachineView views = 1;

  // Default view ID.
  string default_view_id = 2;

  // Recently used views (for quick access).
  repeated string recent_view_ids = 3;
}
```

### 4.4 Use Cases

**Authentication Flow View:**
```json
{
  "id": "auth-flow",
  "name": "Authentication Flow",
  "focused_states": ["Login", "MFA", "SessionActive", "SessionExpired"],
  "event_filter": ["LOGIN", "LOGOUT", "MFA_*", "SESSION_*"],
  "zoom_level": 0.8
}
```

**Error Handling View:**
```json
{
  "id": "error-states",
  "name": "Error States",
  "tag_filter": ["error", "recovery"],
  "highlighted_states": ["NetworkError", "AuthError", "RetryState"]
}
```

---

## 5. Trace Visualization

### 5.1 Problem Statement

Users need to visualize execution history:
- Debug: "How did we get to this error state?"
- Analyze: "Which paths are most common?"
- Review: "Walk through this user's session step by step"

### 5.2 TraceVisualizationConfig

```protobuf
// TraceVisualizationConfig controls how execution traces are rendered.
// Stored in Statechart-level extensions.
message TraceVisualizationConfig {
  // Primary render mode.
  TraceRenderMode render_mode = 1;

  // Number of recent steps to highlight in breadcrumb mode.
  int32 trail_length = 2;

  // Color gradient for recency (newest to oldest).
  // CSS color strings, e.g., ["#00ff00", "#ffff00", "#ff0000"].
  repeated string trail_colors = 3;

  // Animation/playback settings.
  PlaybackConfig playback = 4;

  // Whether to show dwell times between transitions.
  bool show_dwell_times = 5;

  // Heatmap configuration (for frequency analysis).
  HeatmapConfig heatmap = 6;

  // Timeline configuration.
  TimelineConfig timeline = 7;
}
```

### 5.3 Render Modes

```protobuf
// TraceRenderMode determines the visual style for trace display.
enum TraceRenderMode {
  TRACE_RENDER_MODE_UNSPECIFIED = 0;

  // BREADCRUMB: Highlight the path taken through states.
  // Recent states glow brighter, fading for older visits.
  TRACE_RENDER_MODE_BREADCRUMB = 1;

  // HEATMAP: Color states by visit frequency.
  // Hot = frequently visited, cold = rarely visited.
  TRACE_RENDER_MODE_HEATMAP = 2;

  // TIMELINE: Horizontal timeline with state durations.
  // Shows temporal sequence separate from spatial layout.
  TRACE_RENDER_MODE_TIMELINE = 3;

  // SEQUENCE: UML sequence diagram style.
  // Vertical lifelines with message arrows.
  TRACE_RENDER_MODE_SEQUENCE = 4;

  // OVERLAY: Annotate existing layout with trace info.
  // Numbers on states, arrows with timing.
  TRACE_RENDER_MODE_OVERLAY = 5;
}
```

### 5.4 PlaybackConfig

```protobuf
// PlaybackConfig controls trace animation/playback.
message PlaybackConfig {
  // Duration per transition step (milliseconds).
  double step_duration_ms = 1;

  // Whether camera auto-pans to follow active state.
  bool auto_follow = 2;

  // Whether to loop playback.
  bool loop = 3;

  // Playback speed multiplier: 1.0 = normal, 2.0 = 2x speed.
  double speed = 4;

  // Pause on specific events (for debugging).
  repeated string pause_on_events = 5;

  // Pause when entering specific states.
  repeated string pause_on_states = 6;

  // Show action execution during playback.
  bool show_actions = 7;

  // Show guard evaluation during playback.
  bool show_guards = 8;
}
```

### 5.5 HeatmapConfig

```protobuf
// HeatmapConfig controls frequency-based visualization.
message HeatmapConfig {
  // Metric to visualize.
  HeatmapMetric metric = 1;

  // Color for low values (cold).
  string cold_color = 2;

  // Color for high values (hot).
  string hot_color = 3;

  // Whether to use logarithmic scale.
  bool logarithmic = 4;

  // Minimum value for color scale (0 = auto).
  double min_value = 5;

  // Maximum value for color scale (0 = auto).
  double max_value = 6;
}

// HeatmapMetric specifies what data drives the heatmap.
enum HeatmapMetric {
  HEATMAP_METRIC_UNSPECIFIED = 0;

  // VISIT_COUNT: Number of times state was entered.
  HEATMAP_METRIC_VISIT_COUNT = 1;

  // DWELL_TIME: Total time spent in state.
  HEATMAP_METRIC_DWELL_TIME = 2;

  // TRANSITION_FREQUENCY: How often transitions fire.
  HEATMAP_METRIC_TRANSITION_FREQUENCY = 3;

  // ERROR_COUNT: Number of errors in state.
  HEATMAP_METRIC_ERROR_COUNT = 4;
}
```

### 5.6 TimelineConfig

```protobuf
// TimelineConfig controls the timeline visualization mode.
message TimelineConfig {
  // Time scale: "linear", "logarithmic", "compressed".
  string time_scale = 1;

  // Whether to show state labels on timeline bars.
  bool show_labels = 2;

  // Whether to show event markers between states.
  bool show_events = 3;

  // Group parallel regions on separate lanes.
  bool separate_regions = 4;

  // Height of timeline in pixels.
  double height = 5;
}
```

### 5.7 Integration with execution.proto

Trace visualization consumes data from `execution.proto`:

```
ExecutionTrace (execution.proto)
+-- TransitionLogEntry[]
|   +-- timestamp
|   +-- source_config
|   +-- target_config
|   +-- trigger_event
|   +-- processing_time
|
+-- TraceVisualizationConfig (visualization.proto)
    +-- render_mode: HEATMAP
    +-- heatmap.metric: DWELL_TIME
    +-- playback.auto_follow: true

Renderer:
1. Load ExecutionTrace
2. Apply TraceVisualizationConfig
3. For each TransitionLogEntry:
   - Compute dwell time
   - Update state heat values
   - Render according to render_mode
```

---

## 6. Relationship to Other Protos

| Proto | Purpose | Example Fields |
|-------|---------|----------------|
| `layout.proto` | **Static position** - WHERE things are | position, size, anchors, z_index |
| `visualization.proto` | **Dynamic display** - HOW things appear | zoom thresholds, views, trace rendering |
| `execution.proto` | **Trace data** - WHAT happened | log entries, configurations, timestamps |
| `xstate/v1/xstate.proto` | **Editor metadata** - Stately-specific | unique_id, color, assets |

### Data Flow

```
+-------------------------------------------------------------+
|                     VISUALIZATION LAYER                      |
|  +--------------------------------------------------------+ |
|  | visualization.proto                                     | |
|  | * StateDisplayConfig (zoom behavior)                    | |
|  | * MachineView (saved views)                             | |
|  | * TraceVisualizationConfig (trace rendering)            | |
|  +--------------------------------------------------------+ |
|                              |                               |
|                              v                               |
|  +------------------+   +------------------+                 |
|  |  layout.proto    |   | execution.proto  |                 |
|  |  (static layout) |   |  (trace data)    |                 |
|  +------------------+   +------------------+                 |
|                              |                               |
|                              v                               |
|  +--------------------------------------------------------+ |
|  | statecharts.proto (core Harel formalism)                | |
|  | States, Transitions, Events, Guards, Actions            | |
|  +--------------------------------------------------------+ |
+-------------------------------------------------------------+
```

---

## 7. Open Questions

### 7.1 Proto Location

**Options:**
1. `extensions/v1/visualization.proto` - alongside layout.proto
2. `visualization/v1/visualization.proto` - separate package
3. Add to `extensions/v1/layout.proto` - extend existing

**Recommendation:** Option 1 (`extensions/v1/visualization.proto`) - maintains
separation of concerns while keeping related extensions together.

### 7.2 Per-State vs Machine-Level Config

**StateDisplayConfig**: Per-state (different zoom thresholds per region)
**MachineView**: Machine-level (views apply to whole chart)
**TraceVisualizationConfig**: Machine-level (one trace config per visualization)

### 7.3 Integration with web-visualizer

The `cmd/web-visualizer` should:
1. Read `StateDisplayConfig` from state extensions
2. Implement semantic zoom based on canvas zoom level
3. Store `MachineView` in local storage or chart metadata
4. Accept `ExecutionTrace` for trace visualization modes

### 7.4 Defaults for Large Charts

When no `StateDisplayConfig` is provided, auto-compute thresholds:
- Charts with >100 states: auto-collapse at 50% zoom
- Charts with >500 states: auto-collapse at 75% zoom
- Parallel regions: collapse independently

### 7.5 Animation Performance

For large traces (>10,000 entries):
- Downsample for heatmap (aggregate into time buckets)
- Virtual scrolling for timeline
- Progressive loading for breadcrumb trail

---

## 8. Metrics Visualization

### 8.1 Problem Statement

Production statecharts need runtime performance visibility:
- "Which transitions are bottlenecks?"
- "What's the p99 latency for this flow?"
- "Which states have high error rates?"

### 8.2 TransitionMetrics

```protobuf
// TransitionMetrics captures runtime performance data for a transition.
// Computed from execution.proto traces, stored for visualization.
message TransitionMetrics {
  // Transition identifier (label or from->to).
  string transition_id = 1;

  // Throughput: transitions per second (rolling average).
  double throughput_per_sec = 2;

  // Latency distribution.
  LatencyPercentiles latency = 3;

  // Error rate: fraction of firings that resulted in error.
  double error_rate = 4;

  // Total fire count in measurement window.
  uint64 fire_count = 5;

  // Measurement time window.
  google.protobuf.Duration window = 6;
}

// LatencyPercentiles captures latency distribution.
message LatencyPercentiles {
  double p50_ms = 1;   // Median
  double p95_ms = 2;   // 95th percentile
  double p99_ms = 3;   // 99th percentile
  double max_ms = 4;   // Maximum observed
  double min_ms = 5;   // Minimum observed
  double mean_ms = 6;  // Arithmetic mean
}
```

### 8.3 StateMetrics

```protobuf
// StateMetrics captures runtime data for a state.
message StateMetrics {
  // State label.
  string state_label = 1;

  // Dwell time distribution (time spent in state).
  LatencyPercentiles dwell_time = 2;

  // Entry count in measurement window.
  uint64 entry_count = 3;

  // Error count while in this state.
  uint64 error_count = 4;

  // Currently active instance count (for concurrent machines).
  uint64 active_count = 5;
}
```

### 8.4 MetricsVisualizationConfig

```protobuf
// MetricsVisualizationConfig controls how metrics are rendered.
message MetricsVisualizationConfig {
  // Enable metrics overlay on diagram.
  bool enabled = 1;

  // Which metric drives edge thickness (throughput, fire_count).
  string edge_thickness_metric = 2;

  // Which metric drives edge color (latency_p99, error_rate).
  string edge_color_metric = 3;

  // Color gradient for edge coloring.
  ColorGradient edge_gradient = 4;

  // Show badges with numeric values.
  bool show_badges = 5;

  // Badge format: "{throughput}/s" or "p99: {latency_p99}ms".
  string badge_format = 6;

  // Thresholds for warning/error highlighting.
  MetricsThresholds thresholds = 7;
}

// ColorGradient defines a gradient for metric visualization.
message ColorGradient {
  string low_color = 1;    // e.g., "#00ff00" (green = good)
  string mid_color = 2;    // e.g., "#ffff00" (yellow = warning)
  string high_color = 3;   // e.g., "#ff0000" (red = critical)
  double mid_threshold = 4; // Value at which mid_color applies
}

// MetricsThresholds for alerting/highlighting.
message MetricsThresholds {
  double latency_warning_ms = 1;   // Yellow highlight
  double latency_critical_ms = 2;  // Red highlight
  double error_rate_warning = 3;   // e.g., 0.01 = 1%
  double error_rate_critical = 4;  // e.g., 0.05 = 5%
}
```

### 8.5 Rendering Examples

**Edge thickness by throughput:**
```
Thin edge:   ------>  (10 tx/sec)
Medium edge: ======>  (100 tx/sec)
Thick edge:  ########> (1000 tx/sec)
```

**Edge color by latency:**
```
Green edge:  ------>  (p99 < 50ms)
Yellow edge: ------>  (p99 50-200ms)
Red edge:    ------>  (p99 > 200ms)
```

**State badges:**
```
+---------------------------+
|  Processing               |
|  +----------------------+ |
|  | err: 2.3%            | |
|  | dwell: 45ms          | |
|  +----------------------+ |
+---------------------------+
```

---

## 9. Machine Comparison View

### 9.1 Problem Statement

Users need to compare statecharts:
- "What changed between v1.2 and v1.3?"
- "Does the A/B test variant have all the same error handling?"
- "Which events are missing after migration?"

### 9.2 ComparisonView

```protobuf
// ComparisonView configures side-by-side machine comparison.
message ComparisonView {
  // Machines to compare (2+ IDs or versions).
  repeated string machine_ids = 1;

  // Comparison layout mode.
  ComparisonMode mode = 2;

  // Whether pan/zoom is synchronized across views.
  bool sync_navigation = 3;

  // Diff highlighting configuration.
  DiffHighlightConfig highlight = 4;

  // Labels for each machine (e.g., "v1.2", "v1.3").
  repeated string labels = 5;

  // Whether to show only differences (hide unchanged).
  bool diff_only = 6;
}

// ComparisonMode determines layout for comparison.
enum ComparisonMode {
  COMPARISON_MODE_UNSPECIFIED = 0;

  // SIDE_BY_SIDE: Machines displayed in adjacent panels.
  COMPARISON_MODE_SIDE_BY_SIDE = 1;

  // OVERLAY: Machines superimposed with transparency.
  COMPARISON_MODE_OVERLAY = 2;

  // DIFF_ONLY: Show only elements that differ.
  COMPARISON_MODE_DIFF_ONLY = 3;

  // UNIFIED: Single view with diff annotations.
  COMPARISON_MODE_UNIFIED = 4;
}
```

### 9.3 DiffHighlightConfig

```protobuf
// DiffHighlightConfig controls how differences are highlighted.
message DiffHighlightConfig {
  // Colors for different diff types.
  string added_color = 1;      // e.g., "#00ff00" (green)
  string removed_color = 2;    // e.g., "#ff0000" (red)
  string modified_color = 3;   // e.g., "#ffff00" (yellow)
  string unchanged_color = 4;  // e.g., "#888888" (gray)

  // Opacity for unchanged elements (0.0-1.0).
  double unchanged_opacity = 5;

  // Show diff badges ("+", "-", "~").
  bool show_badges = 6;

  // Animate diff highlighting (pulse effect).
  bool animate = 7;
}

// DiffType classifies how an element differs.
enum DiffType {
  DIFF_TYPE_UNSPECIFIED = 0;
  DIFF_TYPE_ADDED = 1;      // Present in B, not in A
  DIFF_TYPE_REMOVED = 2;    // Present in A, not in B
  DIFF_TYPE_MODIFIED = 3;   // Present in both, but changed
  DIFF_TYPE_UNCHANGED = 4;  // Identical in both
}
```

### 9.4 StateDiff / TransitionDiff

```protobuf
// StateDiff captures differences for a single state.
message StateDiff {
  // State label (common identifier).
  string label = 1;

  // Diff classification.
  DiffType diff_type = 2;

  // Specific changes (for MODIFIED).
  repeated string changes = 3;  // e.g., ["type: OR -> AND", "added child: X"]
}

// TransitionDiff captures differences for a transition.
message TransitionDiff {
  // Transition identifier.
  string from = 1;
  string to = 2;
  string event = 3;

  // Diff classification.
  DiffType diff_type = 4;

  // Specific changes.
  repeated string changes = 5;  // e.g., ["guard added", "action removed"]
}
```

### 9.5 Comparison Summary

```protobuf
// ComparisonSummary provides high-level diff statistics.
message ComparisonSummary {
  uint32 states_added = 1;
  uint32 states_removed = 2;
  uint32 states_modified = 3;
  uint32 states_unchanged = 4;

  uint32 transitions_added = 5;
  uint32 transitions_removed = 6;
  uint32 transitions_modified = 7;
  uint32 transitions_unchanged = 8;

  // Events in A but not B.
  repeated string events_removed = 9;

  // Events in B but not A.
  repeated string events_added = 10;

  // Breaking changes detected.
  repeated string breaking_changes = 11;
}
```

### 9.6 Use Cases

**Version Comparison (chart evolution):**
```
+---------------------+   +---------------------+
|      v1.2           |   |      v1.3           |
+---------------------+   +---------------------+
|  [Login]            |   |  [Login]            |
|     |               |   |     |               |
|     v               |   |     v               |
|  [Dashboard]        |   |  [MFA] <- NEW       |
|                     |   |     |               |
|                     |   |     v               |
|                     |   |  [Dashboard]        |
+---------------------+   +---------------------+

Summary: +1 state (MFA), +2 transitions
```

**A/B Test Variant Comparison:**
```
Overlay mode with transparency:
- Variant A states: solid blue
- Variant B states: solid green
- Shared states: purple (blended)
- Unique to A: blue outline
- Unique to B: green outline
```

### 9.7 Integration with evolution.proto

Comparison view consumes `ChartDiff` from `evolution.proto`:

```
evolution.proto                visualization.proto
+--------------------+        +--------------------+
| ChartDiff          |------->| ComparisonView     |
| - states_added     |        | - machine_ids      |
| - states_removed   |        | - mode             |
| - transitions_mod  |        | - highlight        |
+--------------------+        +--------------------+
```

---

## 10. Population Comparison View

### 10.1 Problem Statement

Compare how different cohorts flow through the SAME machine:
- "How do mobile vs desktop users navigate differently?"
- "What's the difference between free and premium user paths?"
- "How did behavior change after the feature flag?"

### 10.2 PopulationView

```protobuf
// PopulationView compares multiple user cohorts on the same machine.
message PopulationView {
  // Machine to analyze.
  string machine_id = 1;

  // Populations to compare (2+ required).
  repeated PopulationConfig populations = 2;

  // How to display the comparison.
  PopulationDisplayMode display = 3;

  // Show percentages vs absolute counts.
  bool normalize_counts = 4;

  // Color scheme: "distinct", "gradient", "categorical".
  string color_scheme = 5;
}

// PopulationConfig defines a cohort for comparison.
message PopulationConfig {
  // Unique population identifier.
  string id = 1;

  // Display name: "Mobile Users", "Power Users", "Test A".
  string name = 2;

  // Distinct color for this population.
  string color = 3;

  // Trace filter criteria.
  TraceFilter filter = 4;
}

// TraceFilter selects which traces belong to a population.
message TraceFilter {
  // Time range for traces.
  TimeRange time_range = 1;

  // User cohort/segment IDs.
  repeated string user_cohorts = 2;

  // Custom metadata filters (key-value match).
  map<string, string> metadata = 3;

  // Event presence filter: traces must contain these events.
  repeated string required_events = 4;

  // State filter: traces must visit these states.
  repeated string required_states = 5;
}

// TimeRange for filtering.
message TimeRange {
  google.protobuf.Timestamp start = 1;
  google.protobuf.Timestamp end = 2;
}
```

### 10.3 Display Modes

```protobuf
// PopulationDisplayMode determines visualization layout.
enum PopulationDisplayMode {
  POPULATION_DISPLAY_MODE_UNSPECIFIED = 0;

  // SIDE_BY_SIDE: Split view with one chart per population.
  POPULATION_DISPLAY_MODE_SIDE_BY_SIDE = 1;

  // OVERLAY: All populations on same chart with distinct colors.
  POPULATION_DISPLAY_MODE_OVERLAY = 2;

  // SANKEY: Flow diagram where width = volume per population.
  POPULATION_DISPLAY_MODE_SANKEY = 3;

  // HEATMAP_GRID: Grid with populations as rows, states as columns.
  POPULATION_DISPLAY_MODE_HEATMAP_GRID = 4;
}
```

### 10.4 Visualization Examples

**Overlay Mode:**
```
+-----------------------------------------+
|  [Login]                                |
|     |                                   |
|     v                                   |
|  [Dashboard] -----------------------    |
|     | ####.... (Desktop: 80%)           |
|     | ....#### (Mobile: 45%)            |
|     v                                   |
|  [Checkout]                             |
|     | ######.. (Desktop: 75%)           |
|     | ..##.... (Mobile: 25%)            |
|     v                                   |
|  [Complete]                             |
+-----------------------------------------+
Edge widths show volume per population
```

**Sankey Mode:**
```
Login ===============+
      (Desktop: 80%) |==> Dashboard ===+
                     |                 |==> Checkout ==> Complete
      (Mobile: 45%)  |==> Dashboard ===+     (divergent paths)
                     +=====================> Exit
```

**Heatmap Grid:**
```
                 Login  Dashboard  Checkout  Complete  Exit
Desktop Users    ####   ####       ###       ##        .
Mobile Users     ####   ##         .         .         ###
Power Users      ####   ####       ####      ####      .
```

### 10.5 Use Cases

| Use Case | Population A | Population B | Insight |
|----------|--------------|--------------|---------|
| Platform | Desktop | Mobile | UX optimization |
| Pricing | Free tier | Premium | Conversion drivers |
| A/B Test | Control | Treatment | Feature impact |
| Time | Before launch | After launch | Behavior change |
| Zelda3 | Speedrunners | Casual | Strategy differences |

---

## 11. Business Metrics Overlay

### 11.1 Problem Statement

Annotate states with business dimensions:
- "Which states generate revenue?"
- "Where are we losing money (cost > revenue)?"
- "What's the fastest path to conversion?"

### 11.2 BusinessMetrics

```protobuf
// BusinessMetrics attaches financial data to statechart elements.
message BusinessMetrics {
  // Per-state business data.
  repeated StateBusinessMetrics states = 1;

  // Per-transition business data.
  repeated TransitionBusinessMetrics transitions = 2;

  // Currency for all monetary values.
  string currency = 3;

  // Measurement period.
  TimeRange period = 4;
}

// StateBusinessMetrics captures business impact of a state.
message StateBusinessMetrics {
  // State identifier.
  string state_id = 1;

  // Revenue generated while in this state.
  MonetaryValue revenue = 2;

  // Cost incurred (compute, support, etc).
  MonetaryValue cost = 3;

  // Average dwell time.
  google.protobuf.Duration avg_dwell_time = 4;

  // Conversion rate: % that proceed to next state vs exit.
  double conversion_rate = 5;

  // Churn risk: probability of abandonment from this state.
  double churn_risk = 6;

  // Lifetime value of users reaching this state.
  MonetaryValue ltv = 7;

  // Sample size for statistical confidence.
  uint64 sample_size = 8;
}

// TransitionBusinessMetrics captures value of transitions.
message TransitionBusinessMetrics {
  // Transition identifier.
  string transition_id = 1;

  // Transaction value (e.g., purchase amount).
  MonetaryValue transaction_value = 2;

  // Average latency to complete transition.
  google.protobuf.Duration avg_latency = 3;

  // Cost per transition (API costs, processing, etc).
  MonetaryValue cost_per_transition = 4;

  // Completion rate (success / attempts).
  double completion_rate = 5;
}

// MonetaryValue represents a currency amount.
message MonetaryValue {
  double amount = 1;
  string currency = 2;  // ISO 4217: "USD", "EUR", "GBP"
}
```

### 11.3 BusinessViewConfig

```protobuf
// BusinessViewConfig controls business metrics visualization.
message BusinessViewConfig {
  // Primary metric (drives size/thickness).
  BusinessMetricType primary_metric = 1;

  // Secondary metric (drives color).
  BusinessMetricType secondary_metric = 2;

  // Show cumulative values along paths.
  bool show_cumulative = 3;

  // Highlight the highest-value route.
  bool highlight_critical_path = 4;

  // Highlight bottlenecks (high dwell + low conversion).
  bool highlight_bottlenecks = 5;

  // Show time-to-value annotations.
  bool show_time_to_value = 6;

  // Profitability threshold for color coding.
  double profit_threshold = 7;
}

// BusinessMetricType specifies which metric to visualize.
enum BusinessMetricType {
  BUSINESS_METRIC_TYPE_UNSPECIFIED = 0;
  BUSINESS_METRIC_TYPE_REVENUE = 1;
  BUSINESS_METRIC_TYPE_COST = 2;
  BUSINESS_METRIC_TYPE_PROFIT = 3;       // revenue - cost
  BUSINESS_METRIC_TYPE_DWELL_TIME = 4;
  BUSINESS_METRIC_TYPE_CONVERSION = 5;
  BUSINESS_METRIC_TYPE_CHURN_RISK = 6;
  BUSINESS_METRIC_TYPE_LTV = 7;          // Lifetime value
  BUSINESS_METRIC_TYPE_TRANSACTION = 8;  // Per-transition value
}
```

### 11.4 Visualization Modes

**State size by revenue:**
```
+-------------------------------------------+
|                                           |
|   +-----+                                 |
|   |Cart |  ------->  +---------------+    |
|   | $0  |            |   Checkout    |    |
|   +-----+            |    $500K      |    |
|                      +---------------+    |
|                            |              |
|                            v              |
|                +---------------------+    |
|                |     Complete        |    |
|                |      $2.5M          |    |
|                +---------------------+    |
+-------------------------------------------+
```

**Profitability color coding:**
```
Green:  Profit > 0 (revenue exceeds cost)
Yellow: Break-even (cost ~ revenue)
Red:    Loss (cost exceeds revenue)

+--------------------------------+
| [Purchase] +$45/user  (green)  |
|      |                         |
|      v                         |
| [Support] ~$0/user   (yellow)  |
|      |                         |
|      v                         |
| [Refund] -$60/user   (red)     |
+--------------------------------+
```

**Critical path highlighting:**
```
Login --> Onboarding --> [Trial] ======> [PURCHASE] ==> Active
                            |
                            +--> Exit (lost: $150K)

=== = Critical path (highest value)
--- = Secondary paths
```

### 11.5 Use Cases

| Industry | State | Metric | Insight |
|----------|-------|--------|---------|
| E-commerce | Checkout | Revenue | Funnel optimization |
| SaaS | Onboarding | Cost | CAC reduction |
| Gaming | In-app purchase | Transaction | Monetization |
| Support | Ticket open | Dwell time | Resolution speed |
| Fintech | KYC pending | Churn risk | Drop-off prevention |

---

## 12. Real-Time Streaming

### 12.1 Problem Statement

Production machines generate continuous state changes. Visualizers need to:
- Display live trace updates without polling
- Handle high-throughput machines (1000+ transitions/sec)
- Gracefully degrade when overwhelmed

### 12.2 StreamingConfig

```protobuf
// StreamingConfig controls real-time trace updates.
message StreamingConfig {
  // Transport protocol: "websocket", "sse", "grpc-stream".
  string transport = 1;

  // Update rate limiting.
  RateLimitConfig rate_limit = 2;

  // Buffering strategy for burst handling.
  BufferConfig buffer = 3;

  // Connection settings.
  ConnectionConfig connection = 4;

  // Which data to stream.
  StreamFilter filter = 5;
}

// RateLimitConfig prevents UI overload.
message RateLimitConfig {
  // Maximum updates per second to render.
  // Excess updates are aggregated/sampled.
  uint32 max_updates_per_sec = 1;

  // Aggregation strategy when rate-limited.
  // "sample": Show every Nth update
  // "aggregate": Batch into summary updates
  // "latest": Show only most recent state
  string overflow_strategy = 2;

  // Minimum interval between renders (milliseconds).
  uint32 min_render_interval_ms = 3;
}

// BufferConfig handles burst traffic.
message BufferConfig {
  // Maximum events to buffer before dropping.
  uint32 max_buffer_size = 1;

  // Buffer flush interval (milliseconds).
  uint32 flush_interval_ms = 2;

  // What to do when buffer is full.
  // "drop_oldest", "drop_newest", "backpressure"
  string overflow_policy = 3;
}

// ConnectionConfig for transport layer.
message ConnectionConfig {
  // Reconnect on disconnect.
  bool auto_reconnect = 1;

  // Reconnect delay (milliseconds).
  uint32 reconnect_delay_ms = 2;

  // Maximum reconnection attempts (0 = unlimited).
  uint32 max_reconnect_attempts = 3;

  // Heartbeat interval for connection health.
  uint32 heartbeat_interval_ms = 4;
}

// StreamFilter selects which updates to receive.
message StreamFilter {
  // Only stream transitions involving these states.
  repeated string state_filter = 1;

  // Only stream these event types.
  repeated string event_filter = 2;

  // Include context/variable changes.
  bool include_context = 3;

  // Include action execution results.
  bool include_actions = 4;
}
```

### 12.3 High-Throughput Strategies

| Throughput | Strategy | Visual Effect |
|------------|----------|---------------|
| < 10/sec | Full fidelity | Every transition animated |
| 10-100/sec | Sampled | 1-in-N transitions shown |
| 100-1000/sec | Aggregated | Batch updates, no animation |
| > 1000/sec | Summary only | Counters + heatmap, no individual events |

### 12.4 Graceful Degradation

```
Normal mode (< 50 updates/sec):
  - Animate each transition
  - Update breadcrumb trail
  - Show event labels

Degraded mode (50-500 updates/sec):
  - Skip animation
  - Update current state only
  - Aggregate counters

Overload mode (> 500 updates/sec):
  - Pause real-time updates
  - Show "High activity" indicator
  - Switch to periodic snapshots
```

---

## 13. Accessibility

### 13.1 Problem Statement

Visualization must be usable by all users:
- Color-blind users (8% of males)
- Screen reader users
- Keyboard-only navigation
- Users with motion sensitivity

### 13.2 AccessibilityConfig

```protobuf
// AccessibilityConfig ensures inclusive visualization.
message AccessibilityConfig {
  // Color palette selection.
  ColorPalette palette = 1;

  // Screen reader configuration.
  ScreenReaderConfig screen_reader = 2;

  // Keyboard navigation settings.
  KeyboardConfig keyboard = 3;

  // Motion/animation settings.
  MotionConfig motion = 4;

  // High contrast mode.
  bool high_contrast = 5;

  // Minimum touch target size (pixels).
  uint32 min_touch_target = 6;
}

// ColorPalette for color-blind safe visualization.
message ColorPalette {
  // Palette type: "default", "deuteranopia", "protanopia",
  // "tritanopia", "monochrome", "high_contrast".
  string type = 1;

  // Override colors for specific semantic meanings.
  map<string, string> semantic_colors = 2;

  // Use patterns in addition to colors for differentiation.
  bool use_patterns = 3;

  // Use shapes in addition to colors.
  bool use_shapes = 4;
}

// ScreenReaderConfig for assistive technology.
message ScreenReaderConfig {
  // Enable ARIA labels on all elements.
  bool enable_aria = 1;

  // Verbosity level: "minimal", "normal", "verbose".
  string verbosity = 2;

  // Announce state transitions.
  bool announce_transitions = 3;

  // Announce metrics changes.
  bool announce_metrics = 4;

  // Custom announcement templates.
  map<string, string> templates = 5;
}

// KeyboardConfig for keyboard-only navigation.
message KeyboardConfig {
  // Enable keyboard navigation.
  bool enabled = 1;

  // Focus indicator style: "outline", "highlight", "both".
  string focus_style = 2;

  // Custom key bindings.
  map<string, string> bindings = 3;

  // Tab order: "visual", "hierarchical", "alphabetical".
  string tab_order = 4;
}

// MotionConfig for users with vestibular disorders.
message MotionConfig {
  // Reduce or disable animations.
  // "full", "reduced", "none"
  string animation_level = 1;

  // Disable auto-scrolling/panning.
  bool disable_auto_pan = 2;

  // Disable zoom animations.
  bool disable_zoom_animation = 3;

  // Respect prefers-reduced-motion OS setting.
  bool respect_os_setting = 4;
}
```

### 13.3 Color-Blind Safe Palettes

**Default (problematic):**
```
Good:    #00FF00 (green)   <- Indistinguishable for deuteranopia
Warning: #FFFF00 (yellow)
Error:   #FF0000 (red)     <- Indistinguishable for deuteranopia
```

**Deuteranopia-safe:**
```
Good:    #0077BB (blue)
Warning: #EE7733 (orange)
Error:   #CC3311 (dark red)
Neutral: #009988 (teal)
```

**Universal (pattern-based):**
```
Good:    Blue + solid fill
Warning: Orange + diagonal stripes
Error:   Red + crosshatch
Neutral: Gray + dots
```

### 13.4 Screen Reader Announcements

```
State entry:
  "Entered state: Checkout. 3 outgoing transitions available."

Transition:
  "Transition: SUBMIT fired. Moving from Checkout to Processing."

Metrics alert:
  "Warning: Error rate in Payment state exceeded 5%."

Playback:
  "Playback paused at step 15 of 42. Current state: Retry."
```

### 13.5 Keyboard Navigation

| Key | Action |
|-----|--------|
| Tab | Move focus to next state |
| Shift+Tab | Move focus to previous state |
| Enter | Expand/collapse state |
| Arrow keys | Navigate within hierarchy |
| Space | Toggle selection |
| / | Open search |
| P | Play/pause trace |
| [ / ] | Step backward/forward |
| Escape | Close dialogs, clear selection |

---

## 14. Alerting Visualization

### 14.1 Problem Statement

Users need immediate visual feedback when:
- Metrics exceed thresholds (SLA breach)
- Anomalies detected (unusual patterns)
- Errors occur (transition failures)
- System health degrades

### 14.2 AlertConfig

```protobuf
// AlertConfig defines visual indicators for threshold violations.
message AlertConfig {
  // Alert rules to evaluate.
  repeated AlertRule rules = 1;

  // Global alert display settings.
  AlertDisplayConfig display = 2;

  // Alert history settings.
  AlertHistoryConfig history = 3;
}

// AlertRule defines when to trigger visual alerts.
message AlertRule {
  // Unique rule identifier.
  string id = 1;

  // Human-readable name.
  string name = 2;

  // Condition expression (CEL or simple comparison).
  // Examples:
  //   "metrics.error_rate > 0.05"
  //   "metrics.latency_p99 > 200"
  //   "state.dwell_time > duration('5m')"
  string condition = 3;

  // Severity: "info", "warning", "error", "critical".
  string severity = 4;

  // Which elements this rule applies to.
  // "all_states", "all_transitions", or specific labels.
  repeated string targets = 5;

  // Visual treatment when triggered.
  AlertVisual visual = 6;

  // Notification settings.
  AlertNotification notification = 7;
}

// AlertVisual defines how to display triggered alerts.
message AlertVisual {
  // Border/outline color for affected element.
  string border_color = 1;

  // Border style: "solid", "dashed", "pulse".
  string border_style = 2;

  // Background highlight color.
  string background_color = 3;

  // Icon to display: "warning", "error", "info", custom.
  string icon = 4;

  // Badge text: "{value}", "SLA BREACH", custom.
  string badge_text = 5;

  // Animation: "none", "pulse", "shake", "glow".
  string animation = 6;
}

// AlertNotification for non-visual feedback.
message AlertNotification {
  // Show toast/banner notification.
  bool show_toast = 1;

  // Play sound alert.
  bool play_sound = 2;

  // Sound type: "chime", "alarm", "subtle".
  string sound_type = 3;

  // Screen reader announcement.
  string announcement = 4;
}

// AlertDisplayConfig for global alert rendering.
message AlertDisplayConfig {
  // Show alert summary panel.
  bool show_summary_panel = 1;

  // Panel position: "top", "bottom", "left", "right".
  string panel_position = 2;

  // Maximum alerts to show in panel.
  uint32 max_visible_alerts = 3;

  // Group alerts by severity.
  bool group_by_severity = 4;

  // Auto-dismiss after duration (0 = manual dismiss).
  uint32 auto_dismiss_seconds = 5;
}

// AlertHistoryConfig for tracking past alerts.
message AlertHistoryConfig {
  // Retain alert history.
  bool enabled = 1;

  // Maximum history entries.
  uint32 max_entries = 2;

  // Show resolved alerts differently.
  bool show_resolved = 3;

  // Retention period for history.
  google.protobuf.Duration retention = 4;
}
```

### 14.3 Severity Indicators

```
+------------------------------------------------------------------+
|  SEVERITY VISUAL HIERARCHY                                        |
+------------------------------------------------------------------+
|                                                                   |
|  CRITICAL (outage/breach)     ERROR (failures)                    |
|  +-------------------+        +-------------------+               |
|  |  ////State////    |        |   ##State##       |               |
|  |  ////     ////    |        |   ##     ##       |               |
|  | [!] RED PULSE     |        | [X] RED SOLID     |               |
|  +-------------------+        +-------------------+               |
|                                                                   |
|  WARNING (threshold)          INFO (attention)                    |
|  +-------------------+        +-------------------+               |
|  |   State           |        |   State           |               |
|  |   ~~~~~~          |        |   . . . .         |               |
|  | [!] ORANGE GLOW   |        | [i] BLUE SUBTLE   |               |
|  +-------------------+        +-------------------+               |
|                                                                   |
+------------------------------------------------------------------+
```

### 14.4 Alert Examples

**SLA Breach:**
```json
{
  "id": "sla-latency",
  "name": "Latency SLA Breach",
  "condition": "metrics.latency_p99 > 500",
  "severity": "critical",
  "targets": ["all_transitions"],
  "visual": {
    "border_color": "#FF0000",
    "border_style": "pulse",
    "icon": "warning",
    "badge_text": "SLA BREACH"
  }
}
```

**Anomaly Detection:**
```json
{
  "id": "anomaly-traffic",
  "name": "Traffic Anomaly",
  "condition": "metrics.throughput < baseline * 0.5",
  "severity": "warning",
  "targets": ["all_states"],
  "visual": {
    "border_color": "#FFA500",
    "icon": "info",
    "badge_text": "ANOMALY"
  }
}
```

### 14.5 Alert Panel Layout

```
+--------------------------------------------------+
| ALERTS (3 active)                           [X]  |
+--------------------------------------------------+
| [!] CRITICAL  Payment latency > 500ms     2m ago |
|     State: Payment -> Confirmation               |
+--------------------------------------------------+
| [!] WARNING   Error rate > 5%             5m ago |
|     State: Checkout                              |
+--------------------------------------------------+
| [i] INFO      New deployment detected    10m ago |
|     Machine: auth-flow-v2.3                      |
+--------------------------------------------------+
| [Show resolved (2)]                              |
+--------------------------------------------------+
```

---

## 15. Implementation Plan

1. **Phase 1**: Create `proto/extensions/v1/visualization.proto`
2. **Phase 2**: Generate Go code, update SDKs
3. **Phase 3**: Implement semantic zoom in web-visualizer
4. **Phase 4**: Add view save/load to web-visualizer
5. **Phase 5**: Implement trace visualization modes
6. **Phase 6**: Add metrics overlay support
7. **Phase 7**: Implement comparison view
8. **Phase 8**: Add population comparison
9. **Phase 9**: Add business metrics overlay
10. **Phase 10**: Add real-time streaming support
11. **Phase 11**: Implement accessibility features
12. **Phase 12**: Add alerting visualization
13. **Phase 13**: Performance optimization for large charts

---

## 16. Proto Organization Summary

All visualization extensions in `proto/extensions/v1/visualization.proto`:

| Message | Scope | Purpose |
|---------|-------|---------|
| `StateDisplayConfig` | Per-state | Semantic zoom thresholds |
| `MachineView` | Machine | Saved camera + filters |
| `ViewRegistry` | Machine | View collection |
| `TraceVisualizationConfig` | Machine | Trace rendering |
| `PlaybackConfig` | Machine | Animation controls |
| `HeatmapConfig` | Machine | Frequency visualization |
| `TimelineConfig` | Machine | Timeline mode settings |
| `TransitionMetrics` | Per-transition | Runtime performance |
| `StateMetrics` | Per-state | Runtime performance |
| `MetricsVisualizationConfig` | Machine | Metrics overlay |
| `ComparisonView` | Multi-machine | Side-by-side diff |
| `DiffHighlightConfig` | Machine | Diff styling |
| `PopulationView` | Machine | Cohort comparison |
| `PopulationConfig` | Per-population | Cohort definition |
| `BusinessMetrics` | Machine | Financial data |
| `BusinessViewConfig` | Machine | Business visualization |
| `StreamingConfig` | Machine | Real-time update settings |
| `RateLimitConfig` | Machine | Throughput throttling |
| `BufferConfig` | Machine | Burst handling |
| `AccessibilityConfig` | Machine | Inclusive design settings |
| `ColorPalette` | Machine | Color-blind safe palettes |
| `ScreenReaderConfig` | Machine | Assistive technology |
| `KeyboardConfig` | Machine | Keyboard navigation |
| `MotionConfig` | Machine | Animation preferences |
| `AlertConfig` | Machine | Threshold alerting |
| `AlertRule` | Per-alert | Individual alert definition |
| `AlertVisual` | Per-alert | Alert rendering style |

---

## References

- [H87] D. Harel, "Statecharts: A visual formalism for complex systems," 1987
- [HN96] D. Harel and A. Naamad, "The STATEMATE semantics of statecharts," 1996
- Zelda3 statechart extraction (786 states, motivating use case)
- Stately.ai editor (commercial implementation reference)
