---
title: extensions.v1
description: API Specification for the extensions.v1 package.
---

<a name="layout-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="extensions-v1-StateLayout"></a>

### StateLayout

StateLayout provides visual positioning for a state node.

STORAGE:
Pack into State.extensions using google.protobuf.Any:
  layout := &extensionspb.StateLayout{...}
  any, _ := anypb.New(layout)
  state.Extensions = append(state.Extensions, any)

WELL-FORMEDNESS:
1. position SHOULD be set for non-root states
2. size SHOULD be set for composite states (OR/AND)
3. z_index: higher values render on top of lower values
4. anchors: indices referenced by TransitionLayout.source_anchor/target_anchor

SEMANTIC PROPERTIES:
- position_locked = true: Layout algorithms MUST NOT reposition this state
- layout_hint: Suggested algorithm, tools MAY ignore




| Field | Type | Description |
| ----- | ---- | ----------- |
| position |[Point](#extensions-v1-Point)| Position of state bounding box origin (top-left corner). In parent's coordinate space for nested states.   |
| size |[Dimensions](#extensions-v1-Dimensions)| Bounding box dimensions. For BASIC states, renderer may use defaults if unset.   |
| style |[VisualStyle](#extensions-v1-VisualStyle)| Visual styling hints. Tools SHOULD interpret these but MAY apply defaults.   |
| z_index |int32| Z-ordering: higher values render on top. Default 0; use negative values for background elements.   |
| anchors[] |[Anchor](#extensions-v1-Anchor)| Named anchor points for transition attachment. If empty, tools use default edge midpoints.   |
| layout_hint |string| Layout algorithm hint for automatic positioning. Values: "manual", "auto", "force-directed", "hierarchical", "orthogonal". Tools use their default if unset or unrecognized.   |
| position_locked |bool| Whether position is user-specified (locked) vs auto-computed. When true, layout algorithms MUST preserve this position.   |
| min_size |[Dimensions](#extensions-v1-Dimensions)| Minimum size constraint (for resizable states).   |
| max_size |[Dimensions](#extensions-v1-Dimensions)| Maximum size constraint.   |
| padding |[EdgeInsets](#extensions-v1-EdgeInsets)| Padding inside composite states (for child layout).   |
| collapsed |bool| Expansion state for collapsible composite states.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TransitionLayout"></a>

### TransitionLayout

TransitionLayout provides visual routing for transition edges.

STORAGE:
Pack into Transition.extensions using google.protobuf.Any.

EDGE ROUTING:
Transitions are rendered as directed edges from source to target.
The waypoints field defines intermediate control points:
  - Empty: direct edge (straight or curved per edge_style)
  - Non-empty: polyline through waypoints

LABEL POSITIONING:
Labels (event, guard, action) are positioned along the edge:
  - label_position: parametric position t ∈ [0,1]
    - 0.0 = at source
    - 0.5 = midpoint (default)
    - 1.0 = at target
  - label_offset: perpendicular displacement
    - positive = right side (following edge direction)
    - negative = left side




| Field | Type | Description |
| ----- | ---- | ----------- |
| waypoints[] |[Point](#extensions-v1-Point)| Ordered waypoints for edge routing. Excludes source and target endpoints (those come from states).   |
| edge_style |string| Edge rendering style. "straight": direct line between waypoints "curved": smooth bezier curves "orthogonal": right-angle segments only "bezier": explicit bezier control points   |
| arrow_style |string| Arrowhead style at target end. "filled": solid triangle (default) "open": outlined triangle "none": no arrowhead   |
| label_position |double| Label position along edge: t ∈ [0.0, 1.0]. 0.0 = source, 0.5 = midpoint, 1.0 = target.   |
| label_offset |double| Label offset perpendicular to edge. Positive = right side, negative = left side.   |
| style |[VisualStyle](#extensions-v1-VisualStyle)| Visual styling for the edge.   |
| source_anchor |int32| Source anchor index (references StateLayout.anchors by position). -1 or unset = auto-select best anchor.   |
| target_anchor |int32| Target anchor index.   |
| self_loop |[SelfLoopLayout](#extensions-v1-SelfLoopLayout)| Self-loop properties (when source == target).   |
| curvature |double| Curvature for curved edges: 0.0 = straight, positive = curve right.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SelfLoopLayout"></a>

### SelfLoopLayout

SelfLoopLayout defines rendering for self-transitions.
When a transition's source and target are the same state,
it must be rendered as a loop.




| Field | Type | Description |
| ----- | ---- | ----------- |
| direction |string| Loop direction: "n", "ne", "e", "se", "s", "sw", "w", "nw".   |
| radius |double| Loop radius in pixels.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Point"></a>

### Point

Point represents a 2D coordinate in the layout coordinate system.

COORDINATE SEMANTICS:
- Origin (0,0) at top-left of canvas
- X increases rightward
- Y increases downward
- Units are logical pixels




| Field | Type | Description |
| ----- | ---- | ----------- |
| x |double|   |
| y |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Dimensions"></a>

### Dimensions

Dimensions represents width and height.




| Field | Type | Description |
| ----- | ---- | ----------- |
| width |double|   |
| height |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-EdgeInsets"></a>

### EdgeInsets

EdgeInsets defines padding/margins on four sides.




| Field | Type | Description |
| ----- | ---- | ----------- |
| top |double|   |
| right |double|   |
| bottom |double|   |
| left |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Anchor"></a>

### Anchor

Anchor defines a named connection point on a state boundary.

ANCHOR POSITIONING:
Relative position uses normalized coordinates:
  - (0, 0) = top-left corner
  - (1, 0) = top-right corner
  - (0.5, 0.5) = center
  - (1, 1) = bottom-right corner

DIRECTION HINTS:
The direction field suggests preferred edge routing:
  - "n", "s", "e", "w": cardinal directions
  - "ne", "nw", "se", "sw": diagonals
  - "auto": let renderer decide




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Unique identifier for this anchor.   |
| relative_position |[Point](#extensions-v1-Point)| Position as fraction of state bounds. (0,0) = top-left, (1,1) = bottom-right.   |
| direction |string| Preferred edge direction: "n", "s", "e", "w", "ne", "nw", "se", "sw", "auto".   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-VisualStyle"></a>

### VisualStyle

VisualStyle provides portable styling hints.

COLOR FORMAT:
Colors use CSS color syntax for maximum portability:
  - Hex: "#RRGGBB", "#RGB", "#RRGGBBAA"
  - Named: "red", "blue", "transparent"
  - RGB: "rgb(255, 0, 0)", "rgba(255, 0, 0, 0.5)"

SEMANTICS:
These are HINTS. Renderers:
  - SHOULD interpret recognized properties
  - MAY apply their own defaults for unset properties
  - MAY ignore properties they don't support




| Field | Type | Description |
| ----- | ---- | ----------- |
| fill_color |string| Fill color (CSS color string).   |
| stroke_color |string| Stroke/border color.   |
| stroke_width |double| Stroke width in pixels.   |
| stroke_pattern |string| Stroke pattern: "solid", "dashed", "dotted".   |
| corner_radius |double| Corner radius for rounded rectangles.   |
| opacity |double| Overall opacity: 0.0 (transparent) to 1.0 (opaque).   |
| font_family |string| Font family for labels: "sans-serif", "monospace", "Arial", etc.   |
| font_size |double| Font size in points.   |
| font_weight |string| Font weight: "normal", "bold", or numeric "100"-"900".   |
| text_color |string| Text color for labels.   |
| shadow |[Shadow](#extensions-v1-Shadow)| Shadow configuration (optional).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Shadow"></a>

### Shadow

Shadow defines drop shadow styling.




| Field | Type | Description |
| ----- | ---- | ----------- |
| offset_x |double| Horizontal offset in pixels.   |
| offset_y |double| Vertical offset in pixels.   |
| blur |double| Blur radius in pixels.   |
| color |string| Shadow color (CSS color string).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-CanvasLayout"></a>

### CanvasLayout

CanvasLayout provides global canvas/diagram settings.

STORAGE:
Can be stored in Statechart-level metadata or as a dedicated extension.

USE CASES:
- Set diagram dimensions for export
- Configure snap-to-grid for editing
- Define default layout direction
- Establish spacing guidelines




| Field | Type | Description |
| ----- | ---- | ----------- |
| size |[Dimensions](#extensions-v1-Dimensions)| Canvas dimensions.   |
| background_color |string| Background color (CSS color string).   |
| grid_spacing |double| Grid spacing for snap-to-grid (0 = no grid).   |
| direction |string| Default layout direction: "TB" (top-bottom), "LR" (left-right), "BT" (bottom-top), "RL" (right-left).   |
| horizontal_spacing |double| Default horizontal spacing between sibling states.   |
| vertical_spacing |double| Default vertical spacing between sibling states.   |
| group_padding |double| Padding around state groups/clusters.   |
| viewport_position |[Point](#extensions-v1-Point)| Viewport/camera position (for editors).   |
| zoom |double| Zoom level: 1.0 = 100%, 2.0 = 200%.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->
 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

