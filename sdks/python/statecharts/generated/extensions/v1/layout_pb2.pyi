from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StateLayout(_message.Message):
    __slots__ = ()
    POSITION_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    STYLE_FIELD_NUMBER: _ClassVar[int]
    Z_INDEX_FIELD_NUMBER: _ClassVar[int]
    ANCHORS_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_HINT_FIELD_NUMBER: _ClassVar[int]
    POSITION_LOCKED_FIELD_NUMBER: _ClassVar[int]
    MIN_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_SIZE_FIELD_NUMBER: _ClassVar[int]
    PADDING_FIELD_NUMBER: _ClassVar[int]
    COLLAPSED_FIELD_NUMBER: _ClassVar[int]
    position: Point
    size: Dimensions
    style: VisualStyle
    z_index: int
    anchors: _containers.RepeatedCompositeFieldContainer[Anchor]
    layout_hint: str
    position_locked: bool
    min_size: Dimensions
    max_size: Dimensions
    padding: EdgeInsets
    collapsed: bool
    def __init__(self, position: _Optional[_Union[Point, _Mapping]] = ..., size: _Optional[_Union[Dimensions, _Mapping]] = ..., style: _Optional[_Union[VisualStyle, _Mapping]] = ..., z_index: _Optional[int] = ..., anchors: _Optional[_Iterable[_Union[Anchor, _Mapping]]] = ..., layout_hint: _Optional[str] = ..., position_locked: _Optional[bool] = ..., min_size: _Optional[_Union[Dimensions, _Mapping]] = ..., max_size: _Optional[_Union[Dimensions, _Mapping]] = ..., padding: _Optional[_Union[EdgeInsets, _Mapping]] = ..., collapsed: _Optional[bool] = ...) -> None: ...

class TransitionLayout(_message.Message):
    __slots__ = ()
    WAYPOINTS_FIELD_NUMBER: _ClassVar[int]
    EDGE_STYLE_FIELD_NUMBER: _ClassVar[int]
    ARROW_STYLE_FIELD_NUMBER: _ClassVar[int]
    LABEL_POSITION_FIELD_NUMBER: _ClassVar[int]
    LABEL_OFFSET_FIELD_NUMBER: _ClassVar[int]
    STYLE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ANCHOR_FIELD_NUMBER: _ClassVar[int]
    TARGET_ANCHOR_FIELD_NUMBER: _ClassVar[int]
    SELF_LOOP_FIELD_NUMBER: _ClassVar[int]
    CURVATURE_FIELD_NUMBER: _ClassVar[int]
    waypoints: _containers.RepeatedCompositeFieldContainer[Point]
    edge_style: str
    arrow_style: str
    label_position: float
    label_offset: float
    style: VisualStyle
    source_anchor: int
    target_anchor: int
    self_loop: SelfLoopLayout
    curvature: float
    def __init__(self, waypoints: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., edge_style: _Optional[str] = ..., arrow_style: _Optional[str] = ..., label_position: _Optional[float] = ..., label_offset: _Optional[float] = ..., style: _Optional[_Union[VisualStyle, _Mapping]] = ..., source_anchor: _Optional[int] = ..., target_anchor: _Optional[int] = ..., self_loop: _Optional[_Union[SelfLoopLayout, _Mapping]] = ..., curvature: _Optional[float] = ...) -> None: ...

class SelfLoopLayout(_message.Message):
    __slots__ = ()
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    RADIUS_FIELD_NUMBER: _ClassVar[int]
    direction: str
    radius: float
    def __init__(self, direction: _Optional[str] = ..., radius: _Optional[float] = ...) -> None: ...

class Point(_message.Message):
    __slots__ = ()
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Dimensions(_message.Message):
    __slots__ = ()
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    width: float
    height: float
    def __init__(self, width: _Optional[float] = ..., height: _Optional[float] = ...) -> None: ...

class EdgeInsets(_message.Message):
    __slots__ = ()
    TOP_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    top: float
    right: float
    bottom: float
    left: float
    def __init__(self, top: _Optional[float] = ..., right: _Optional[float] = ..., bottom: _Optional[float] = ..., left: _Optional[float] = ...) -> None: ...

class Anchor(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_POSITION_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    id: str
    relative_position: Point
    direction: str
    def __init__(self, id: _Optional[str] = ..., relative_position: _Optional[_Union[Point, _Mapping]] = ..., direction: _Optional[str] = ...) -> None: ...

class VisualStyle(_message.Message):
    __slots__ = ()
    FILL_COLOR_FIELD_NUMBER: _ClassVar[int]
    STROKE_COLOR_FIELD_NUMBER: _ClassVar[int]
    STROKE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    STROKE_PATTERN_FIELD_NUMBER: _ClassVar[int]
    CORNER_RADIUS_FIELD_NUMBER: _ClassVar[int]
    OPACITY_FIELD_NUMBER: _ClassVar[int]
    FONT_FAMILY_FIELD_NUMBER: _ClassVar[int]
    FONT_SIZE_FIELD_NUMBER: _ClassVar[int]
    FONT_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    TEXT_COLOR_FIELD_NUMBER: _ClassVar[int]
    SHADOW_FIELD_NUMBER: _ClassVar[int]
    fill_color: str
    stroke_color: str
    stroke_width: float
    stroke_pattern: str
    corner_radius: float
    opacity: float
    font_family: str
    font_size: float
    font_weight: str
    text_color: str
    shadow: Shadow
    def __init__(self, fill_color: _Optional[str] = ..., stroke_color: _Optional[str] = ..., stroke_width: _Optional[float] = ..., stroke_pattern: _Optional[str] = ..., corner_radius: _Optional[float] = ..., opacity: _Optional[float] = ..., font_family: _Optional[str] = ..., font_size: _Optional[float] = ..., font_weight: _Optional[str] = ..., text_color: _Optional[str] = ..., shadow: _Optional[_Union[Shadow, _Mapping]] = ...) -> None: ...

class Shadow(_message.Message):
    __slots__ = ()
    OFFSET_X_FIELD_NUMBER: _ClassVar[int]
    OFFSET_Y_FIELD_NUMBER: _ClassVar[int]
    BLUR_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    offset_x: float
    offset_y: float
    blur: float
    color: str
    def __init__(self, offset_x: _Optional[float] = ..., offset_y: _Optional[float] = ..., blur: _Optional[float] = ..., color: _Optional[str] = ...) -> None: ...

class CanvasLayout(_message.Message):
    __slots__ = ()
    SIZE_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_COLOR_FIELD_NUMBER: _ClassVar[int]
    GRID_SPACING_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_SPACING_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_SPACING_FIELD_NUMBER: _ClassVar[int]
    GROUP_PADDING_FIELD_NUMBER: _ClassVar[int]
    VIEWPORT_POSITION_FIELD_NUMBER: _ClassVar[int]
    ZOOM_FIELD_NUMBER: _ClassVar[int]
    size: Dimensions
    background_color: str
    grid_spacing: float
    direction: str
    horizontal_spacing: float
    vertical_spacing: float
    group_padding: float
    viewport_position: Point
    zoom: float
    def __init__(self, size: _Optional[_Union[Dimensions, _Mapping]] = ..., background_color: _Optional[str] = ..., grid_spacing: _Optional[float] = ..., direction: _Optional[str] = ..., horizontal_spacing: _Optional[float] = ..., vertical_spacing: _Optional[float] = ..., group_padding: _Optional[float] = ..., viewport_position: _Optional[_Union[Point, _Mapping]] = ..., zoom: _Optional[float] = ...) -> None: ...
