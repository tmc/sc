import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

public struct FlowView: View {
    @Binding var nodes: [FlowNode]
    @Binding var edges: [FlowEdge]
    @Binding var activeStateIDs: Set<UUID>
    @Binding var scale: CGFloat
    @Binding var offset: CGSize
    @Binding var selection: Set<UUID>
    @Binding var selectionRect: CGRect?
    
    // Access Extension Manager
    @State private var manager = ExtensionManager.shared
    
    @State var currentDragOffset: CGSize = .zero
    
    // Dragging Logic
    private enum DragMode: Equatable {
        case idle, pan, node(UUID), marquee
    }
    @State private var dragMode: DragMode = .idle
    @State private var dragStartNodePositions: [UUID: CGPoint] = [:]
    @State private var initialScale: CGFloat? = nil // For robust zooming
    
    // Reparenting Logic
    @State var hoveredParentID: UUID? = nil
    @State var hoveredNodeID: UUID? = nil
    
    // Context Menu Logic
    @State private var lastContextLocation: CGPoint? = nil
    
    // Space Panning (macOS)
    @State private var isSpacePressed = false
    @State private var eventMonitor: Any? = nil
    
    // Connection Logic
    @State var connectingEdge: (source: UUID, currentPoint: CGPoint)? = nil
    
    // Callbacks
    var onNodeMoveEnded: (([UUID: CGPoint]) -> Void)?
    var onNodeDelete: ((UUID) -> Void)?
    var onNodeDuplicate: ((UUID) -> Void)?
    var onAddTransitionFrom: ((UUID) -> Void)?

    @State private var editingNodeID: UUID? = nil
    @State private var editingText: String = ""
    @FocusState private var isEditingFocus: Bool
    
    // Additional Callbacks
    var onNodeRename: ((UUID, String) -> Void)?
    var onAddStateAt: ((CGPoint) -> Void)?
    var onAddSubstate: ((UUID) -> Void)?
    var onNodeReparent: ((UUID, UUID?) -> Void)? // childID, newParentID

    // MARK: - Drawing Accessors (read-only)
    var drawing_currentDragOffset: CGSize { currentDragOffset }
    var drawing_hoveredNodeID: UUID? { hoveredNodeID }
    var drawing_hoveredParentID: UUID? { hoveredParentID }
    var drawing_connectingEdge: (source: UUID, currentPoint: CGPoint)? { connectingEdge }
    
    public init(nodes: Binding<[FlowNode]>,
                edges: Binding<[FlowEdge]>,
                activeStateIDs: Binding<Set<UUID>>,
                scale: Binding<CGFloat>,
                offset: Binding<CGSize>,
                selection: Binding<Set<UUID>>,
                selectionRect: Binding<CGRect?> = .constant(nil),
                onNodeMoveEnded: (([UUID: CGPoint]) -> Void)? = nil,
                onNodeDelete: ((UUID) -> Void)? = nil,
                onNodeDuplicate: ((UUID) -> Void)? = nil,
                onAddTransitionFrom: ((UUID) -> Void)? = nil,
                onNodeRename: ((UUID, String) -> Void)? = nil,
                onAddStateAt: ((CGPoint) -> Void)? = nil,
                onAddSubstate: ((UUID) -> Void)? = nil,
                onNodeReparent: ((UUID, UUID?) -> Void)? = nil) {
        _nodes = nodes
        _edges = edges
        _activeStateIDs = activeStateIDs
        _scale = scale
        _offset = offset
        _selection = selection
        _selectionRect = selectionRect
        self.onNodeMoveEnded = onNodeMoveEnded
        self.onNodeDelete = onNodeDelete
        self.onNodeDuplicate = onNodeDuplicate
        self.onAddTransitionFrom = onAddTransitionFrom
        self.onNodeRename = onNodeRename
        self.onAddStateAt = onAddStateAt
        self.onAddSubstate = onAddSubstate
        self.onNodeReparent = onNodeReparent
    }
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background with Parallax
                FlowBackground(scale: scale, offset: (offset + currentDragOffset).parallax())
                    .onTapGesture {
                        // Clear selection/editing on background tap
                        editingNodeID = nil
                        selection = []
                    }
                
                // Canvas
                TimelineView(.animation) { timeline in
                    ZStack {
                        // 1. Edges Layer
                        FlowEdgesLayer(
                            edges: $edges,
                            nodes: nodes,
                            selection: $selection,
                            scale: $scale,
                            offset: $offset,
                            currentDragOffset: drawing_currentDragOffset,
                            connectingEdge: drawing_connectingEdge
                        )
                        
                        // 2. Nodes Layer
                        FlowNodesLayer(
                            nodes: $nodes,
                            selection: $selection,
                            activeStateIDs: $activeStateIDs,
                            hoveredNodeID: drawing_hoveredNodeID,
                            hoveredParentID: drawing_hoveredParentID,
                            scale: $scale,
                            offset: $offset,
                            currentDragOffset: drawing_currentDragOffset,
                            selectionRect: selectionRect,
                            phase: timeline.date.timeIntervalSinceReferenceDate
                        )
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .local)
                        .onChanged { value in
                            handleDragChanged(location: value.location, translation: value.translation, startLocation: value.startLocation, in: geometry.size)
                        }
                        .onEnded { value in
                            handleDragEnded(location: value.location, translation: value.translation, startLocation: value.startLocation, in: geometry.size)
                        }
                )
                .onTapGesture(count: 2) {
                    // Smart Zoom Logic
                    let targetScale: CGFloat
                    let targetOffset: CGSize
                    
                    if nodes.isEmpty {
                        targetScale = 1.0
                        targetOffset = .zero
                    } else {
                        let minX = nodes.map { $0.position.x }.min() ?? 0
                        let maxX = nodes.map { $0.position.x + $0.size.width }.max() ?? 0
                        let minY = nodes.map { $0.position.y }.min() ?? 0
                        let maxY = nodes.map { $0.position.y + $0.size.height }.max() ?? 0
                        
                        let contentWidth = maxX - minX
                        let contentHeight = maxY - minY
                        let contentCenter = CGPoint(x: minX + contentWidth / 2, y: minY + contentHeight / 2)
                        
                        let viewSize = geometry.size
                        let padding: CGFloat = 100
                        
                        let scaleX = (viewSize.width - padding) / contentWidth
                        let scaleY = (viewSize.height - padding) / contentHeight
                        let fitScale = min(scaleX, scaleY, 2.0)
                        
                        if abs(scale - fitScale) < 0.1 {
                            targetScale = 1.0
                            targetOffset = CGSize(width: -contentCenter.x * 1.0, height: -contentCenter.y * 1.0)
                        } else {
                            targetScale = fitScale
                            targetOffset = CGSize(width: -contentCenter.x * fitScale, height: -contentCenter.y * fitScale)
                        }
                    }
                    
                    withAnimation(.spring(.snappy(duration: 0.4))) {
                        scale = targetScale
                        offset = targetOffset
                    }
                    Theme.Haptics.play(.medium)
                }
                .onTapGesture(count: 1) {
                    // Clear selection/editing on background tap
                    editingNodeID = nil
                    selection = []
                }

                // Single tap handled by overlay primarily, leaving this for drag disambiguation?
                // Actually remove general tap gesture here to let overlay handle specific taps
                
                // Interaction Overlay (Refactored)
                FlowInteractionOverlay(
                    size: geometry.size,
                    offset: offset,
                    scale: scale,
                    currentDragOffset: currentDragOffset,
                    nodes: $nodes,
                    editingNodeID: $editingNodeID,
                    editingText: $editingText,
                    onNodeRename: onNodeRename,
                    onAddSubstate: onAddSubstate,
                    onNodeDuplicate: onNodeDuplicate,
                    onNodeDelete: onNodeDelete,
                    onHitTest: { location in
                         hitTest(location: location, in: geometry.size)
                    },
                    onDragChanged: { value in
                        handleDragChanged(location: value.location, translation: value.translation, startLocation: value.startLocation, in: geometry.size)
                    },
                    onDragEnded: { value in
                        handleDragEnded(location: value.location, translation: value.translation, startLocation: value.startLocation, in: geometry.size)
                    }
                )
                
                // Custom Extension Rendering Overlay
                extensionOverlay(in: geometry)
                
                // Top Layer: Input Handling (Scroll & Double-Tap)
                // This sits on top to capture scroll/zoom via monitor (pass-through clicks)
                // And captures double-taps specifically.
                // And captures double-taps specifically.
                ScrollEventView(offset: $offset, scale: $scale, nodes: $nodes)
                   .allowsHitTesting(false) // The NSView manually handles events via monitor, so View hit test false is fine? 
                                            // Actually, the NSView must be in window hierarchy. If allowsHitTesting false, SwiftUI might remove it from hierarchy or nsview.isHidden = true?
                                            // Safe bet: allowsHitTesting(true), but NSView.hitTest returns nil.
                
                // Double Tap for adding state (Must be on top to not be blocked by Canvas)
                // Removed Color.black blocker - taps handled by FlowBackground
            }
            .contentShape(Rectangle()) // Ensure ZStack is hit-testable for drags passing through
            // Removed .simultaneousGesture(SpatialTapGesture(count: 2)) as it's now on background
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                     lastContextLocation = location
                     // Hit test for cursor and hover effect
                     let hitID = hitTestNode(at: location, in: geometry.size)
                     if hitID != hoveredNodeID {
                         hoveredNodeID = hitID
                     }
                     #if os(macOS)
                     cursorForCurrentState().set()
                     #endif
                case .ended:
                     hoveredNodeID = nil
                     lastContextLocation = nil
                     #if os(macOS)
                     NSCursor.arrow.set()
                     #endif
                }
            }
            .contextMenu {
                // ...
                Button {
                    let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
                    let tapLocation = lastContextLocation ?? center
                    
                    let chartPoint = CGPoint(
                        x: (tapLocation.x - center.x - offset.width) / scale,
                        y: (tapLocation.y - center.y - offset.height) / scale
                    )
                    onAddStateAt?(chartPoint)
                } label: {
                    Label("Add State Here", systemImage: "plus.square")
                }
            }
        }
        .onAppear {
            #if os(macOS)
            eventMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { event in
                if event.keyCode == 49 { // Space
                    if event.type == .keyDown && !event.isARepeat {
                        isSpacePressed = true
                    } else if event.type == .keyUp {
                        isSpacePressed = false
                    }
                    return event
                }
                return event
            }
            #endif
        }
        .onDisappear {
            #if os(macOS)
            if let monitor = eventMonitor {
                NSEvent.removeMonitor(monitor)
                eventMonitor = nil
            }
            #endif
        }
    }
    
    // MARK: - Event Handling (Refactored for clarity)
    
    private func handleDragChanged(location: CGPoint, translation: CGSize, startLocation: CGPoint, in size: CGSize) {
        if dragMode == .idle && connectingEdge == nil {
            // Space+Drag forces Pan Mode
            if isSpacePressed {
                dragMode = .pan
            } else if let nodeID = hitTestNode(at: startLocation, in: size) {
                // Connection Trigger Logic
                if let node = nodes.first(where: { $0.id == nodeID }) {
                    let center = CGPoint(x: size.width / 2, y: size.height / 2)
                    let chartPoint = CGPoint(
                        x: (startLocation.x - center.x - offset.width) / scale,
                        y: (startLocation.y - center.y - offset.height) / scale
                    )
                    
                    // Check Ports (Priority if selected)
                    if selection.contains(node.id) {
                        let nodeRect = CGRect(origin: node.position, size: node.size)
                        let ports = [
                            CGPoint(x: nodeRect.midX, y: nodeRect.minY), // Top
                            CGPoint(x: nodeRect.maxX, y: nodeRect.midY), // Right
                            CGPoint(x: nodeRect.midX, y: nodeRect.maxY), // Bottom
                            CGPoint(x: nodeRect.minX, y: nodeRect.midY)  // Left
                        ]
                        
                        // Hit test ports with generous radius
                        if ports.contains(where: { distance($0, chartPoint) < 20 }) {
                            connectingEdge = (source: nodeID, currentPoint: chartPoint)
                            return
                        }
                    }
                    
                    // Fallback: Bottom Area (Legacy/Easy Access)
                    let bottomRect = CGRect(x: node.position.x, y: node.position.y + node.size.height - 15, width: node.size.width, height: 15)
                    if bottomRect.contains(chartPoint) {
                        connectingEdge = (source: nodeID, currentPoint: chartPoint)
                        return
                    }
                }
                
                dragMode = .node(nodeID)
                if !selection.contains(nodeID) { 
                    selection = [nodeID] 
                    Theme.Haptics.selection()
                }
                Theme.Haptics.play(.soft) // Soft pick-up feedback
                dragStartNodePositions = nodes.reduce(into: [:]) { dict, node in
                    if selection.contains(node.id) { dict[node.id] = node.position }
                }
            } else {
                // Background Drag Logic
                if isSpacePressed {
                    dragMode = .pan
                } else {
                    var shiftHeld = false
                    #if os(macOS)
                    shiftHeld = NSEvent.modifierFlags.contains(.shift)
                    #endif
                    
                    if !selection.isEmpty && !shiftHeld {
                       // Click on background clears selection unless Shift held
                       // Actually tap handles clear, drag might not?
                       // Standard: Background click = clear. Background Drag = marquee.
                       dragMode = .marquee
                    } else {
                       dragMode = .marquee
                    }
                }
                
                if dragMode == .marquee {
                     // Start Marquee
                     // let center = CGPoint(x: size.width / 2, y: size.height / 2)
                     /*
                     let chartPoint = CGPoint(
                         x: (startLocation.x - center.x - offset.width) / scale,
                         y: (startLocation.y - center.y - offset.height) / scale
                     )
                     */
                     // Using dragStartNodePositions to store Start Point? No, separate var.
                     // Re-use currentDragOffset for translation? Yes.
                     
                     // Store initial selection for Shift-Select behavior
                     // We don't have separate state for initialSelection, but we can assume selection at start is it.
                     // But we didn't add the state variable yet.
                     // We'll perform additive logic dynamically or just Replace if no shift?
                }
            }
        }
        
        if let connecting = connectingEdge {
            // ... (Connector Logic)
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let currentChartPoint = CGPoint(
                 x: (location.x - center.x - offset.width) / scale,
                 y: (location.y - center.y - offset.height) / scale
            )// ...
            // ...
            connectingEdge = (source: connecting.source, currentPoint: currentChartPoint)
        } else {
            switch dragMode {
            case .pan:
                currentDragOffset = translation
            case .marquee:
                // Calculate Marquee Rect
                // Start: startLocation
                // Current: location
                // Convert both to Logic Space
                
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                let startPoint = CGPoint(
                    x: (startLocation.x - center.x - offset.width) / scale,
                    y: (startLocation.y - center.y - offset.height) / scale
                )
                let endPoint = CGPoint(
                    x: (location.x - center.x - offset.width) / scale,
                    y: (location.y - center.y - offset.height) / scale
                )
                
                let rect = CGRect(x: min(startPoint.x, endPoint.x),
                                  y: min(startPoint.y, endPoint.y),
                                  width: abs(endPoint.x - startPoint.x),
                                  height: abs(endPoint.y - startPoint.y))
                
                self.selectionRect = rect
                
                // Do Filtering
                let hitNodes = nodes.filter { node in
                    let nodeRect = CGRect(origin: node.position, size: node.size)
                    return rect.intersects(nodeRect)
                }
                
                let hitIDs = Set(hitNodes.map { $0.id })
                
                // Additive Logic (Shift)
                /* 
                 For simplicity here:
                 If Shift held (checked via NSEvent or assumption), Union.
                 Else Replace.
                 Ideally we captured 'initialSelection' at dragStart.
                 Since we lack that state var in this edit block, we'll do simple replace or union logic.
                 Wait, I can access 'selection' but if I replace it, I lose context.
                 I need to add `initialSelection` state var to FlowView class first or use a closure.
                 I'll default to Replace for now as MVP Marquee.
                */
                selection = hitIDs
                
            case .node:
                // ... (Node Drag Logic)
                let scale = self.scale
                // ...
                let snapGrid: CGFloat = 20.0
                
                // Reparenting Check
                // Assuming we are dragging 1 node for now (or taking the first)
                if let draggedID = dragStartNodePositions.keys.first,
                   let index = nodes.firstIndex(where: { $0.id == draggedID }) {
                    
                    let draggedNode = nodes[index]
                    let draggedRect = CGRect(origin: draggedNode.position, size: draggedNode.size)
                    let center = CGPoint(x: draggedRect.midX, y: draggedRect.midY)
                    
                    // Simple Hit Test for Parent
                    // Must generally contain the center of the dragged node
                    // And not be the node itself, or a descendant (prevent cycles - tricky without graph traversal, but id check is basic)
                    hoveredParentID = nodes.last(where: { candidate in
                        if candidate.id == draggedID { return false }
                        // Parent must be compound or machine? For Statechart, basically any node can be a parent in some formalisms, 
                        // but usually "atomic" implies no children. Let's assume .compound or just check containment for now.
                        // Ideally, we check type == .compound || .parallel || .machine
                        // For now, allow any non-atomic? Or basically just geometry check + type check.
                        if candidate.type == .atomic || candidate.type == .history || candidate.type == .final { return false }
                        
                        let rect = CGRect(origin: candidate.position, size: candidate.size)
                        return rect.contains(center)
                    })?.id
                }
                
                for (id, initialPos) in dragStartNodePositions {
                    if let index = nodes.firstIndex(where: { $0.id == id }) {
                        let rawX = initialPos.x + translation.width / scale
                        let rawY = initialPos.y + translation.height / scale
                        
                        let snappedX = round(rawX / snapGrid) * snapGrid
                        let snappedY = round(rawY / snapGrid) * snapGrid
                        
                        let newPos = CGPoint(x: snappedX, y: snappedY)
                        if newPos != nodes[index].position {
                             Theme.Haptics.play(Theme.Haptics.FeedbackStyle.light) // Snap feedback
                        }
                        nodes[index].position = newPos
                    }
                }
            case .idle: break
            }
        }
    }
    
    private func handleDragEnded(location: CGPoint, translation: CGSize, startLocation: CGPoint, in size: CGSize) {
        if let connecting = connectingEdge {
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let dropPoint = CGPoint(
                x: (location.x - center.x - offset.width) / scale,
                y: (location.y - center.y - offset.height) / scale
            )
            
            if let targetNode = nodes.last(where: { node in
                let rect = CGRect(origin: node.position, size: node.size)
                return rect.contains(dropPoint)
            }), targetNode.id != connecting.source {
                let newEdge = FlowEdge(source: connecting.source, target: targetNode.id)
                edges.append(newEdge)
                Theme.Haptics.notification(Theme.Haptics.NotificationType.success) // Connection success
            }
            connectingEdge = nil
        } else {
            switch dragMode {
            case .pan:
                offset.width += translation.width
                offset.height += translation.height
                currentDragOffset = .zero
            case .marquee:
                self.selectionRect = nil
            case .node:
                // Check if we need to reparent or unnest
                if let draggedID = dragStartNodePositions.keys.first,
                   let index = nodes.firstIndex(where: { $0.id == draggedID }) {
                    
                    let currentParentID = nodes[index].parentID
                    // If hoveredParentID is nil, we might be unnesting (dropping on root)
                    // If hoveredParentID is set, we are nesting
                    
                        if hoveredParentID != currentParentID {
                        onNodeReparent?(draggedID, hoveredParentID)
                    }
                }
                Theme.Haptics.play(.rigid) // Snappy drop feedback
                
                onNodeMoveEnded?(dragStartNodePositions) // Register undo (position change still happened)
                dragStartNodePositions.removeAll()
                hoveredParentID = nil // Reset hover
            case .idle: break
            }
            dragMode = .idle
        }
    }
    
    // MARK: - Cursor Logic
    
    #if canImport(AppKit)
    private func cursorForCurrentState() -> NSCursor {
        if isSpacePressed { return .openHand }
        if connectingEdge != nil { return .crosshair }
        switch dragMode {
        case .pan: return .closedHand
        case .node: return .closedHand
        case .marquee: return .arrow
        case .idle:
            if hoveredNodeID != nil { return .pointingHand }
            return .arrow
        }
    }
    #endif

    // Drawing logic moved to FlowView+Drawing.swift
    
    // MARK: - Extension Overlay
    
    @ViewBuilder
    private func extensionOverlay(in geo: GeometryProxy) -> some View {
        if let renderer = manager.activeRenderer {
            ForEach(nodes) { node in
                let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                let viewX = center.x + offset.width + currentDragOffset.width + node.position.x * scale
                let viewY = center.y + offset.height + currentDragOffset.height + node.position.y * scale
                
                if viewX > -200 && viewX < geo.size.width + 200 &&
                   viewY > -200 && viewY < geo.size.height + 200 {
                    
                    renderer.render(state: StateData(
                        id: node.id.uuidString,
                        label: node.label,
                        type: "\(node.type)",
                        meta: [:]
                    ))
                    .position(x: viewX + (node.size.width * scale / 2), y: viewY + (node.size.height * scale / 2))
                    .scaleEffect(scale)
                    .allowsHitTesting(false)
                }
            }
        }
    }
    
    // MARK: - Hit Testing
    
    private func hitTest(location: CGPoint, in size: CGSize) {
        if let nodeID = hitTestNode(at: location, in: size) {
            selection = [nodeID]
        } else if let edgeID = hitTestEdge(at: location, in: size) {
            selection = [edgeID]
        } else {
            selection = []
        }
    }
    
    private func hitTestNode(at location: CGPoint, in size: CGSize) -> UUID? {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let chartPoint = CGPoint(
            x: (location.x - center.x - offset.width) / scale,
            y: (location.y - center.y - offset.height) / scale
        )
        
        return nodes.last(where: { node in
            CGRect(origin: node.position, size: node.size).contains(chartPoint)
        })?.id
    }
    
    private func hitTestEdge(at location: CGPoint, in size: CGSize) -> UUID? {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let chartPoint = CGPoint(
            x: (location.x - center.x - offset.width) / scale,
            y: (location.y - center.y - offset.height) / scale
        )
        // Increased threshold for easier tapping
        let threshold: CGFloat = 20.0 / scale
        
        // Check selection first to prioritize editing active selection if overlapping? 
        // Or reverse order to hit top-most? Edges are drawn in order.
        // We'll search in reverse to hit the "top" edge if they overlap.
        return edges.reversed().first(where: { edge in
            guard let source = nodes.first(where: { $0.id == edge.source }),
                  let target = nodes.first(where: { $0.id == edge.target }) else { return false }
            
            let sRect = CGRect(origin: source.position, size: source.size)
            let tRect = CGRect(origin: target.position, size: target.size)
            let (start, end) = FlowRenderer.calculateConnectionPoints(from: sRect, to: tRect)
            
            // 1. Reflexive (Self-loop) logic matching drawEdge
            if source.id == target.id {
                let control1 = CGPoint(x: start.x + 50, y: start.y - 50)
                let control2 = CGPoint(x: start.x - 50, y: start.y - 50)
                
                // Sample Bezier curve for hit testing (10 steps)
                var prevPoint = start
                for i in 1...10 {
                    let t = CGFloat(i) / 10.0
                    // Cubic Bezier formula
                    let u = 1 - t
                    let tt = t * t
                    let uu = u * u
                    let uuu = uu * u
                    let ttt = tt * t
                    
                    // Breakdown for compiler speed
                    let termX1 = uuu * start.x
                    let termX2 = 3 * uu * t * control1.x
                    let termX3 = 3 * u * tt * control2.x
                    let termX4 = ttt * end.x
                    
                    let termY1 = uuu * start.y
                    let termY2 = 3 * uu * t * control1.y
                    let termY3 = 3 * u * tt * control2.y
                    let termY4 = ttt * end.y
                    
                    let p = CGPoint(
                        x: termX1 + termX2 + termX3 + termX4,
                        y: termY1 + termY2 + termY3 + termY4
                    )
                    
                    if distance(from: chartPoint, toLineSegment: (prevPoint, p)) < threshold {
                        return true
                    }
                    prevPoint = p
                }
                return false
            }
            
            // 2. Orthogonal (Standard)
            // Note: This must match drawEdge logic exactly
            if let waypoints = edge.waypoints, !waypoints.isEmpty {
                // Custom Waypoints
                var prev = start
                for point in waypoints {
                    if distance(from: chartPoint, toLineSegment: (prev, point)) < threshold { return true }
                    prev = point
                }
                if distance(from: chartPoint, toLineSegment: (prev, end)) < threshold { return true }
                return false
            } else {
                // Auto Orthogonal
                let midY = (start.y + end.y) / 2
                let p1 = CGPoint(x: start.x, y: midY)
                let p2 = CGPoint(x: end.x, y: midY)
                
                let segments = [(start, p1), (p1, p2), (p2, end)]
                for segment in segments {
                    if distance(from: chartPoint, toLineSegment: segment) < threshold { return true }
                }
                return false
            }
        })?.id
    }
    
    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2))
    }

    private func distance(from p: CGPoint, toLineSegment segment: (CGPoint, CGPoint)) -> CGFloat {
        let v = segment.0
        let w = segment.1
        let l2 = (v.x - w.x) * (v.x - w.x) + (v.y - w.y) * (v.y - w.y)
        if l2 == 0 { return sqrt((p.x - v.x) * (p.x - v.x) + (p.y - v.y) * (p.y - v.y)) }
        var t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2
        t = max(0, min(1, t))
        let proj = CGPoint(x: v.x + t * (w.x - v.x), y: v.y + t * (w.y - v.y))
        return sqrt((p.x - proj.x) * (p.x - proj.x) + (p.y - proj.y) * (p.y - proj.y))
    }
}

// MARK: - Flow Background


// Helper to add CGSize
extension CGSize {
    static func +(lhs: CGSize, rhs: CGSize) -> CGSize {
        CGSize(width: lhs.width + rhs.width, height: lhs.height + rhs.height)
    }
    static func +=(lhs: inout CGSize, rhs: CGSize) {
        lhs = lhs + rhs
    }
    
    func parallax(factor: CGFloat = 0.5) -> CGSize {
        CGSize(width: width * factor, height: height * factor)
    }
}

// MARK: - Previews
#Preview("Flow View") {
    struct PreviewWrapper: View {
        @State var nodes: [FlowNode] = [
            FlowNode(id: UUID(), position: CGPoint(x: 100, y: 100), label: "Start", type: .atomic),
            FlowNode(id: UUID(), position: CGPoint(x: 400, y: 100), label: "Process", type: .compound),
        ]
        @State var edges: [FlowEdge] = []
        @State var scale: CGFloat = 1.0
        @State var offset: CGSize = .zero
        @State var selection: Set<UUID> = []
        @State var activeStateIDs: Set<UUID> = []
        
        var body: some View {
            FlowView(nodes: $nodes, edges: $edges, activeStateIDs: $activeStateIDs, scale: $scale, offset: $offset, selection: $selection)
        }
    }
    return PreviewWrapper()
}

