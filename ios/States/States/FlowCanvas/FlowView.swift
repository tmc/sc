import SwiftUI
import UIKit


public struct FlowView: View {
    @Binding var nodes: [FlowNode]
    @Binding var edges: [FlowEdge]
    @Binding var activeStateIDs: Set<UUID>
    @Binding var scale: CGFloat
    @Binding var offset: CGSize
    @Binding var selection: Set<UUID>
    
    // Access Extension Manager
    @State private var manager = ExtensionManager.shared
    
    @State private var currentDragOffset: CGSize = .zero
    
    // Dragging Logic
    private enum DragMode: Equatable {
        case idle, pan, node(UUID)
    }
    @State private var dragMode: DragMode = .idle
    @State private var dragStartNodePositions: [UUID: CGPoint] = [:]
    @State private var initialScale: CGFloat? = nil // For robust zooming
    
    // Connection Logic
    @State private var connectingEdge: (source: UUID, currentPoint: CGPoint)? = nil
    
    // Callbacks
    var onNodeMoveEnded: (([UUID: CGPoint]) -> Void)?
    
    public init(nodes: Binding<[FlowNode]>, 
                edges: Binding<[FlowEdge]>, 
                activeStateIDs: Binding<Set<UUID>>,
                scale: Binding<CGFloat>, 
                offset: Binding<CGSize>, 
                selection: Binding<Set<UUID>>,
                onNodeMoveEnded: (([UUID: CGPoint]) -> Void)? = nil) {
        _nodes = nodes
        _edges = edges
        _activeStateIDs = activeStateIDs
        _scale = scale
        _offset = offset
        _selection = selection
        self.onNodeMoveEnded = onNodeMoveEnded
    }
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                FlowBackground(scale: scale, offset: offset + currentDragOffset)
                
                // Canvas
                Canvas { context, size in
                    let totalOffset = offset + currentDragOffset
                    context.translateBy(x: size.width / 2 + totalOffset.width,
                                      y: size.height / 2 + totalOffset.height)
                    context.scaleBy(x: scale, y: scale)
                    
                    // Draw Edges
                    for edge in edges {
                        if let sourceNode = nodes.first(where: { $0.id == edge.source }),
                           let targetNode = nodes.first(where: { $0.id == edge.target }) {
                            drawEdge(context: context, source: sourceNode, target: targetNode, edge: edge)
                        }
                    }
                    
                    // Draw Connection Dragging
                    if let connecting = connectingEdge,
                       let sourceNode = nodes.first(where: { $0.id == connecting.source }) {
                        drawConnectionDrag(context: context, source: sourceNode, endPoint: connecting.currentPoint)
                    }
                    
                    // Draw Nodes (Sorted by hierarchy depth or simply ensuring parents draw first)
                    // Simple heuristic: Nodes with nil parent first, then their children.
                    // Or simple sort: if A.parent == B, B must be drawn before A.
                    // Topological sort is ideal, but for now let's just sort by "has parent".
                    // Or assumes nodes are in order? No.
                    // Let's perform a simple stable sort: those with nil parent, then keypath depth.
                    let sortedNodes = nodes.sorted { (a, b) -> Bool in
                        // If a is b's parent, a comes first.
                        if a.id == b.parentID { return true }
                        if b.id == a.parentID { return false }
                        // Otherwise, nil parents first
                        if a.parentID == nil && b.parentID != nil { return true }
                        if a.parentID != nil && b.parentID == nil { return false }
                        return false
                    }
                    
                    for node in sortedNodes {
                        drawNode(context: context, node: node)
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .local)
                        .onChanged { value in
                            if dragMode == .idle && connectingEdge == nil {
                                // Determine mode on first change
                                if let nodeID = hitTestNode(at: value.startLocation, in: geometry.size) {
                                    // Check if drag started at bottom of node to trigger connection
                                    // Need node frame
                                    if let node = nodes.first(where: { $0.id == nodeID }) {
                                        // Coordinate transform for hit test is complex here because 'value.startLocation' is local to view?
                                        // Actually hitTestNode handled the transform.
                                        
                                        // Let's implement: If we start dragging from the BOTTOM EDGE (last 15 points of height).
                                        let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
                                        let chartPoint = CGPoint(
                                            x: (value.startLocation.x - center.x - offset.width) / scale,
                                            y: (value.startLocation.y - center.y - offset.height) / scale
                                        )
                                        
                                        let bottomRect = CGRect(x: node.position.x, y: node.position.y + node.size.height - 15, width: node.size.width, height: 15)
                                        if bottomRect.contains(chartPoint) {
                                            connectingEdge = (source: nodeID, currentPoint: chartPoint)
                                            return
                                        }
                                    }
                                    
                                    dragMode = .node(nodeID)
                                    if !selection.contains(nodeID) { selection = [nodeID] }
                                    dragStartNodePositions = nodes.reduce(into: [:]) { dict, node in
                                        if selection.contains(node.id) { dict[node.id] = node.position }
                                    }
                                } else {
                                    dragMode = .pan
                                }
                            }
                            
                            if let connecting = connectingEdge {
                                // Transform current translation to chart coordinates
                                let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
                                let currentChartPoint = CGPoint(
                                    x: (value.location.x - center.x - offset.width) / scale,
                                    y: (value.location.y - center.y - offset.height) / scale
                                )
                                connectingEdge = (source: connecting.source, currentPoint: currentChartPoint)
                            } else {
                                switch dragMode {
                                case .pan:
                                    currentDragOffset = value.translation
                                case .node:
                                    let scale = self.scale
                                    let snapGrid: CGFloat = 20.0
                                    
                                    for (id, initialPos) in dragStartNodePositions {
                                        if let index = nodes.firstIndex(where: { $0.id == id }) {
                                            let rawX = initialPos.x + value.translation.width / scale
                                            let rawY = initialPos.y + value.translation.height / scale
                                            
                                            let snappedX = round(rawX / snapGrid) * snapGrid
                                            let snappedY = round(rawY / snapGrid) * snapGrid
                                            
                                            nodes[index].position = CGPoint(x: snappedX, y: snappedY)
                                        }
                                    }
                                case .idle: break
                                }
                            }
                        }
                        .onEnded { value in
                            if let connecting = connectingEdge {
                                // Hit test for target
                                let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
                                let dropPoint = CGPoint(
                                    x: (value.location.x - center.x - offset.width) / scale,
                                    y: (value.location.y - center.y - offset.height) / scale
                                )
                                
                                // Simple manual hit test here to avoid issues with 'hitTestNode' which takes view coordinates
                                if let targetNode = nodes.last(where: { node in
                                    let rect = CGRect(origin: node.position, size: node.size)
                                    return rect.contains(dropPoint)
                                }), targetNode.id != connecting.source {
                                    // Create Edge
                                    let newEdge = FlowEdge(source: connecting.source, target: targetNode.id, label: nil)
                                    edges.append(newEdge)
                                }
                                connectingEdge = nil
                            } else {
                                switch dragMode {
                                case .pan:
                                    offset.width += value.translation.width
                                    offset.height += value.translation.height
                                    currentDragOffset = .zero
                                case .node:
                                    dragStartNodePositions.removeAll()
                                case .idle: break
                                }
                                dragMode = .idle
                            }
                        }
                )
                
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            if initialScale == nil {
                                initialScale = scale
                            }
                            if let startScale = initialScale {
                                scale = startScale * value
                            }
                        }
                        .onEnded { _ in
                            initialScale = nil
                        }
                )
                .onTapGesture { location in
                    hitTest(location: location, in: geometry.size)
                }
            }
            .overlay {
                // Custom Extension Rendering Overlay
                GeometryReader { geo in
                    ZStack {
                        if let renderer = manager.activeRenderer {
                            ForEach(nodes) { node in
                                // Convert node position to view coordinates
                                let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                                let viewX = center.x + offset.width + currentDragOffset.width + node.position.x * scale
                                let viewY = center.y + offset.height + currentDragOffset.height + node.position.y * scale
                                
                                // Only render if visible (simple bounds check)
                                if viewX > -200 && viewX < geo.size.width + 200 &&
                                   viewY > -200 && viewY < geo.size.height + 200 {
                                    
                                    renderer.render(state: StateData(
                                        id: node.id.uuidString,
                                        label: node.label,
                                        type: "\(node.type)",
                                        meta: [:] // TODO: Pass actual meta
                                    ))
                                    .position(x: viewX + (node.size.width * scale / 2), y: viewY + (node.size.height * scale / 2))
                                    .scaleEffect(scale)
                                    .allowsHitTesting(false) 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    private func drawEdge(context: GraphicsContext, source: FlowNode, target: FlowNode, edge: FlowEdge) {
        let sourceRect = CGRect(origin: source.position, size: source.size)
        let targetRect = CGRect(origin: target.position, size: target.size)
        
        // Calculate best connection points
        let (startPoint, endPoint) = calculateConnectionPoints(from: sourceRect, to: targetRect)
        
        var path = Path()
        path.move(to: startPoint)
        
        switch edge.routingType {
        case .straight:
            path.addLine(to: endPoint)
            
        case .orthogonal:
            if let waypoints = edge.waypoints, !waypoints.isEmpty {
                for point in waypoints {
                    path.addLine(to: point)
                }
                path.addLine(to: endPoint)
            } else {
                // Simple Manhattan routing: Move X then Y, or Y then X
                // Improved: Avoid crossing directly through source node if possible
                let midX = (startPoint.x + endPoint.x) / 2
                let midY = (startPoint.y + endPoint.y) / 2
                
                // If standard vertical flow
                path.addLine(to: CGPoint(x: startPoint.x, y: midY))
                path.addLine(to: CGPoint(x: endPoint.x, y: midY))
                path.addLine(to: endPoint)
            }
            
        case .curved:
            // Improved Bezier CURVE
            // Control points depend on relative positioning (e.g., if target is below, curve down)
            
            let deltaX = abs(endPoint.x - startPoint.x)
            let deltaY = abs(endPoint.y - startPoint.y)
            
            // Heuristic for control point distance
            let curvature = max(deltaX, deltaY) / 2
            
            // Standard vertical statechart flow is usually top-to-bottom
            let controlPoint1 = CGPoint(x: startPoint.x, y: startPoint.y + curvature)
            let controlPoint2 = CGPoint(x: endPoint.x, y: endPoint.y - curvature)
            
            path.addCurve(to: endPoint, control1: controlPoint1, control2: controlPoint2)
        }
        
        let isSelected = selection.contains(edge.id)
        let strokeColor: Color = isSelected ? .accentColor : .gray
        let strokeWidth: CGFloat = isSelected ? 3 : 2
        
        context.stroke(path, with: .color(strokeColor), lineWidth: strokeWidth)
        
        // arrow logic would go here
        drawArrow(context: context, endPoint: endPoint, startPoint: startPoint, color: strokeColor)
    }
    
    private func calculateConnectionPoints(from sourceSpy: CGRect, to targetSpy: CGRect) -> (CGPoint, CGPoint) {
        // Simple logic: connect closest edges or centers
        // For Statecharts, usually Exit Bottom -> Enter Top is preferred for hierarchical
        // But for generic, let's just use Center-Center projected to bounds?
        // Or specific ports?
        // Let's stick strictly to Top/Bottom/Left/Right centers for Orthogonal routing.
        
        let sourceCenter = CGPoint(x: sourceSpy.midX, y: sourceSpy.midY)
        let targetCenter = CGPoint(x: targetSpy.midX, y: targetSpy.midY)
        
        // Naive implementation: Use rigid ports for now: Bottom of Source -> Top of Target
        // This usually looks best for trees.
        // For loops, it might look bad.
        // Let's refine:
        
        if targetSpy.minY >= sourceSpy.maxY {
            // Target is below Source
            return (CGPoint(x: sourceSpy.midX, y: sourceSpy.maxY), CGPoint(x: targetSpy.midX, y: targetSpy.minY))
        } else if targetSpy.maxY <= sourceSpy.minY {
            // Target is above Source
            return (CGPoint(x: sourceSpy.midX, y: sourceSpy.minY), CGPoint(x: targetSpy.midX, y: targetSpy.maxY))
        } else if targetSpy.minX >= sourceSpy.maxX {
            // Target is right of Source
            return (CGPoint(x: sourceSpy.maxX, y: sourceSpy.midY), CGPoint(x: targetSpy.minX, y: targetSpy.midY))
        } else {
            // Target is left of Source
            return (CGPoint(x: sourceSpy.minX, y: sourceSpy.midY), CGPoint(x: targetSpy.maxX, y: targetSpy.midY))
        }
    }
    
    private func drawArrow(context: GraphicsContext, endPoint: CGPoint, startPoint: CGPoint, color: Color) {
        // Calculate angle
        let dx = endPoint.x - startPoint.x
        let dy = endPoint.y - startPoint.y
        let angle = atan2(dy, dx)
        
        // Only draw if line is long enough
        if sqrt(dx*dx + dy*dy) < 10 { return }
        
        // Arrow head
        let arrowLength: CGFloat = 10
        let arrowAngle: CGFloat = .pi / 6 // 30 degrees
        
        let p1 = CGPoint(
            x: endPoint.x - arrowLength * cos(angle - arrowAngle),
            y: endPoint.y - arrowLength * sin(angle - arrowAngle)
        )
        let p2 = CGPoint(
            x: endPoint.x - arrowLength * cos(angle + arrowAngle),
            y: endPoint.y - arrowLength * sin(angle + arrowAngle)
        )
        
        var arrowPath = Path()
        arrowPath.move(to: endPoint)
        arrowPath.addLine(to: p1)
        arrowPath.addLine(to: p2)
        arrowPath.closeSubpath()
        
        context.fill(arrowPath, with: .color(color))
    }
    
    private func drawConnectionDrag(context: GraphicsContext, source: FlowNode, endPoint: CGPoint) {
        let sourceRect = CGRect(x: source.position.x, y: source.position.y, width: 120, height: 60)
        let startPoint = CGPoint(x: sourceRect.midX, y: sourceRect.maxY)
        
        var path = Path()
        path.move(to: startPoint)
        path.addLine(to: endPoint)
        
        context.stroke(path, with: .color(.accentColor), style: StrokeStyle(lineWidth: 2, dash: [5, 5]))
    }

    private func drawNode(context: GraphicsContext, node: FlowNode) {
        let nodeRect = CGRect(origin: node.position, size: node.size)
        // Basic check for dark mode from environment
        let isDark = context.environment.colorScheme == .dark
        
        let isSelected = selection.contains(node.id)
        let isActive = activeStateIDs.contains(node.id)
        
        // --- 1. Base Shape & Path ---
        var cornerRadius: CGFloat = 8
        if node.type == .compound || node.type == .parallel { cornerRadius = 12 }
        if node.type == .history || node.type == .final { cornerRadius = node.size.width / 2 }
        
        let path = Path(roundedRect: nodeRect, cornerRadius: cornerRadius)
        
        // Active State Glow
        if isActive {
            var glowContext = context
            let glowColor: Color = .green
            glowContext.addFilter(.shadow(color: glowColor.opacity(0.6), radius: isDark ? 10 : 6, x: 0, y: 0))
            glowContext.stroke(path, with: .color(glowColor), lineWidth: 4)
        }
        
        // --- 2. Shadows (Hierarchy Depth) ---
        if !isActive {
           if !isDark {
               // Light Mode: Standard drop shadow
               var shadowContext = context
               shadowContext.addFilter(.shadow(color: .black.opacity(0.15), radius: 4, x: 0, y: 2))
               shadowContext.fill(path, with: .color(.white.opacity(0.01)))
           }
        }
        
        // --- 3. Fill Styling ---
        var fillColor: Color
        
        switch node.type {
        case .atomic:
            fillColor = isDark ? Color(red: 0.15, green: 0.15, blue: 0.17) : .white
            if isSelected { fillColor = isDark ? Color.blue.opacity(0.3) : Color.accentColor.opacity(0.05) }
            if isActive { fillColor = Color.green.opacity(0.1) }
            
        case .compound:
            fillColor = isDark ? Color(red: 0.10, green: 0.10, blue: 0.12) : Color(white: 0.96)
            
        case .parallel:
            fillColor = .clear
            
        case .final:
            fillColor = isDark ? Color(red: 0.2, green: 0.2, blue: 0.2) : .white
            
        case .history:
            fillColor = .yellow.opacity(isDark ? 0.3 : 0.2)
        }

        context.fill(path, with: .color(fillColor))
        
        // --- 4. Special Headers for Compound ---
        if node.type == .compound {
            let headerHeight: CGFloat = 24
            let headerRect = CGRect(x: nodeRect.minX, y: nodeRect.minY, width: nodeRect.width, height: headerHeight)
            let headerPath = Path(roundedRect: headerRect, cornerSize: CGSize(width: cornerRadius, height: cornerRadius), style: .continuous)
            
            var headerContext = context
            headerContext.clip(to: path)
            
            let headerFill = isDark ? Color(white: 0.2) : Color(white: 0.93)
            headerContext.fill(headerPath, with: .color(headerFill))
        }
        
        // --- 5. Borders / Strokes ---
        var strokeColor: Color = isDark ? .white.opacity(0.2) : .secondary.opacity(0.5)
        var strokeStyle = StrokeStyle(lineWidth: 1)
        
        if isSelected {
            strokeColor = .accentColor
            strokeStyle.lineWidth = 3
        }
        
        if node.type == .parallel {
            strokeStyle.dash = [5, 5]
            if !isSelected { strokeColor = isDark ? .white.opacity(0.3) : .gray.opacity(0.5) }
        }
        
        if node.type == .final {
            strokeStyle.lineWidth = 4
            strokeColor = isDark ? .white : .black
        }
        
        context.stroke(path, with: .color(strokeColor), style: strokeStyle)
        
        // --- 6. Typography ---
        let textColor = isDark ? Color.white : Color.primary
        var textPoint = CGPoint(x: nodeRect.midX, y: nodeRect.midY)
        var font: Font = .system(size: 14, weight: .medium)
        
        if node.type == .compound || node.type == .parallel {
            textPoint = CGPoint(x: nodeRect.minX + 8, y: nodeRect.minY + 12)
            font = .system(size: 12, weight: .bold)
            context.draw(Text(node.label).font(font).foregroundColor(isDark ? .white.opacity(0.8) : .secondary), at: textPoint, anchor: .leading)
        } else {
            if node.type != .final {
                context.draw(Text(node.label).font(font).foregroundColor(textColor), at: textPoint)
            }
        }
    }
    
    private func doubleStroke(_ selected: Bool) -> Bool {
        return false // Simplified
    }
    
    private func hitTest(location: CGPoint, in size: CGSize) {
        if let nodeID = hitTestNode(at: location, in: size) {
            // Priority to nodes
            selection = [nodeID]
        } else if let edgeID = hitTestEdge(at: location, in: size) {
            // Then edges
            selection = [edgeID]
        } else {
            // Background tap
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
            let rect = CGRect(x: node.position.x, y: node.position.y, width: 120, height: 60)
            return rect.contains(chartPoint)
        })?.id
    }
    
    private func hitTestEdge(at location: CGPoint, in size: CGSize) -> UUID? {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let chartPoint = CGPoint(
            x: (location.x - center.x - offset.width) / scale,
            y: (location.y - center.y - offset.height) / scale
        )
        
        // Threshold adapts to scale so it feels consistent to user touch
        let threshold: CGFloat = 12.0 / scale 
        
        return edges.first(where: { edge in
            guard let source = nodes.first(where: { $0.id == edge.source }),
                  let target = nodes.first(where: { $0.id == edge.target }) else { return false }
            
            let sourceRect = CGRect(origin: source.position, size: source.size)
            let targetRect = CGRect(origin: target.position, size: target.size)
            
            let (startPoint, endPoint) = calculateConnectionPoints(from: sourceRect, to: targetRect)
            
            // Check all segments
            var segments: [(CGPoint, CGPoint)] = []
            
            switch edge.routingType {
            case .orthogonal:
                if let waypoints = edge.waypoints, !waypoints.isEmpty {
                    var current = startPoint
                    for wp in waypoints {
                        segments.append((current, wp))
                        current = wp
                    }
                    segments.append((current, endPoint))
                } else {
                    // Match drawEdge fallback: Start -> (Optionally X first or Y first)
                    // drawEdge logic: midY used. Start -> (Start.x, midY) -> (End.x, midY) -> End
                    let midY = (startPoint.y + endPoint.y) / 2
                    let p1 = CGPoint(x: startPoint.x, y: midY)
                    let p2 = CGPoint(x: endPoint.x, y: midY)
                    
                    segments.append((startPoint, p1))
                    segments.append((p1, p2))
                    segments.append((p2, endPoint))
                }
                
            case .straight:
                segments.append((startPoint, endPoint))
                
            case .curved:
                 // Approximate bezier with straight line for now
                 // Better approximation: Divide into 2 segments via control points? 
                 // Simple straight check is okay for MVP
                 segments.append((startPoint, endPoint))
            }
            
            for segment in segments {
                if distance(from: chartPoint, toLineSegment: segment) < threshold {
                    return true
                }
            }
            return false
        })?.id
    }
    
    // Distance from point p to line segment (v, w)
    private func distance(from p: CGPoint, toLineSegment segment: (CGPoint, CGPoint)) -> CGFloat {
        let v = segment.0
        let w = segment.1
        let l2 = distanceSquared(v, w)
        if l2 == 0 { return distanceSquared(p, v).squareRoot() }
        
        var t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2
        t = max(0, min(1, t))
        
        let projection = CGPoint(x: v.x + t * (w.x - v.x), y: v.y + t * (w.y - v.y))
        return distanceSquared(p, projection).squareRoot()
    }
    
    private func distanceSquared(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        let dx = p1.x - p2.x
        let dy = p1.y - p2.y
        return dx*dx + dy*dy
    }
    
    // Helper to get selected node for Context Menu (SwiftUI 2024 ContextMenu is view-based, so we might need an overlay approach or just rely on selection commands for now)
    // A better approach for Canvas interaction is to overlay invisible views or handling proper right-clicks if on macOS. 
    // For now, let's keep interactions simple and handled via the wrapper view or main menu commands.
}

struct FlowBackground: View {
    let scale: CGFloat
    let offset: CGSize
    
    var body: some View {
        Canvas { context, size in
            let spacing: CGFloat = 40 * scale
            // Grid implementation...
            let center = CGPoint(x: size.width / 2 + offset.width, y: size.height / 2 + offset.height)
            let path = Path { p in
                 var x = center.x.remainder(dividingBy: spacing)
                if x < 0 { x += spacing }
                while x < size.width {
                    p.move(to: CGPoint(x: x, y: 0))
                    p.addLine(to: CGPoint(x: x, y: size.height))
                    x += spacing
                }
                var y = center.y.remainder(dividingBy: spacing)
                if y < 0 { y += spacing }
                while y < size.height {
                    p.move(to: CGPoint(x: 0, y: y))
                    p.addLine(to: CGPoint(x: size.width, y: y))
                    y += spacing
                }
            }
            context.stroke(path, with: .color(Color.primary.opacity(0.05)), lineWidth: 1)
        }
        .background(Color(UIColor.systemGroupedBackground))
        .drawingGroup()
    }
}

// Helper to add CGSize
extension CGSize {
    static func +(lhs: CGSize, rhs: CGSize) -> CGSize {
        CGSize(width: lhs.width + rhs.width, height: lhs.height + rhs.height)
    }
    static func +=(lhs: inout CGSize, rhs: CGSize) {
        lhs = lhs + rhs
    }
}

#Preview("Flow View - Complex Graph") {
    // State Injection Pattern for Binding support in Previews
    struct PreviewWrapper: View {
        @State var nodes: [FlowNode] = [
            FlowNode(id: UUID(), position: CGPoint(x: 100, y: 100), label: "Start", type: .atomic),
            FlowNode(id: UUID(), position: CGPoint(x: 400, y: 100), label: "Process", type: .compound),
            FlowNode(id: UUID(), position: CGPoint(x: 400, y: 300), label: "End", type: .final)
        ]
        @State var edges: [FlowEdge] = []
        @State var scale: CGFloat = 1.0
        @State var offset: CGSize = .zero
        @State var selection: Set<UUID> = []
        @State var activeStateIDs: Set<UUID> = []
        
        init() {
            // Setup edges in init to access node IDs
            let start = nodes[0].id
            let process = nodes[1].id
            let end = nodes[2].id
            
            _edges = State(initialValue: [
                FlowEdge(source: start, target: process, label: "begin"),
                FlowEdge(source: process, target: end, label: "finish")
            ])
            
            _activeStateIDs = State(initialValue: [start])
        }
        
        var body: some View {
            FlowView(nodes: $nodes, edges: $edges, activeStateIDs: $activeStateIDs, scale: $scale, offset: $offset, selection: $selection)
                .overlay(alignment: .bottomLeading) {
                    VStack(alignment: .leading) {
                        Text("Scale: \(scale, format: .number.precision(.fractionLength(2)))")
                        Text("Offset: \(offset.width, format: .number.precision(.fractionLength(0))), \(offset.height, format: .number.precision(.fractionLength(0)))")
                        Text("Selection: \(selection.count) items")
                    }
                    .font(.caption)
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .padding()
                }
        }
    }
    
    return PreviewWrapper()
}
