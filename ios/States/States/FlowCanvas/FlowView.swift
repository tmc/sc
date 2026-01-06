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
    
    // Reparenting Logic
    @State private var hoveredParentID: UUID? = nil
    
    // Context Menu Logic
    @State private var lastContextLocation: CGPoint? = nil
    
    // Space Panning (macOS)
    @State private var isSpacePressed = false
    @State private var eventMonitor: Any? = nil
    
    // Connection Logic
    @State private var connectingEdge: (source: UUID, currentPoint: CGPoint)? = nil
    
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

    public init(nodes: Binding<[FlowNode]>,
                edges: Binding<[FlowEdge]>,
                activeStateIDs: Binding<Set<UUID>>,
                scale: Binding<CGFloat>,
                offset: Binding<CGSize>,
                selection: Binding<Set<UUID>>,
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
                    .overlay(ScrollEventView(offset: $offset, scale: $scale)) // Capture Scroll Events
                    .onTapGesture {
                        // Clear selection/editing on background tap
                        editingNodeID = nil
                        selection = []
                    }

                
                // Canvas
                TimelineView(.animation) { timeline in
                    let phase = timeline.date.timeIntervalSinceReferenceDate
                    
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
                        
                        // Draw Nodes (Sorted by hierarchy)
                        let sortedNodes = nodes.sorted { (a, b) -> Bool in
                            if a.id == b.parentID { return true }
                            if b.id == a.parentID { return false }
                            if a.parentID == nil && b.parentID != nil { return true }
                            if a.parentID != nil && b.parentID == nil { return false }
                            return false
                        }
                        
                        for node in sortedNodes {
                            drawNode(context: context, node: node, phase: phase)
                        }
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .local)
                        .onChanged { value in
                            handleDragChanged(value: value, in: geometry.size)
                        }
                        .onEnded { value in
                            handleDragEnded(value: value, in: geometry.size)
                        }
                )

                // Single tap handled by overlay primarily, leaving this for drag disambiguation?
                // Actually remove general tap gesture here to let overlay handle specific taps
                
                // Interaction Overlay
                interactionOverlay(in: geometry)
                
                // Custom Extension Rendering Overlay
                extensionOverlay(in: geometry)

            }
            .contentShape(Rectangle()) // Ensure ZStack captures touches
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                     lastContextLocation = location
                case .ended: break
                }
            }
            .contextMenu {
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
    
    @ViewBuilder
    private func interactionOverlay(in geo: GeometryProxy) -> some View {
        let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
        
        ForEach($nodes) { $node in
            let nodePos = node.position
            let viewX = center.x + offset.width + currentDragOffset.width + nodePos.x * scale
            let viewY = center.y + offset.height + currentDragOffset.height + nodePos.y * scale
            let width = node.size.width * scale
            let height = node.size.height * scale
            
            // Interaction Zone
            Color.white.opacity(0.01)
                .contentShape(RoundedRectangle(cornerRadius: 8))
                .frame(width: width, height: height)
                .position(x: viewX + width/2, y: viewY + height/2)
                .onTapGesture(count: 2) {
                    editingNodeID = node.id
                    editingText = node.label
                    isEditingFocus = true
                }
                .onTapGesture(count: 1) {
                    hitTest(location: CGPoint(x: viewX + width/2, y: viewY + height/2), in: geo.size)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(node.label)
                .accessibilityIdentifier(node.label)
                .accessibilityAddTraits(.isButton)
                .contextMenu {
                    Button {
                        editingNodeID = node.id
                        editingText = node.label
                        isEditingFocus = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    Button {
                        onAddSubstate?(node.id)
                    } label: {
                        Label("Add Sub-state", systemImage: "plus.square.on.square")
                    }
                    Button {
                        onNodeDuplicate?(node.id)
                    } label: {
                        Label("Duplicate", systemImage: "doc.on.doc")
                    }
                    Divider()
                    Button(role: .destructive) {
                        onNodeDelete?(node.id)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
        }
        
        // Editing TextField
        if let id = editingNodeID, let node = nodes.first(where: { $0.id == id }) {
            let viewX = center.x + offset.width + currentDragOffset.width + node.position.x * scale
            let viewY = center.y + offset.height + currentDragOffset.height + node.position.y * scale
             
            TextField("Label", text: $editingText)
                .textFieldStyle(.plain)
                .font(Theme.Typography.nodeLabel(scale: scale))
                .multilineTextAlignment(.center)
                .padding(4)
                .background(Theme.Colors.nodeBackground) // Match background
                .cornerRadius(4)
                .focused($isEditingFocus)
                .frame(width: node.size.width * scale * 1.5) // Allow some overflow
                .position(x: viewX + (node.size.width * scale / 2), y: viewY + (node.size.height * scale / 2))
                .onSubmit {
                    commitRename(id: id)
                }
                .onChange(of: isEditingFocus) { oldValue, newValue in
                    if !newValue { commitRename(id: id) }
                }
        }
    }
    
    private func commitRename(id: UUID) {
        guard let index = nodes.firstIndex(where: { $0.id == id }) else { return }
        if nodes[index].label != editingText {
            nodes[index].label = editingText
            onNodeRename?(id, editingText)
        }
        editingNodeID = nil
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
                dragStartNodePositions = nodes.reduce(into: [:]) { dict, node in
                    if selection.contains(node.id) { dict[node.id] = node.position }
                }
            } else {
                dragMode = .pan
            }
        }
        
        if let connecting = connectingEdge {
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            var currentChartPoint = CGPoint(
                x: (location.x - center.x - offset.width) / scale,
                y: (location.y - center.y - offset.height) / scale
            )
            
            // Magnetic Snapping to Ports
            if let targetID = hitTestNode(at: location, in: size),
               let targetNode = nodes.first(where: { $0.id == targetID }),
               targetID != connecting.source {
                
                let nodeRect = CGRect(origin: targetNode.position, size: targetNode.size)
                let ports = [
                    CGPoint(x: nodeRect.midX, y: nodeRect.minY), // Top
                    CGPoint(x: nodeRect.maxX, y: nodeRect.midY), // Right
                    CGPoint(x: nodeRect.midX, y: nodeRect.maxY), // Bottom
                    CGPoint(x: nodeRect.minX, y: nodeRect.midY)  // Left
                ]
                
                // Find nearest port
                if let nearest = ports.min(by: { distance($0, currentChartPoint) < distance($1, currentChartPoint) }),
                   distance(nearest, currentChartPoint) < 50 { // Snap threshold
                    currentChartPoint = nearest
                    // Haptic feedback could go here if state changed
                }
            }
            
            connectingEdge = (source: connecting.source, currentPoint: currentChartPoint)
        } else {
            switch dragMode {
            case .pan:
                currentDragOffset = translation
            case .node:
                let scale = self.scale
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
                let newEdge = FlowEdge(source: connecting.source, target: targetNode.id, label: nil)
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
                
                onNodeMoveEnded?(dragStartNodePositions) // Register undo (position change still happened)
                dragStartNodePositions.removeAll()
                hoveredParentID = nil // Reset hover
            case .idle: break
            }
            dragMode = .idle
        }
    }
    
    // MARK: - Drawing Logic
    
    private func drawNode(context: GraphicsContext, node: FlowNode, phase: TimeInterval) {
        let nodeRect = CGRect(origin: node.position, size: node.size)
        // Basic check for dark mode from environment
        // let isDark = context.environment.colorScheme == .dark
        
        // Use Theme
        let isSelected = selection.contains(node.id)
        let isActive = activeStateIDs.contains(node.id)
        
        // Corner Radius
        var cornerRadius = Theme.Layout.nodeCornerRadius
        if node.type == .compound || node.type == .parallel { cornerRadius = 12 }
        if node.type == .history || node.type == .final { cornerRadius = node.size.width / 2 }
        
        let path = Path(roundedRect: nodeRect, cornerRadius: cornerRadius)
        
        // 1. Shadows (High Quality)
        if !isActive {
            var shadowContext = context
            shadowContext.addFilter(.shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 3))
            shadowContext.fill(path, with: .color(.white)) // Invisible fill to cast shadow
        }
        
        // 2. Active Glow & Breathing
        if isActive {
            var glowContext = context
            
            // Calculate Breathing: Oscillate between 0.0 and 1.0 roughly every 2.5s
            let breath = (sin(phase * 2.5) + 1) / 2
            // Map to opacity range [0.3, 0.7]
            let opacity = 0.3 + (breath * 0.4)
            let radius = 8 + (breath * 6)
            
            // Layer 1: Wide, breathing glow
            glowContext.addFilter(.shadow(color: Theme.Colors.activeNodeGlow.opacity(opacity), radius: radius, x: 0, y: 0))
            glowContext.stroke(path, with: .color(Theme.Colors.activeNodeGlow), lineWidth: 4)
            
            // Layer 2: Core brightness
            let coreContext = context
            coreContext.stroke(path, with: .color(.white.opacity(0.4)), lineWidth: 1)
        }
        
        // 2b. Reparent Highlight
        if hoveredParentID == node.id {
             var highlightContext = context
             highlightContext.addFilter(.shadow(color: Theme.Colors.accent.opacity(0.6), radius: 10, x: 0, y: 0))
             highlightContext.stroke(path, with: .color(Theme.Colors.accent), lineWidth: 4)
        }
        
        // 3. Fill
        var fillColor: Color
        switch node.type {
        case .atomic:
            fillColor = Theme.Colors.nodeBackground
            if isActive { fillColor = Theme.Colors.activeNodeTint }
        case .compound:
            fillColor = Color.white.opacity(0.9)
        case .parallel:
            fillColor = .clear
        case .final:
            fillColor = .primary // Adaptive black/white
        case .history:
            fillColor = .yellow.opacity(0.2)
        }
        
        context.fill(path, with: .color(fillColor))
        
        // 4. Compound Headers
        if node.type == .compound {
            let headerHeight: CGFloat = 28
            let headerRect = CGRect(x: nodeRect.minX, y: nodeRect.minY, width: nodeRect.width, height: headerHeight)
            let headerPath = Path(roundedRect: headerRect, cornerSize: CGSize(width: cornerRadius, height: cornerRadius), style: .continuous)
            
            var headerContext = context
            headerContext.clip(to: path)
            headerContext.fill(headerPath, with: .color(Color.black.opacity(0.03)))
            
            // Separator line
            let lineY = nodeRect.minY + headerHeight
            var linePath = Path()
            linePath.move(to: CGPoint(x: nodeRect.minX, y: lineY))
            linePath.addLine(to: CGPoint(x: nodeRect.maxX, y: lineY))
            context.stroke(linePath, with: .color(Theme.Colors.nodeBorder), lineWidth: 1)
        }
        
        // 5. Borders / Selection Halo
        var strokeColor = Theme.Colors.nodeBorder
        var strokeStyle = StrokeStyle(lineWidth: 1)
        
        if isSelected {
            strokeColor = Theme.Colors.accent
            strokeStyle.lineWidth = 2.5
        }
        
        if node.type == .parallel {
            strokeStyle.dash = [6, 4]
        }
        
        context.stroke(path, with: .color(strokeColor), style: strokeStyle)
        
        // 6. Typography
        let resolvedFont = Theme.Typography.nodeLabel(scale: scale)
        let textColor = Color.primary
        
        var textPoint = CGPoint(x: nodeRect.midX, y: nodeRect.midY)
        
        if node.type == .compound || node.type == .parallel {
            textPoint = CGPoint(x: nodeRect.minX + 10, y: nodeRect.minY + 14) // Adjusted for header
            // Use SwiftUI Text for rendering
            context.draw(Text(node.label).font(.system(size: 13, weight: .semibold)).foregroundStyle(.secondary), at: textPoint, anchor: .leading)
        } else if node.type != .final {
             context.draw(Text(node.label).font(resolvedFont).foregroundStyle(textColor), at: textPoint)
        }
        
        // 7. Connection Ports (Connection or Selection)
        if connectingEdge != nil || isSelected {
            drawPorts(context: context, rect: nodeRect, isHovered: false) // isHovered logic TODO
        }
    }
    
    private func drawPorts(context: GraphicsContext, rect: CGRect, isHovered: Bool) {
        let ports = [
            CGPoint(x: rect.midX, y: rect.minY), // Top
            CGPoint(x: rect.maxX, y: rect.midY), // Right
            CGPoint(x: rect.midX, y: rect.maxY), // Bottom
            CGPoint(x: rect.minX, y: rect.midY)  // Left
        ]
        
        for port in ports {
            let portRect = CGRect(x: port.x - 4, y: port.y - 4, width: 8, height: 8)
            let portPath = Path(ellipseIn: portRect)
            
            context.fill(portPath, with: .color(Theme.Colors.accent))
            context.stroke(portPath, with: .color(.white), lineWidth: 1.5)
        }
    }
    
    private func drawEdge(context: GraphicsContext, source: FlowNode, target: FlowNode, edge: FlowEdge) {
        let sourceRect = CGRect(origin: source.position, size: source.size)
        let targetRect = CGRect(origin: target.position, size: target.size)
        
        let (startPoint, endPoint) = calculateConnectionPoints(from: sourceRect, to: targetRect)
        
        var path = Path()
        path.move(to: startPoint)
        
        // Simple Orthogonal Fallback for now, could use edge.routingType
        if let waypoints = edge.waypoints, !waypoints.isEmpty {
            for point in waypoints { path.addLine(to: point) }
            path.addLine(to: endPoint)
        } else {
            let midY = (startPoint.y + endPoint.y) / 2
            path.addLine(to: CGPoint(x: startPoint.x, y: midY))
            path.addLine(to: CGPoint(x: endPoint.x, y: midY))
            path.addLine(to: endPoint)
        }
        
        let isSelected = selection.contains(edge.id)
        let color = isSelected ? Theme.Colors.accent : Color.gray.opacity(0.8)
        let width: CGFloat = isSelected ? 3 : 2
        
        context.stroke(path, with: .color(color), lineWidth: width)
        drawArrow(context: context, endPoint: endPoint, startPoint: startPoint, color: color)
    }
    
    private func drawArrow(context: GraphicsContext, endPoint: CGPoint, startPoint: CGPoint, color: Color) {
         let dx = endPoint.x - startPoint.x
         let dy = endPoint.y - startPoint.y
         let angle = atan2(dy, dx)
         if sqrt(dx*dx + dy*dy) < 10 { return }
         
         let arrowLength: CGFloat = 10
         let arrowAngle: CGFloat = .pi / 6
         
         let p1 = CGPoint(x: endPoint.x - arrowLength * cos(angle - arrowAngle), y: endPoint.y - arrowLength * sin(angle - arrowAngle))
         let p2 = CGPoint(x: endPoint.x - arrowLength * cos(angle + arrowAngle), y: endPoint.y - arrowLength * sin(angle + arrowAngle))
         
         var arrowPath = Path()
         arrowPath.move(to: endPoint)
         arrowPath.addLine(to: p1)
         arrowPath.addLine(to: p2)
         arrowPath.closeSubpath()
         
         context.fill(arrowPath, with: .color(color))
    }
    
    private func drawConnectionDrag(context: GraphicsContext, source: FlowNode, endPoint: CGPoint) {
        let sourceRect = CGRect(origin: source.position, size: source.size)
        let startPoint = CGPoint(x: sourceRect.midX, y: sourceRect.maxY)
        
        var path = Path()
        path.move(to: startPoint)
        path.addLine(to: endPoint)
        
        context.stroke(path, with: .color(Theme.Colors.accent), style: StrokeStyle(lineWidth: 2, dash: [5, 5]))
    }

    private func calculateConnectionPoints(from sourceSpy: CGRect, to targetSpy: CGRect) -> (CGPoint, CGPoint) {
        if targetSpy.minY >= sourceSpy.maxY {
             return (CGPoint(x: sourceSpy.midX, y: sourceSpy.maxY), CGPoint(x: targetSpy.midX, y: targetSpy.minY))
        } else if targetSpy.maxY <= sourceSpy.minY {
             return (CGPoint(x: sourceSpy.midX, y: sourceSpy.minY), CGPoint(x: targetSpy.midX, y: targetSpy.maxY))
        } else if targetSpy.minX >= sourceSpy.maxX {
             return (CGPoint(x: sourceSpy.maxX, y: sourceSpy.midY), CGPoint(x: targetSpy.minX, y: targetSpy.midY))
        } else {
             return (CGPoint(x: sourceSpy.minX, y: sourceSpy.midY), CGPoint(x: targetSpy.maxX, y: targetSpy.midY))
        }
    }
    
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
        let threshold: CGFloat = 12.0 / scale
        
        return edges.first(where: { edge in
            guard let source = nodes.first(where: { $0.id == edge.source }),
                  let target = nodes.first(where: { $0.id == edge.target }) else { return false }
            
            let sRect = CGRect(origin: source.position, size: source.size)
            let tRect = CGRect(origin: target.position, size: target.size)
            let (start, end) = calculateConnectionPoints(from: sRect, to: tRect)
            
            // Simple mid-point check for orthogonal
            let midY = (start.y + end.y) / 2
            let p1 = CGPoint(x: start.x, y: midY)
            let p2 = CGPoint(x: end.x, y: midY)
            
            let segments = [(start, p1), (p1, p2), (p2, end)]
            for segment in segments {
                if distance(from: chartPoint, toLineSegment: segment) < threshold { return true }
            }
            return false
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
