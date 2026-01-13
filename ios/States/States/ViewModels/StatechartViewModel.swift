import SwiftUI
import Observation
import SwiftProtobuf

@Observable
class StatechartViewModel {
    var machine: StatechartWrapper
    var nodes: [FlowNode] = []
    var edges: [FlowEdge] = [] // Placeholder for edges
    var selection: Set<UUID> = []
    var activeStateIDs: Set<UUID> = [] // IDs of currently active states
    var scale: CGFloat = 1.0
    var offset: CGSize = .zero
    
    // MARK: - Marquee Selection
    var selectionRect: CGRect? = nil
    var initialSelectionBeforeDrag: Set<UUID> = []
    
    // MARK: - Analysis
    let analysisEngine = AnalysisEngine()
    var analysisReport: AnalysisReport?
    var showAnalysis: Bool = false
    
    // MARK: - Async Loading
    var isLoading: Bool = false
    private var loadTask: Task<Void, Never>?
    
    // MARK: - Engine (Semantics)
    var engine: StatechartEngine?
    var showContextInspector: Bool = false
    
    // MARK: - Editor Mode
    enum EditorMode {
        case editing
        case simulation
    }
    var mode: EditorMode = .editing
    
    init(machine: StatechartWrapper) {
        self.machine = machine
        load(machine: machine)
    }
    
    func load(machine: StatechartWrapper) {
        self.machine = machine
        
        // Cancel previous task
        loadTask?.cancel()
        
        // Reset state immediately
        self.nodes = []
        self.edges = []
        self.selection = []
        self.activeStateIDs = []
        self.simulationHistory = []
        self.currentStepIndex = 0
        self.mode = .editing
        self.isLoading = true
        
        guard let json = machine.jsonContent, !json.isEmpty else {
            self.isLoading = false
            return
        }
        
        // Background Parsing
        loadTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            
            // Perform heavy parsing in background
            var newNodes: [FlowNode] = []
            var newEdges: [FlowEdge] = []
            var newActive: Set<UUID> = []
            var newHistory: [Set<UUID>] = []
            
            // 1. Attempt Proto (Fastest if available?)
            // We need to re-create the wrapper logic here or just decode manually.
            // StatechartWrapper.proto computes it property.
            // We can just use StatechartWrapper logic if it's thread safe (it's a struct, so yes).
            // But 'machine' capture is thread safe? Yes, value type.
            
            // Attempt Proto
            if let proto = machine.proto {
                 let (n, e, a) = StatechartViewModel.parseProtoStatic(proto)
                 newNodes = n
                 newEdges = e
                 newActive = a
            } else if let data = json.data(using: .utf8) {
                // Attempt Standard
                if let definition = try? JSONDecoder().decode(StandardStatechart.self, from: data) {
                    let (n, e, a) = StatechartViewModel.parseStandardStatic(definition)
                    newNodes = n
                    newEdges = e
                    newActive = a
                } else if let definition = try? JSONDecoder().decode(StatelyDefinition.self, from: data) {
                    // Attempt Stately
                    let (n, e) = StatechartViewModel.parseStatelyStatic(definition)
                    newNodes = n
                    newEdges = e
                    // Stately doesn't define active?
                    if let first = n.first { newActive = [first.id] }
                } else {
                     // Error
                     // We can handle error on main actor
                }
            }
            
            if newActive.isEmpty, let first = newNodes.first(where: { $0.type == .atomic }) {
                newActive = [first.id]
            }
            if !newActive.isEmpty { newHistory = [newActive] }
            
            // Update UI on MainActor
            if !Task.isCancelled {
                await MainActor.run {
                    self.nodes = newNodes
                    self.edges = newEdges
                    self.activeStateIDs = newActive
                    self.simulationHistory = newHistory
                    self.isLoading = false
                    
                    self.zoomToFit(viewSize: CGSize(width: 800, height: 600)) // Default size, view will resize on appear
                }
            }
        }
    }
    
    // MARK: - Simulation State
    var simulationHistory: [Set<UUID>] = []
    var currentStepIndex: Int = 0
    
    // MARK: - Simulation Actions
    
    /// Resets simulation to initial state
    func resetSimulation() {
        if let first = simulationHistory.first {
            activeStateIDs = first
            simulationHistory = [first]
            currentStepIndex = 0
            #if os(iOS)
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            #endif
        }
    }

    func stepBack() {
        guard currentStepIndex > 0 else { return }
        currentStepIndex -= 1
        activeStateIDs = simulationHistory[currentStepIndex]
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        #endif
    }

    func stepForward() {
        guard currentStepIndex < simulationHistory.count - 1 else { return }
        currentStepIndex += 1
        activeStateIDs = simulationHistory[currentStepIndex]
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        #endif
    }
    
    func jumpToStep(_ index: Int) {
        guard index >= 0 && index < simulationHistory.count else { return }
        currentStepIndex = index
        activeStateIDs = simulationHistory[index]
    }
    
    /// Simulates a state change by toggling the given state ID
    func simulateToggle(nodeID: UUID) {
        // If we are in history, truncate future
        if currentStepIndex < simulationHistory.count - 1 {
            simulationHistory = Array(simulationHistory.prefix(currentStepIndex + 1))
        }
        
        var newActive = activeStateIDs
        if newActive.contains(nodeID) {
            newActive.remove(nodeID)
        } else {
            newActive.insert(nodeID)
        }
        
        currentStepIndex = simulationHistory.count - 1
    }
    
    /// Sends an event to the state machine, triggering transitions if valid
    func sendEvent(_ eventName: String) {
        // Lazy init engine if needed or update it
        if engine == nil {
            engine = StatechartEngine(nodes: nodes, edges: edges)
            // Sync context if needed
        } else {
            // Ensure graph is up to date (naive sync for now)
            engine?.updateDefinition(nodes: nodes, edges: edges)
        }
        
        // Sync Active State IDs -> Engine (if we manually toggled things outside engine)
        engine?.activeStateIDs = activeStateIDs
        engine?.context = [:] // TODO: Persist context in VM or let Engine own it?
        
        engine?.sendEvent(eventName)
        
        // Sync Result Back
        if let newActive = engine?.activeStateIDs {
            
            // History Management
            if currentStepIndex < simulationHistory.count - 1 {
                simulationHistory = Array(simulationHistory.prefix(currentStepIndex + 1))
            }
            
            if newActive != activeStateIDs {
                activeStateIDs = newActive
                simulationHistory.append(newActive)
                currentStepIndex = simulationHistory.count - 1
                
                #if os(iOS)
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                #endif
            } else {
                // No change
                #if os(iOS)
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
                #endif
            }
        }
        
        // Context Update?
        // We need to inspect context in UI.
        // engine.context is updated.
    }
    
    // MARK: - Analysis Actions
    
    func analyze() {
        self.analysisReport = analysisEngine.analyze(nodes: nodes, edges: edges)
        self.showAnalysis = true
    }
    
    // MARK: - Protobuf Parsing
    
    private static func parseProtoStatic(_ definition: Statecharts_V1_Statechart) -> ([FlowNode], [FlowEdge], Set<UUID>) {
        var pathMap: [String: UUID] = [:]
        var nodes: [FlowNode] = []
        var active: Set<UUID> = []
        
        // 1. Flatten Nodes
        if definition.hasRootState {
            nodes = StatechartViewModel.visitProtoNode(
                definition.rootState,
                parentID: nil,
                currentPath: [],
                pathMap: &pathMap,
                parentOrigin: .zero,
                suggestedOffset: CGPoint(x: 50, y: 50)
            )
        }
        
        // 2. Map Edges
        let edges = StatechartViewModel.mapProtoEdges(from: definition.transitions, pathMap: pathMap)
        
        // 3. Set Initial Active State
        if let firstAtomic = nodes.first(where: { $0.type == .atomic }) {
            active.insert(firstAtomic.id)
        }
        
        return (nodes, edges, active)
    }
    
    
    private static func visitProtoNode(_ node: Statecharts_V1_State, parentID: UUID?, currentPath: [String], pathMap: inout [String: UUID], parentOrigin: CGPoint, suggestedOffset: CGPoint) -> [FlowNode] {
        var result: [FlowNode] = []
        let id = UUID()
        
        // Update Path
        let newPath = currentPath + [node.label]
        let pathKey = newPath.joined(separator: "###")
        pathMap[pathKey] = id
        
        // Determine Type
        var nodeType: FlowNode.NodeType = .atomic
        switch node.type {
        case .parallel, .and: nodeType = .parallel
        case .or, .normal: nodeType = .compound
        case .basic: nodeType = .atomic
        default:
            if !node.children.isEmpty {
                nodeType = .compound
            }
        }
        
        // --- Layout Extraction ---
        var finalPosition = CGPoint(x: parentOrigin.x + suggestedOffset.x, y: parentOrigin.y + suggestedOffset.y) // Default to suggested
        var size = CGSize(width: 150, height: 80)
        
        // Attempt to extract Extensions_V1_StateLayout
        for ext in node.extensions {
            if let layout = try? Extensions_V1_StateLayout(unpackingAny: ext) {
                if layout.hasPosition {
                    finalPosition = CGPoint(x: parentOrigin.x + layout.position.x, y: parentOrigin.y + layout.position.y)
                }
                if layout.hasSize {
                    size = CGSize(width: layout.size.width, height: layout.size.height)
                }
                break
            }
        }
        
        let flowNode = FlowNode(
            id: id,
            position: finalPosition,
            label: node.label,
            type: nodeType,
            size: size,
            parentID: parentID
        )
        result.append(flowNode)
        
        // Recursion
        if !node.children.isEmpty {
            var childOffsetY: CGFloat = 80 // Padding from top of parent
            let childOffsetX: CGFloat = 30
            
            for child in node.children {
                // If parent used explicit layout, children likely should too.
                // But we still provide naive suggestions.
                let suggestion = CGPoint(x: childOffsetX, y: childOffsetY)
                
                let childNodes = visitProtoNode(child, parentID: id, currentPath: newPath, pathMap: &pathMap, parentOrigin: finalPosition, suggestedOffset: suggestion)
                result.append(contentsOf: childNodes)
                
                // Stack Naively
                childOffsetY += 100
                
                // If the child was massive, we should jump more?
                // Without measuring child, hard to say. 100 is safe default.
            }
        }
        
        return result
    }
    
    private static func mapProtoEdges(from transitions: [Statecharts_V1_Transition], pathMap: [String: UUID]) -> [FlowEdge] {
        var result: [FlowEdge] = []
        for t in transitions {
            let fromKey = t.from.joined(separator: "###")
            let toKey = t.to.joined(separator: "###")
            
            if let sourceID = pathMap[fromKey], let targetID = pathMap[toKey] {
                let event = t.event
                let guardExpr = t.hasGuard ? t.guard.expression : nil
                // Proto might have actions? 
                // t.actions is repeated Action.
                // We'll join them for now or pick first.
                let action = t.actions.map { $0.label }.joined(separator: "; ")
                
                var waypoints: [CGPoint]? = nil
                var routingType: FlowEdge.RoutingType = .orthogonal
                
                // Attempt to extract TransitionLayout
                for ext in t.extensions {
                    if let layout = try? Extensions_V1_TransitionLayout(unpackingAny: ext) {
                        if !layout.waypoints.isEmpty {
                            waypoints = layout.waypoints.map { CGPoint(x: $0.x, y: $0.y) }
                        }
                        
                        switch layout.edgeStyle {
                        case "straight": routingType = .straight
                        case "curved", "bezier": routingType = .curved
                        case "orthogonal": routingType = .orthogonal
                        default: break // Keep default
                        }
                        break
                    }
                }
                
                result.append(FlowEdge(source: sourceID, target: targetID, event: event, guardExpression: guardExpr, action: action, routingType: routingType, waypoints: waypoints))
            }
        }
        return result
    }
    
    // MARK: - Standard Format Parsing
    
    private static func parseStandardStatic(_ definition: StandardStatechart) -> ([FlowNode], [FlowEdge], Set<UUID>) {
        var pathMap: [String: UUID] = [:]
        var active: Set<UUID> = []
        
        // 1. Flatten Nodes
        let nodes = StatechartViewModel.visitStandardNode(
            definition.rootState,
            parentID: nil,
            currentPath: [],
            pathMap: &pathMap,
            position: CGPoint(x: 50, y: 50)
        )
        
        // 2. Map Edges
        var edges: [FlowEdge] = []
        if let transitions = definition.transitions {
            edges = StatechartViewModel.mapStandardEdges(from: transitions, pathMap: pathMap)
        }
        
        // 3. Set Initial Active State
        if let firstAtomic = nodes.first(where: { $0.type == .atomic }) {
            active.insert(firstAtomic.id)
        }
        
        return (nodes, edges, active)
    }
    
    private static func visitStandardNode(_ node: StandardState, parentID: UUID?, currentPath: [String], pathMap: inout [String: UUID], position: CGPoint) -> [FlowNode] {
        var result: [FlowNode] = []
        let id = UUID()
        
        // Update Path
        let newPath = currentPath + [node.label]
        // Store path string for mapping. Assuming standard dot notation or similar?
        // The JSON has ["Parent", "Child"] arrays for transitions, so we can key by joined string or handle specific format.
        // Let's key by joined array for simplicity, or just match exactly.
        // Actually, the keys in definitions are often arrays.
        // Let's use a standard joiner like "/" or stringify.
        // Wait, the JSON transition "from" is ["Root", "Child"].
        // So we should map [String] -> UUID? Arrays aren't hashable directly in Swift without wrapper.
        // Let's use Joined separator.
        let pathKey = newPath.joined(separator: "###") 
        pathMap[pathKey] = id
        // Also map by label for simple transitions (last write wins for duplicates)
        pathMap[node.label] = id
        
        // Determine Type
        var nodeType: FlowNode.NodeType = .atomic
        switch node.type {
        case "PARALLEL": nodeType = .parallel
        case "OR", "COMPOUND": nodeType = .compound
        case "BASIC", "ATOMIC", "LEAF": nodeType = .atomic
        default:
             if let children = node.children, !children.isEmpty {
                 nodeType = .compound
             }
        }
        
        // Layout (Naive Auto-Layout basics)
        // For large charts, we need something better, but for now let's just stagger them so they aren't on top of each other.
        // Real layouting requires a pass *after* loading or a layout algorithm.
        // We'll increment Y for now.
        
        let flowNode = FlowNode(
            id: id,
            position: position, 
            label: node.label,
            type: nodeType,
            size: CGSize(width: 150, height: 80),
            parentID: parentID
        )
        result.append(flowNode)
        
        if let children = node.children {
            var childY = position.y + 100
            for child in children {
                let childPos = CGPoint(x: position.x + 50, y: childY)
                // Recursive
                let childNodes = visitStandardNode(child, parentID: id, currentPath: newPath, pathMap: &pathMap, position: childPos)
                result.append(contentsOf: childNodes)
                childY += 100 // Very naive
            }
        }
        
        return result
    }
    
    private static func mapStandardEdges(from transitions: [StandardTransition], pathMap: [String: UUID]) -> [FlowEdge] {
        var result: [FlowEdge] = []
        for t in transitions {
            let fromKey = t.from.joined(separator: "###")
            let toKey = t.to.joined(separator: "###")
            
            if let sourceID = pathMap[fromKey], let targetID = pathMap[toKey] {
                // Derive event name from available StandardTransition fields (if any)
                let event: String? = t.event
                
                result.append(FlowEdge(source: sourceID, target: targetID, event: event, guardExpression: t.guardDef?.expression, action: nil, routingType: .orthogonal))
            } else {
                // If direct match failed, maybe the path in transition excludes Root?
                // Try fuzzy matching or removing first element?
                // This is risky. Let's assume strict match for now.
                // Or maybe the transition path starts from Root's child?
                // In the JSON: "root_state" label is "__root__".
                // Transition from: ["SpriteInstances", ...].
                // So "__root__" is NOT in the transition path.
                // We need to check if the node path map includes root.
                // In flatten: we added root label.
                // So we should try matching with root prefix removed?
                
                 // Try removing root from keys in map, or check suffix?
            }
        }
        
        // Re-run with "Smart" matching if empty?
        if result.isEmpty && !transitions.isEmpty {
             // Fallback: Try to match ignoring the root label in our map
             // Actually, let's fix the map generation.
             // If root label is "__root__", maybe we just don't add it to the path for children?
        }
        
        return result
    }
    
    // MARK: - Stately Parsing
    
    private static func parseStatelyStatic(_ definition: StatelyDefinition) -> ([FlowNode], [FlowEdge]) {
         var localIDMap: [String: UUID] = [:]
         
         // 1. Flatten Nodes & Build ID Map
         let nodes = StatechartViewModel.parseNodes(
             rootNode: definition.rootNode,
             parentID: nil,
             idMap: &localIDMap
         )
         
         // 2. Map Edges using ID Map
         let edges = StatechartViewModel.mapEdges(from: definition.edges, idMap: localIDMap)
         
         return (nodes, edges)
    }


    
    // MARK: - Parsing Helpers
    
    private static func parseNodes(rootNode: StatelyNode, parentID: UUID?, idMap: inout [String: UUID]) -> [FlowNode] {
        var result: [FlowNode] = []
        
        // Sometimes the root node itself is a container we want to check,
        // but typically in Stately export, rootNode.nodes are the top-level states.
        // If rootNode is the "root", we don't render it directly usually?
        // Let's assume we start from rootNode's CHILDREN.
        
        if let children = rootNode.nodes {
            for child in children {
                result.append(contentsOf: visitNode(child, parentID: parentID, idMap: &idMap))
            }
        }
        return result
    }
    
    private static func visitNode(_ node: StatelyNode, parentID: UUID?, idMap: inout [String: UUID]) -> [FlowNode] {
        var nodes: [FlowNode] = []
        
        // Generate UUID
        let nodeUUID = UUID()
        idMap[node.id] = nodeUUID
        
        // Determine Type
        var nodeType: FlowNode.NodeType = .atomic
        if let type = node.type {
            switch type {
            case "final": nodeType = .final
            case "parallel": nodeType = .parallel
            case "compound": nodeType = .compound // explicit
            default:
                // Implicit type check: if it has children, it's compound
                if let children = node.nodes, !children.isEmpty {
                    nodeType = .compound
                }
            }
        } else if let children = node.nodes, !children.isEmpty {
            nodeType = .compound
        }
        
        // Position & Size
        let x = node.position?.x ?? 0
        let y = node.position?.y ?? 0
        let width = node.size?.width ?? 120
        let height = node.size?.height ?? 60
        
        let flowNode = FlowNode(
            id: nodeUUID,
            position: CGPoint(x: x, y: y),
            label: node.data?.key ?? node.id, // Fallback to ID if key missing
            type: nodeType,
            size: CGSize(width: width, height: height),
            parentID: parentID
        )
        nodes.append(flowNode)
        
        // Recurse for children
        if let children = node.nodes {
            for child in children {
                nodes.append(contentsOf: visitNode(child, parentID: nodeUUID, idMap: &idMap))
            }
        }
        
        return nodes
    }
    
    private static func mapEdges(from statelyEdges: [StatelyEdge], idMap: [String: UUID]) -> [FlowEdge] {
        var result: [FlowEdge] = []
        
        for edge in statelyEdges {
            if let sourceID = idMap[edge.source],
               let targetKey = edge.target,
               let targetID = idMap[targetKey] {
                
                let event = edge.data?.eventTypeData?.eventType
                // Stately edges might contain guard/actions in `data`?
                // For now just event.
                
                let flowEdge = FlowEdge(source: sourceID, target: targetID, event: event, routingType: .orthogonal)
                result.append(flowEdge)
            }
        }
        return result
    }
    
    // MARK: - View Helpers
    
    func zoomToFit(viewSize: CGSize) {
        guard viewSize.width > 0, viewSize.height > 0 else { return }
        guard !nodes.isEmpty else { return }
        
        let nodeRects = nodes.map { CGRect(origin: $0.position, size: $0.size) }
        let boundingBox = nodeRects.reduce(nodeRects[0]) { $0.union($1) }
        
        // Add padding (Generous padding for HUDs)
        let padding: CGFloat = 100
        let paddedRect = boundingBox.insetBy(dx: -padding, dy: -padding)
        
        guard paddedRect.width > 0, paddedRect.height > 0 else { return }
        
        let scaleX = viewSize.width / paddedRect.width
        let scaleY = viewSize.height / paddedRect.height
        
        // Clamp initial fit
        let fitScale = min(scaleX, scaleY, 1.2) // Don't zoom in too much if chart is tiny
        
        withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
            self.scale = fitScale
            
            let contentCenter = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
            self.offset = CGSize(
                width: -contentCenter.x * fitScale,
                height: -contentCenter.y * fitScale
            )
        }
    }
    
    func centerNode(id: UUID, viewSize: CGSize) {
        guard let node = nodes.first(where: { $0.id == id }) else { return }
        
        // Calculate offset to center this node
        // center of node
        let nodeCenter = CGPoint(x: node.position.x + node.size.width / 2, y: node.position.y + node.size.height / 2)
        
        // We want (nodeCenter * scale) + offset = (viewSize / 2)
        // offset = (viewSize / 2) - (nodeCenter * scale)
        
        withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
            self.offset = CGSize(
                width: (viewSize.width / 2) - (nodeCenter.x * self.scale),
                height: (viewSize.height / 2) - (nodeCenter.y * self.scale)
            )
        }
    }
    
    // MARK: - Undo/Redo Support
    var undoManager: UndoManager?
    
    // MARK: - Editing Actions
    
    func addState(at position: CGPoint) {
        let newNode = FlowNode(id: UUID(), position: position, label: "New State", type: .atomic)
        nodes.append(newNode)
        selection = [newNode.id]

        #if os(iOS)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        #endif

        // Register Undo: Remove the added node
        undoManager?.registerUndo(withTarget: self) { target in
            target.deleteNode(id: newNode.id)
        }
    }
    
    func deleteSelection() {
        // Identify nodes to delete (intersection of nodes and selection)
        let nodesToDelete = nodes.filter { selection.contains($0.id) }
        let nodeIDsToDelete = Set(nodesToDelete.map { $0.id })

        // Identify edges to delete (connected to these nodes OR selected themselves)
        let edgesToDelete = edges.filter { edge in
            selection.contains(edge.id) ||
            nodeIDsToDelete.contains(edge.source) ||
            nodeIDsToDelete.contains(edge.target)
        }

        guard !nodesToDelete.isEmpty || !edgesToDelete.isEmpty else { return }

        // Perform Deletion (Action)
        nodes.removeAll { selection.contains($0.id) }
        edges.removeAll { edgesToDelete.contains($0) }

        selection.removeAll()

        #if os(iOS)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        #endif

        // Register Undo: Restore nodes and edges
        undoManager?.registerUndo(withTarget: self) { target in
            target.nodes.append(contentsOf: nodesToDelete)
            target.edges.append(contentsOf: edgesToDelete)
            target.selection = Set(nodesToDelete.map { $0.id }).union(edgesToDelete.map { $0.id })
        }
    }
    
    // MARK: - Marquee Logic
    
    func startMarquee(at startPoint: CGPoint, isAdditive: Bool) {
        if !isAdditive {
            selection.removeAll()
        }
        initialSelectionBeforeDrag = selection
        // selectionRect starts as zero-size at startPoint (in Logic Space)
        // But the View handles the rect creation usually. 
        // We just need to know we are starting.
    }
    
    func updateMarquee(rect: CGRect) {
        self.selectionRect = rect
        
        // Find nodes intersecting rect
        let hitNodes = nodes.filter { node in
            let nodeRect = CGRect(origin: node.position, size: node.size)
            return rect.intersects(nodeRect)
        }
        
        let hitIDs = Set(hitNodes.map { $0.id })
        
        // Combine with initial selection
        // Typical behavior: Shift adds/toggles? 
        // For simplicity: Union with initial.
        // If we want "Rubber Band Select" usually it selects what is in the box.
        // If Shift held, we add to what was selected BEFORE drag.
        
        selection = initialSelectionBeforeDrag.union(hitIDs)
    }
    
    func endMarquee() {
        self.selectionRect = nil
        self.initialSelectionBeforeDrag = []
    }
    
    // Internal helper for undoing an add
    // Made public and robust for Context Menu usage
    func deleteNode(id: UUID) {
        guard let index = nodes.firstIndex(where: { $0.id == id }) else { return }
        let node = nodes[index]
        
        // Find connected edges
        let connectedEdges = edges.filter { $0.source == id || $0.target == id }
        
        // Perform Delete
        nodes.remove(at: index)
        edges.removeAll { connectedEdges.contains($0) }
        
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        #endif
        
        // Register Undo
        undoManager?.registerUndo(withTarget: self) { target in
            target.nodes.append(node)
            target.edges.append(contentsOf: connectedEdges)
            
            // Redo
            target.undoManager?.registerUndo(withTarget: target) { target in
                target.deleteNode(id: id)
            }
        }
    }
    
    func duplicateSelection() {
        for id in selection {
            duplicateNode(id: id)
        }
    }
    
    func duplicateNode(id: UUID) {
        guard let node = nodes.first(where: { $0.id == id }) else { return }
        
        let offset = CGPoint(x: 20, y: 20)
        let newPos = CGPoint(x: node.position.x + offset.x, y: node.position.y + offset.y)
        
        let newNode = FlowNode(
            id: UUID(),
            position: newPos,
            label: "\(node.label) Copy",
            type: node.type,
            size: node.size,
            parentID: node.parentID
        )
        
        nodes.append(newNode)
        selection = [newNode.id]
        
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        #endif
        
        undoManager?.registerUndo(withTarget: self) { target in
            target.deleteNode(id: newNode.id)
        }
    }
    
    // Undo for movement
    func registerMoveUndo(oldPositions: [UUID: CGPoint]) {
        // Capture current positions for Redo
        let currentPositions = nodes.reduce(into: [UUID: CGPoint]()) { dict, node in
            if oldPositions.keys.contains(node.id) {
                dict[node.id] = node.position
            }
        }
        
        undoManager?.registerUndo(withTarget: self) { target in
            target.restorePositions(oldPositions)
            
            // Register Redo
            target.undoManager?.registerUndo(withTarget: target) { target in
                target.restorePositions(currentPositions)
            }
        }
    }
    
    private func restorePositions(_ positions: [UUID: CGPoint]) {
        for (id, pos) in positions {
            if let index = nodes.firstIndex(where: { $0.id == id }) {
                nodes[index].position = pos
            }
        }
    }
    
    func registerRenameUndo(id: UUID, oldLabel: String) {
        guard let index = nodes.firstIndex(where: { $0.id == id }) else { return }
        let currentLabel = nodes[index].label
        
        undoManager?.registerUndo(withTarget: self) { target in
            target.renameNode(id: id, label: oldLabel)
            
            // Redo
            target.undoManager?.registerUndo(withTarget: target) { target in
                target.renameNode(id: id, label: currentLabel)
            }
        }
    }
    
    private func renameNode(id: UUID, label: String) {
        if let index = nodes.firstIndex(where: { $0.id == id }) {
            nodes[index].label = label
        }
    }

    func addSubstate(parentID: UUID) {
        // Find parent to get position estimate
        guard let parent = nodes.first(where: { $0.id == parentID }) else { return }
        
        let position = CGPoint(x: parent.position.x + 20, y: parent.position.y + 60)
        let newNode = FlowNode(id: UUID(), position: position, label: "Substate", type: .atomic, parentID: parentID)
        
        nodes.append(newNode)
        selection = [newNode.id]
        
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        #endif
        
        undoManager?.registerUndo(withTarget: self) { target in
            target.deleteNode(id: newNode.id)
        }
    }
    
    func reparent(childID: UUID, newParentID: UUID?) {
        guard let index = nodes.firstIndex(where: { $0.id == childID }) else { return }
        
        let oldParentID = nodes[index].parentID
        if oldParentID == newParentID { return } // No change
        
        nodes[index].parentID = newParentID
        
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        #endif
        
        undoManager?.registerUndo(withTarget: self) { target in
            target.reparent(childID: childID, newParentID: oldParentID)
        }
    }
}

