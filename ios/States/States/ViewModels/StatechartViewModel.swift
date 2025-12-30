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
        
        // Reset state
        self.nodes = []
        self.edges = []
        self.selection = []
        self.activeStateIDs = []
        self.simulationHistory = []
        self.currentStepIndex = 0
        self.mode = .editing
        
        guard let json = machine.jsonContent, !json.isEmpty, let data = json.data(using: .utf8) else {
            zoomToFit(viewSize: CGSize(width: 800, height: 600))
            return
        }
            
        // Attempt 0: Protobuf
        if let proto = machine.proto {
            parseProtoDefinition(proto)
            zoomToFit(viewSize: CGSize(width: 800, height: 600))
            return
        }
            
        // Attempt 1: Standard SC Format
        do {
            let definition = try JSONDecoder().decode(StandardStatechart.self, from: data)
            parseStandardDefinition(definition)
        } catch let standardError {
            // Attempt 2: Stately Format
            do {
                let definition = try JSONDecoder().decode(StatelyDefinition.self, from: data)
                parseStatelyDefinition(definition)
            } catch let statelyError {
                 print("Error decoding Standard JSON: \(standardError)")
                 print("Error decoding Stately JSON: \(statelyError)")
                 
                let errorID = UUID()
                self.nodes = [
                    FlowNode(id: errorID, position: CGPoint(x: 100, y: 100), label: "Error Parsing JSON: \(standardError.localizedDescription) / \(statelyError.localizedDescription)", type: .atomic)
                ]
            }
        }
        
        // Initial fit
        zoomToFit(viewSize: CGSize(width: 800, height: 600))
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
        }
    }
    
    func stepBack() {
        guard currentStepIndex > 0 else { return }
        currentStepIndex -= 1
        activeStateIDs = simulationHistory[currentStepIndex]
    }
    
    func stepForward() {
        guard currentStepIndex < simulationHistory.count - 1 else { return }
        currentStepIndex += 1
        activeStateIDs = simulationHistory[currentStepIndex]
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
        var nextActiveIDs = activeStateIDs
        var handled = false
        
        // Find outgoing edges from currently active states that match eventName
        // Note: This is a naive implementation. Harel statecharts have complex priority rules (depth-first, etc.)
        // For visualizer parity, we initially support simple atomic transitions.
        
        let validEdges = edges.filter { edge in
            activeStateIDs.contains(edge.source) && edge.label == eventName
        }
        
        for edge in validEdges {
            // Traverse
            // 1. Exit source (if atomic) or just switch?
            // Simple model: remove source, add target.
            // Assumption: transitions are atomic-to-atomic or compound-to-atomic?
            // If source is active, we leave it.
            if nextActiveIDs.contains(edge.source) {
                nextActiveIDs.remove(edge.source)
                nextActiveIDs.insert(edge.target)
                handled = true
            }
        }
        
        if handled {
            // Record History
            if currentStepIndex < simulationHistory.count - 1 {
                simulationHistory = Array(simulationHistory.prefix(currentStepIndex + 1))
            }
            activeStateIDs = nextActiveIDs
            simulationHistory.append(nextActiveIDs)
            currentStepIndex = simulationHistory.count - 1
        } else {
            print("Event '\(eventName)' ignored (no valid transitions from current state).")
        }
    }
    
    // MARK: - Protobuf Parsing
    
    private func parseProtoDefinition(_ definition: Statecharts_V1_Statechart) {
        var pathMap: [String: UUID] = [:]
        
        // 1. Flatten Nodes
        if definition.hasRootState {
            self.nodes = StatechartViewModel.visitProtoNode(
                definition.rootState,
                parentID: nil,
                currentPath: [],
                pathMap: &pathMap,
                parentOrigin: .zero,
                suggestedOffset: CGPoint(x: 50, y: 50)
            )
        }
        
        // 2. Map Edges
        self.edges = StatechartViewModel.mapProtoEdges(from: definition.transitions, pathMap: pathMap)
        
        // 3. Set Initial Active State
        if let firstAtomic = self.nodes.first(where: { $0.type == .atomic }) {
            self.activeStateIDs.insert(firstAtomic.id)
        }
        self.simulationHistory = [self.activeStateIDs]
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
        var usedExplicitLayout = false
        
        // Attempt to extract Extensions_V1_StateLayout
        for ext in node.extensions {
            if let layout = try? Extensions_V1_StateLayout(unpackingAny: ext) {
                if layout.hasPosition {
                    finalPosition = CGPoint(x: parentOrigin.x + layout.position.x, y: parentOrigin.y + layout.position.y)
                    usedExplicitLayout = true
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
                let label = t.event
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
                
                result.append(FlowEdge(source: sourceID, target: targetID, label: label, routingType: routingType, waypoints: waypoints))
            }
        }
        return result
    }
    
    // MARK: - Standard Format Parsing
    
    private func parseStandardDefinition(_ definition: StandardStatechart) {
        var pathMap: [String: UUID] = [:] // "root.Child" -> UUID
        
        // 1. Flatten Nodes & Build Path Map
        // Root label usually ignored in path for children? Or implied?
        // Let's assume path is based on labels.
        self.nodes = StatechartViewModel.visitStandardNode(
            definition.rootState,
            parentID: nil,
            currentPath: [],
            pathMap: &pathMap,
            position: CGPoint(x: 50, y: 50) 
        )
        
        // 2. Map Edges
        if let transitions = definition.transitions {
            self.edges = StatechartViewModel.mapStandardEdges(from: transitions, pathMap: pathMap)
        }
        
        // 3. Set Initial Active State
        if let firstAtomic = self.nodes.first(where: { $0.type == .atomic }) {
            self.activeStateIDs.insert(firstAtomic.id)
        }
        self.simulationHistory = [self.activeStateIDs]
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
                let label = t.guardDef?.expression ?? ""
                result.append(FlowEdge(source: sourceID, target: targetID, label: label, routingType: .orthogonal))
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
    
    private func parseStatelyDefinition(_ definition: StatelyDefinition) {
         var localIDMap: [String: UUID] = [:]
         var parsedNodes: [FlowNode] = []
         
         // 1. Flatten Nodes & Build ID Map
         parsedNodes = StatechartViewModel.parseNodes(
             rootNode: definition.rootNode,
             parentID: nil,
             idMap: &localIDMap
         )
         
         self.nodes = parsedNodes
         
         // 2. Map Edges using ID Map
         self.edges = StatechartViewModel.mapEdges(from: definition.edges, idMap: localIDMap)
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
                
                let label = edge.data?.eventTypeData?.eventType ?? ""
                let flowEdge = FlowEdge(source: sourceID, target: targetID, label: label, routingType: .orthogonal)
                result.append(flowEdge)
            }
        }
        return result
    }
    
    // MARK: - View Helpers
    
    func zoomToFit(viewSize: CGSize) {
        guard !nodes.isEmpty else { return }
        
        let nodeRects = nodes.map { CGRect(origin: $0.position, size: $0.size) }
        let boundingBox = nodeRects.reduce(nodeRects[0]) { $0.union($1) }
        
        // Add padding
        let paddedRect = boundingBox.insetBy(dx: -50, dy: -50)
        
        let scaleX = viewSize.width / paddedRect.width
        let scaleY = viewSize.height / paddedRect.height
        let fitScale = min(scaleX, scaleY, 1.0)
        
        self.scale = fitScale
        let contentCenter = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
        let viewCenter = CGPoint(x: viewSize.width/2, y: viewSize.height/2)
        
        self.offset = CGSize(
            width: viewCenter.x - contentCenter.x * fitScale,
            height: viewCenter.y - contentCenter.y * fitScale
        )
    }
    
    // MARK: - Undo/Redo Support
    var undoManager: UndoManager?
    
    // MARK: - Editing Actions
    
    func addState(at position: CGPoint) {
        let newNode = FlowNode(id: UUID(), position: position, label: "New State", type: .atomic)
        nodes.append(newNode)
        selection = [newNode.id]
        
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
        
        // Perform Deletion (Action)
        nodes.removeAll { selection.contains($0.id) }
        edges.removeAll { edgesToDelete.contains($0) } // Equatable check or ID check if FlowEdge is Equatable
        
        selection.removeAll()
        
        // Register Undo: Restore nodes and edges
        undoManager?.registerUndo(withTarget: self) { target in
            target.nodes.append(contentsOf: nodesToDelete)
            target.edges.append(contentsOf: edgesToDelete)
            // Restore selection? Optional, but nice.
            target.selection = Set(nodesToDelete.map { $0.id }).union(edgesToDelete.map { $0.id })
        }
    }
    
    // Internal helper for undoing an add
    private func deleteNode(id: UUID) {
        if let index = nodes.firstIndex(where: { $0.id == id }) {
            let node = nodes[index]
            nodes.remove(at: index)
            
            // Register Redo: Add it back
            undoManager?.registerUndo(withTarget: self) { target in
                target.nodes.append(node)
                // If we redo an add, we should probably set selection to it
                target.selection = [node.id]
            }
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
    
}
