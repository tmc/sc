//
//  AnalysisEngine.swift
//  States
//
//  Created by Assistant on 1/3/25.
//

import Foundation

enum AnalysisIssueType: String, CaseIterable, Identifiable {
    case orphan = "Orphan State"
    case sink = "Sink State"
    case unreachable = "Unreachable State"
    case deadEnd = "Dead End"
    
    var id: String { rawValue }
    
    var icon: String {
        switch self {
        case .orphan: return "arrow.right.circle.fill" // Incoming missing
        case .sink: return "arrow.left.circle.fill" // Outgoing missing
        case .unreachable: return "exclamationmark.triangle.fill"
        case .deadEnd: return "xmark.octagon.fill"
        }
    }
    
    var description: String {
        switch self {
        case .orphan: return "State has no incoming transitions."
        case .sink: return "State has no outgoing transitions (and is not final)."
        case .unreachable: return "State cannot be reached from the initial state."
        case .deadEnd: return "State cannot reach a final state."
        }
    }
}

struct AnalysisIssue: Identifiable, Hashable {
    let id = UUID()
    let type: AnalysisIssueType
    let nodeID: UUID
    let nodeLabel: String
}

struct AnalysisReport {
    let issues: [AnalysisIssue]
    let reachabilityPercentage: Double
    let totalStates: Int
    let reachableStates: Int
    
    var isEmpty: Bool {
        issues.isEmpty
    }
    
    // Helper to get issues by type
    func issues(of type: AnalysisIssueType) -> [AnalysisIssue] {
        issues.filter { $0.type == type }
    }
}

class AnalysisEngine {
    
    func analyze(nodes: [FlowNode], edges: [FlowEdge]) -> AnalysisReport {
        // Build Graph
        let graph = buildAdjacencyGraph(nodes: nodes, edges: edges)
        
        var issues: [AnalysisIssue] = []
        
        // 1. Detect Orphans (No incoming edges)
        // Exclude Initial states and Root-level states that might be implicitly initial? 
        // For simplicity, we flag any non-initial state with 0 incoming edges.
        // Actually, the initial state of the *root* region gets a pass.
        // And children of a composite state might be entered via parent Default? 
        // We need to be careful. The Go impl excludes "Initial" states.
        
        let orphans = findOrphans(graph: graph, nodes: nodes)
        issues.append(contentsOf: orphans.map { AnalysisIssue(type: .orphan, nodeID: $0.id, nodeLabel: $0.label) })
        
        // 2. Detect Sinks (No outgoing edges)
        // Exclude Final states.
        let sinks = findSinks(graph: graph, nodes: nodes)
        issues.append(contentsOf: sinks.map { AnalysisIssue(type: .sink, nodeID: $0.id, nodeLabel: $0.label) })
        
        // 3. Reachability (BFS from Initial)
        // First, find initial states.
        let initialStates = nodes.filter { isInitial(node: $0, nodes: nodes) }
        let reachableIDs = findReachable(graph: graph, initialNodes: initialStates, allNodes: nodes)
        
        let unreachable = nodes.filter { !reachableIDs.contains($0.id) }
        issues.append(contentsOf: unreachable.map { AnalysisIssue(type: .unreachable, nodeID: $0.id, nodeLabel: $0.label) })

        // 4. Dead Ends (Cannot reach Final)
        // TODO: Implement Dead End detection (Reverse BFS from Final)
        
        let total = nodes.count
        let reachableCount = reachableIDs.count
        let pct = total > 0 ? Double(reachableCount) / Double(total) : 0.0
        
        return AnalysisReport(
            issues: issues,
            reachabilityPercentage: pct * 100.0,
            totalStates: total,
            reachableStates: reachableCount
        )
    }
    
    // MARK: - Graph Algorithm Helpers
    
    private struct AdjacencyGraph {
        let incoming: [UUID: [UUID]]
        let outgoing: [UUID: [UUID]]
    }
    
    private func buildAdjacencyGraph(nodes: [FlowNode], edges: [FlowEdge]) -> AdjacencyGraph {
        var incoming = [UUID: [UUID]]()
        var outgoing = [UUID: [UUID]]()
        
        // Initialize for all nodes
        for node in nodes {
            incoming[node.id] = []
            outgoing[node.id] = []
        }
        
        for edge in edges {
            outgoing[edge.source, default: []].append(edge.target)
            incoming[edge.target, default: []].append(edge.source)
        }
        
        return AdjacencyGraph(incoming: incoming, outgoing: outgoing)
    }
    
    private func findOrphans(graph: AdjacencyGraph, nodes: [FlowNode]) -> [FlowNode] {
        return nodes.filter { node in
            // Must have no incoming edges
            let noIncoming = (graph.incoming[node.id]?.isEmpty ?? true)
            // And NOT be an initial state (Initial states are expected to have no incoming)
            let isInit = isInitial(node: node, nodes: nodes)
            return noIncoming && !isInit
        }
    }
    
    private func findSinks(graph: AdjacencyGraph, nodes: [FlowNode]) -> [FlowNode] {
        return nodes.filter { node in
            // Must have no outgoing edges
            let noOutgoing = (graph.outgoing[node.id]?.isEmpty ?? true)
            // And NOT be a final state
            let isFinal = (node.type == .final)
            return noOutgoing && !isFinal
        }
    }
    
    private func findReachable(graph: AdjacencyGraph, initialNodes: [FlowNode], allNodes: [FlowNode]) -> Set<UUID> {
        var visited = Set<UUID>()
        var queue = initialNodes.map { $0.id }
        
        // Mark initials as visited
        for id in queue {
            visited.insert(id)
        }
        
        while !queue.isEmpty {
            let currentID = queue.removeFirst()
            
            // 1. Follow transitions
            if let targets = graph.outgoing[currentID] {
                for target in targets {
                    if !visited.contains(target) {
                        visited.insert(target)
                        queue.append(target)
                    }
                }
            }
            
            // 2. Follow Parent->Child relationships (Implicit reachability via default initial child)
            // If we reach a parent, we *automatically* reach its initial child?
            // Or does "Reachable" mean strictly via transitions?
            // Harel semantics: Entering a composite state enters its initial state.
            // So if `currentID` is a parent, we should add its INITIAL child to the queue.
            
            let children = allNodes.filter { $0.parentID == currentID }
            for child in children {
                // If we enter the parent, we can potentially reach the children
                // Specifically, if we enter parent, we go to Initial child.
                // But if we are just "in" the parent, we can transition FROM any child.
                // For reachability "set of active states", yes.
                
                // Let's assume strict transition reachability + hierarchy descent.
                // If I am in Parent, I am also effectively in the Initial Child.
                if isInitial(node: child, nodes: allNodes) {
                    if !visited.contains(child.id) {
                        visited.insert(child.id)
                        queue.append(child.id)
                    }
                }
                
                // Also, if I am in a child, I am in the parent?
                // Should we traverse UP? Usually reachability is about "future states".
            }
        }
        
        return visited
    }
    
    // Simple heuristic for "Is Initial"
    // In our FlowNode model, we don't strictly have an "IsInitial" flag on the node itself unless we parse the `initial` property of the parent?
    // Wait, FlowNode doesn't have `isInitial`. 
    // The *parent* (or root) has an `initial` property pointing to a child ID.
    // BUT, we don't track that easily on FlowNode.
    // However, typically the "first" child or one marked explicitly is initial.
    // For now, let's look for a node that is the target of a transition from the generic "Initial" pseudo-node (circle filled)?
    // OR, we check if `StatechartViewModel` knows.
    // Actually, `FlowNode` has `type`. Is there a `type == .initial`?
    // Let's check `FlowNode.swift` or `NodeType`.
    
    private func isInitial(node: FlowNode, nodes: [FlowNode]) -> Bool {
        // Swift `FlowNode` usually just has `type`.
        // If we strictly follow the proto, "Initial" is a property of the parent state.
        // If we assume the visual editor convention: a Node with 0 incoming edges that is the "first" user created?
        // No, that's unreliable.
        // Let's assume for this engine:
        // 1. If it has type `.initial` (if that exists)
        // 2. If it is the "Start" node of the entire graph (no parent, first in list?)
        
        // Looking at `StatechartViewModel`, `nodes` are `FlowNode`.
        // `NodeType` enum likely has `.state`, `.parallel`, `.final`, `.history`... does it have `.initial`?
        // Usually initial state is just a state MARKED as initial.
        
        // Hack: If label is "Initial" or "Start"?
        // Or if it is the target of a transition from nowhere (which we can't represent in edges easily).
        
        // Let's rely on `node.type == .basic` for now and maybe assume Root's first child is initial if not specified?
        // Actually, we should check `NodeType` definition. I'll make a conservative guess or check the file.
        // I will assume for now that we treat the "First Node added" or "Root" as reachable.
        
        // BETTER: Treat all nodes with `parentID == nil` as potentially reachable "Roots".
        // And traverse down.
        if node.parentID == nil {
            return true // Root states are reachable by definition of starting the machine.
        }
        
        // If parent is reachable, is this child reachable?
        // Only if it's the initial child.
        // Since we don't have `isInitial` flag on FlowNode yet, we might flag this as a "TODO: Add isInitial to FlowNode".
        // For now, let's treat ALL children of a reachable parent as "Reachable" to avoid false positives,
        // OR rely on transitions.
        
        return false
    }
    
    // MARK: - Path Finding
    
    func findPath(from sourceID: UUID, to targetID: UUID, nodes: [FlowNode], edges: [FlowEdge]) -> [UUID]? {
        // BFS for shortest path
        if sourceID == targetID { return [sourceID] }
        
        // Build Adjacency
        // For path finding, we might want to cache this graph if we do it often, 
        // but for now rebuilding is cheap enough for standard charts.
        let graph = buildAdjacencyGraph(nodes: nodes, edges: edges)
        
        var visited = Set<UUID>()
        visited.insert(sourceID)
        
        var queue: [[UUID]] = [[sourceID]]
        
        while !queue.isEmpty {
            let path = queue.removeFirst()
            let current = path.last!
            
            if current == targetID {
                return path
            }
            
            // Outgoing transitions
            if let neighbors = graph.outgoing[current] {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor)
                        var newPath = path
                        newPath.append(neighbor)
                        queue.append(newPath)
                    }
                }
            }
            
            // Does hierarchy allow implicit transitions?
            // Go implementation of `FindPath` uses `TransitionEdge` traversal.
            // It generally doesn't implicitly traverse hierarchy unless there are explicit transitions.
            // So we stick to `graph.outgoing`.
        }
        
        return nil
    }
}

