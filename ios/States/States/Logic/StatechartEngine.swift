//
//  StatechartEngine.swift
//  States
//
//  Created by Assistant on 1/3/25.
//

import Foundation
import Combine

class StatechartEngine: ObservableObject {
    
    // --- State ---
    @Published var activeStateIDs: Set<UUID> = []
    @Published var context: [String: Any] = [:]
    
    // --- Configuration ---
    private var nodes: [FlowNode]
    private var edges: [FlowEdge]
    
    // --- Helpers ---
    private let guardEvaluator = GuardEvaluator()
    private var history: [[Set<UUID>]] = [] // For simple undo/history tracking
    
    init(nodes: [FlowNode], edges: [FlowEdge]) {
        self.nodes = nodes
        self.edges = edges
        self.context = [:] // Start empty
    }
    
    // Update graph definition (e.g. after edit)
    func updateDefinition(nodes: [FlowNode], edges: [FlowEdge]) {
        self.nodes = nodes
        self.edges = edges
    }
    
    // MARK: - Lifecycle
    
    func start() {
        // Enter initial states
        // Strategy: Find root-most atomic states or specific initial markers.
        // Naive 1: Find first Atomic node.
        if let firstAtomic = nodes.first(where: { $0.type == .atomic }) {
            enterState(firstAtomic.id)
        }
    }
    
    func sendEvent(_ eventName: String) {
        // 1. Find valid transitions from active states
        // Harel semantics: Check transitions from active states, respecting hierarchy priority (Child overrides Parent).
        // Since our FlowGraph is flat nodes with ParentID, we can check all edges where source is Active.
        
        let activeEdges = edges.filter { edge in
            activeStateIDs.contains(edge.source) && (edge.event == eventName)
        }
        
        // 2. Evaluate Guards
        var transitionsToTake: [FlowEdge] = []
        
        for edge in activeEdges {
            // Check Guard
            if let guardExpr = edge.guardExpression, !guardExpr.isEmpty {
                if guardEvaluator.evaluate(expression: guardExpr, context: context) {
                    transitionsToTake.append(edge)
                }
            } else {
                // No guard = pass
                transitionsToTake.append(edge)
            }
        }
        
        // 3. Selection Strategy (Priority)
        // If multiple transitions are valid, which one to take?
        // - Determinism: usually prefer inner-most source state (Hierarchy).
        // - Doc order?
        // For visualizer currently: take FIRST valid.
        
        if let chosen = transitionsToTake.first {
            executeTransition(chosen)
        } else {
            print("Event '\(eventName)' ignored (guard failed or no match).")
        }
    }
    
    // MARK: - Execution
    
    private func executeTransition(_ edge: FlowEdge) {
        // 1. Exit Source
        // If atomic, exit. If composite, exit children too?
        // For flat visualizer model: we just toggle IDs.
        // Proper semantics: exit hierarchy up to LCA (Least Common Ancestor).
        
        // Simplified Step:
        exitState(edge.source)
        
        // 2. Execute Action
        if let actionScript = edge.action, !actionScript.isEmpty {
            if let newContext = guardEvaluator.executeAction(script: actionScript, context: context) {
                self.context = newContext
            }
        }
        
        // 3. Enter Target
        enterState(edge.target)
    }
    
    private func enterState(_ id: UUID) {
        if activeStateIDs.contains(id) { return }
        
        activeStateIDs.insert(id)
        
        // Find Node
        guard let node = nodes.first(where: { $0.id == id }) else { return }
        
        // TODO: Enter actions
        
        // Handling Children
        // 1. If Parallel: Enter ALL children (Fork)
        // 2. If Compound (normal): Enter Initial child (if defined)
        
        let children = nodes.filter { $0.parentID == id }
        
        if node.type == .parallel {
            for child in children {
                enterState(child.id)
            }
        } else if !children.isEmpty {
            // Compound: Enter initial state
            // Logic: Is there a child marked "initial"? Or separate Initial pseudo-state?
            // Current FlowNode doesn't have isInitial flag directly visible in simple struct, 
            // but we might infer from "Standard" def or just pick first for now.
            // A better way: check incoming edges from an Initial pseudo-node?
            // For now: if NO child is active, pick first.
            
            let activeChildren = children.filter { activeStateIDs.contains($0.primaryKey) } // primaryKey not avail? -> id
            let anyChildActive = children.contains { activeStateIDs.contains($0.id) }
            
            if !anyChildActive {
                 // Try to find one that looks like "initial" or default to first
                 if let first = children.first {
                     enterState(first.id)
                 }
            }
        }
    }
    
    private func exitState(_ id: UUID) {
        if !activeStateIDs.contains(id) { return }
        
        activeStateIDs.remove(id)
        // TODO: Exit actions
        
        // Cascade Exit: Exit all children
        let children = nodes.filter { $0.parentID == id }
        for child in children {
            if activeStateIDs.contains(child.id) {
                exitState(child.id)
            }
        }
    }
}
