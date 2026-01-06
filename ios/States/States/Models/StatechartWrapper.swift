import Foundation
import SwiftProtobuf

// NOTE: This assumes the generated protobuf files are added to the Xcode target.
// You might need to add `import Statecharts` or similar if it's in a separate module.

struct StatechartWrapper: Identifiable, Hashable {
    let id: UUID
    let name: String
    
    var jsonContent: String?
    
    var proto: Statecharts_V1_Statechart? {
        guard let jsonContent = jsonContent else { return nil }
        // Attempt to decode as Protobuf JSON
        do {
            var options = JSONDecodingOptions()
            options.ignoreUnknownFields = true
            return try Statecharts_V1_Statechart(jsonString: jsonContent, options: options)
        } catch {
            // Only print if it's not a known alternative format
            // print("Proto decode failed: \(error)")
            return nil
        }
    }
    
    init(name: String, jsonContent: String? = nil) {
        self.id = UUID()
        self.name = name
        self.jsonContent = jsonContent
    }
}
import Foundation
import CoreGraphics

// MARK: - Stately JSON Structures

struct StatelyDocsMachine: Codable {
    let id: String
    let name: String
    let definition: StatelyDefinition
}

struct StatelyDefinition: Codable {
    let id: String
    let rootNode: StatelyNode
    let edges: [StatelyEdge]
}

struct StatelyNode: Codable {
    let id: String
    let data: StatelyNodeData?
    let nodes: [StatelyNode]?
    let position: StatelyPosition?
    let size: StatelySize?
    let type: String? // "parallel", "final", etc.
}

struct StatelyNodeData: Codable {
    let key: String? // The label often used
    let initial: String?
    let type: String?
}

struct StatelyPosition: Codable {
    let x: Double
    let y: Double
}

struct StatelySize: Codable {
    let width: Double
    let height: Double
}

struct StatelyEdge: Codable {
    let id: String
    let source: String
    let target: String?
    let data: StatelyEdgeData?
}

struct StatelyEdgeData: Codable {
    let eventTypeData: StatelyEventTypeData?
}

struct StatelyEventTypeData: Codable {
    let eventType: String?
}
