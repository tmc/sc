import Foundation

// A Standard JSON representation for Statecharts (SC Format)
struct StandardStatechart: Codable {
    let name: String
    let rootState: StandardState
    let transitions: [StandardTransition]?
    
    enum CodingKeys: String, CodingKey {
        case name
        case rootState = "root_state"
        case transitions
    }
}

struct StandardState: Codable {
    let label: String
    let type: String? // "PARALLEL", "OR", "BASIC"
    let children: [StandardState]?
    let isInitial: Bool?
    
    enum CodingKeys: String, CodingKey {
        case label
        case type
        case children
        case isInitial = "is_initial"
    }
}

struct StandardTransition: Codable {
    let from: [String]
    let to: [String]
    let event: String?
    let guardDef: StandardGuard?
    
    enum CodingKeys: String, CodingKey {
        case from
        case to
        case event
        case guardDef = "guard"
    }
}

struct StandardGuard: Codable {
    let expression: String?
}
