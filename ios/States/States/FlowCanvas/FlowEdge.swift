import Foundation

public struct FlowEdge: Identifiable, Equatable {
    public let id: UUID
    public let source: UUID
    public let target: UUID
    
    public var event: String?
    public var guardExpression: String?
    public var action: String?
    
    public var label: String? {
        var parts: [String] = []
        if let e = event, !e.isEmpty { parts.append(e) }
        if let g = guardExpression, !g.isEmpty { parts.append("[\(g)]") }
        if let a = action, !a.isEmpty { parts.append("/ \(a)") }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }
    
    public enum RoutingType: Equatable {
        case straight
        case orthogonal
        case curved
    }
    
    public var routingType: RoutingType
    public var waypoints: [CGPoint]?
    
    public init(id: UUID = UUID(), source: UUID, target: UUID, event: String? = nil, guardExpression: String? = nil, action: String? = nil, routingType: RoutingType = .orthogonal, waypoints: [CGPoint]? = nil) {
        self.id = id
        self.source = source
        self.target = target
        self.event = event
        self.guardExpression = guardExpression
        self.action = action
        self.routingType = routingType
        self.waypoints = waypoints
    }
}
