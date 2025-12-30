import Foundation

public struct FlowEdge: Identifiable, Equatable {
    public let id: UUID
    public let source: UUID
    public let target: UUID
    public var label: String?
    
    public enum RoutingType: Equatable {
        case straight
        case orthogonal
        case curved
    }
    
    public var routingType: RoutingType
    public var waypoints: [CGPoint]?
    
    public init(id: UUID = UUID(), source: UUID, target: UUID, label: String? = nil, routingType: RoutingType = .orthogonal, waypoints: [CGPoint]? = nil) {
        self.id = id
        self.source = source
        self.target = target
        self.label = label
        self.routingType = routingType
        self.waypoints = waypoints
    }
}
