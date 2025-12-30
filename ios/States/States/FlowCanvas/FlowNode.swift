import Foundation
import CoreGraphics

public struct FlowNode: Identifiable, Equatable {
    public let id: UUID
    public var position: CGPoint
    public var label: String
    public var type: NodeType
    public var isSelected: Bool = false
    
    public var parentID: UUID?
    public var size: CGSize
    
    public enum NodeType: String, Codable {
        case atomic, compound, parallel, final, history
    }
    
    public init(id: UUID = UUID(), position: CGPoint, label: String, type: NodeType = .atomic, size: CGSize = CGSize(width: 120, height: 60), parentID: UUID? = nil) {
        self.id = id
        self.position = position
        self.label = label
        self.type = type
        self.size = size
        self.parentID = parentID
    }
}
