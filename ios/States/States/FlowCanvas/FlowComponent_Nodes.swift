import SwiftUI

struct FlowNodesLayer: View {
    @Binding var nodes: [FlowNode]
    @Binding var selection: Set<UUID>
    @Binding var activeStateIDs: Set<UUID>
    var hoveredNodeID: UUID?
    var hoveredParentID: UUID?
    @Binding var scale: CGFloat
    @Binding var offset: CGSize
    var currentDragOffset: CGSize
    var selectionRect: CGRect?
    var phase: TimeInterval
    
    // Derived for culling
    var visibleRect: CGRect { .zero }
    
    var body: some View {
        Canvas { context, size in
            let totalOffset = offset + currentDragOffset
            let center = CGPoint(x: size.width / 2 + totalOffset.width, y: size.height / 2 + totalOffset.height)
            
            context.translateBy(x: center.x, y: center.y)
            context.scaleBy(x: scale, y: scale)
            
            let renderer = FlowRenderer(
                scale: scale,
                offset: offset,
                currentDragOffset: currentDragOffset,
                selection: selection,
                activeStateIDs: activeStateIDs,
                hoveredNodeID: hoveredNodeID,
                hoveredParentID: hoveredParentID,
                connectingEdge: nil
            )
            
            // Frustum Culling
            let cullRect = CGRect(
                x: -center.x / scale,
                y: -center.y / scale,
                width: size.width / scale,
                height: size.height / scale
            ).insetBy(dx: -200, dy: -200)
            
            // Sort Nodes (Hierarchy)
            let sortedNodes = nodes.sorted { (a, b) -> Bool in
                if a.id == b.parentID { return true }
                if b.id == a.parentID { return false }
                if a.parentID == nil && b.parentID != nil { return true }
                if a.parentID != nil && b.parentID == nil { return false }
                return false
            }
            
            let visibleNodes = sortedNodes.filter { node in
                let nodeRect = CGRect(origin: node.position, size: node.size)
                return cullRect.intersects(nodeRect)
            }
            
            for node in visibleNodes {
                var nodeCopy = node
                nodeCopy.isSelected = selection.contains(node.id) // update selection state for drawing
                renderer.drawNode(context: context, node: nodeCopy, phase: phase)
            }
            
            // Draw Marquee Selection
            if let selectionRect = selectionRect {
                let path = Path(selectionRect)
                context.fill(path, with: .color(Color.blue.opacity(0.1)))
                context.stroke(path, with: .color(Color.blue), lineWidth: 1.0 / scale)
            }
        }
        .drawingGroup() // Optimize rendering
    }
}
