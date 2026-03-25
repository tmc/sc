import SwiftUI

struct FlowEdgesLayer: View {
    @Binding var edges: [FlowEdge]
    var nodes: [FlowNode] 
    @Binding var selection: Set<UUID>
    @Binding var scale: CGFloat
    @Binding var offset: CGSize
    var currentDragOffset: CGSize
    var connectingEdge: (source: UUID, currentPoint: CGPoint)?
    
    var body: some View {
        Canvas { context, size in
            let totalOffset = offset + currentDragOffset
            context.translateBy(x: size.width / 2 + totalOffset.width,
                              y: size.height / 2 + totalOffset.height)
            context.scaleBy(x: scale, y: scale)
            
            let renderer = FlowRenderer(
                scale: scale,
                offset: offset,
                currentDragOffset: currentDragOffset,
                selection: selection,
                activeStateIDs: [],
                hoveredNodeID: nil,
                hoveredParentID: nil,
                connectingEdge: connectingEdge
            )
            
            // Draw Edges
            for edge in edges {
                if let sourceNode = nodes.first(where: { $0.id == edge.source }),
                   let targetNode = nodes.first(where: { $0.id == edge.target }) {
                    renderer.drawEdge(context: context, source: sourceNode, target: targetNode, edge: edge)
                }
            }
            
            // Draw Connection Dragging
            if let connecting = connectingEdge,
               let sourceNode = nodes.first(where: { $0.id == connecting.source }) {
                renderer.drawConnectionDrag(context: context, source: sourceNode, endPoint: connecting.currentPoint)
            }
        }
    }
}
