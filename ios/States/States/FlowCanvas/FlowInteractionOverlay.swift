import SwiftUI

struct FlowInteractionOverlay: View {
    let size: CGSize
    let offset: CGSize
    let scale: CGFloat
    let currentDragOffset: CGSize
    
    @Binding var nodes: [FlowNode]
    @Binding var editingNodeID: UUID?
    @Binding var editingText: String
    
    // Callbacks
    // Callbacks
    var onNodeRename: ((UUID, String) -> Void)?
    var onAddSubstate: ((UUID) -> Void)?
    var onNodeDuplicate: ((UUID) -> Void)?
    var onNodeDelete: ((UUID) -> Void)?
    var onHitTest: ((CGPoint) -> Void)?
    
    // Drag Callbacks
    var onDragChanged: ((DragGesture.Value) -> Void)?
    var onDragEnded: ((DragGesture.Value) -> Void)?
    
    @FocusState private var isEditingFocus: Bool
    
    var body: some View {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        
        ForEach($nodes) { $node in
            let nodePos = node.position
            let viewX = center.x + offset.width + currentDragOffset.width + nodePos.x * scale
            let viewY = center.y + offset.height + currentDragOffset.height + nodePos.y * scale
            let width = node.size.width * scale
            let height = node.size.height * scale
            
            // Interaction Zone
            Color.white.opacity(0.01)
                .contentShape(RoundedRectangle(cornerRadius: 8))
                .frame(width: width, height: height)
                .position(x: viewX + width/2, y: viewY + height/2)
                .onTapGesture(count: 2) {
                    editingNodeID = node.id
                    editingText = node.label
                    isEditingFocus = true
                }
                .simultaneousGesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .local)
                        .onChanged { value in
                            onDragChanged?(value)
                        }
                        .onEnded { value in
                            onDragEnded?(value)
                        }
                )
                .onTapGesture(count: 1) {
                    onHitTest?(CGPoint(x: viewX + width/2, y: viewY + height/2))
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(node.label)
                .accessibilityIdentifier(node.label)
                .accessibilityAddTraits(.isButton)
                .contextMenu {
                    Button {
                        editingNodeID = node.id
                        editingText = node.label
                        isEditingFocus = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    Button {
                        onAddSubstate?(node.id)
                    } label: {
                        Label("Add Sub-state", systemImage: "plus.square.on.square")
                    }
                    Button {
                        onNodeDuplicate?(node.id)
                    } label: {
                        Label("Duplicate", systemImage: "doc.on.doc")
                    }
                    Divider()
                    Button(role: .destructive) {
                        onNodeDelete?(node.id)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
        }
        
        // Editing TextField
        if let id = editingNodeID, let node = nodes.first(where: { $0.id == id }) {
            let viewX = center.x + offset.width + currentDragOffset.width + node.position.x * scale
            let viewY = center.y + offset.height + currentDragOffset.height + node.position.y * scale
             
            TextField("Label", text: $editingText)
                .textFieldStyle(.plain)
                .font(Theme.Typography.nodeLabel(scale: scale))
                .multilineTextAlignment(.center)
                .padding(4)
                .background(Theme.Colors.nodeBackground)
                .cornerRadius(4)
                .focused($isEditingFocus)
                .frame(width: node.size.width * scale * 1.5)
                .position(x: viewX + (node.size.width * scale / 2), y: viewY + (node.size.height * scale / 2))
                .onSubmit {
                    commitRename(id: id)
                }
                .onChange(of: isEditingFocus) { oldValue, newValue in
                    if !newValue { commitRename(id: id) }
                }
        }
    }
    
    private func commitRename(id: UUID) {
        guard let index = nodes.firstIndex(where: { $0.id == id }) else { return }
        if nodes[index].label != editingText {
            nodes[index].label = editingText
            onNodeRename?(id, editingText)
        }
        editingNodeID = nil
    }
}
