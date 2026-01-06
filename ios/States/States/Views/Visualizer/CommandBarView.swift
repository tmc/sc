import SwiftUI

struct CommandBarView: View {
    @Binding var isPresented: Bool
    var viewModel: StatechartViewModel
    
    @State private var query: String = ""
    @State private var selectedIndex: Int = 0
    
    struct Command: Identifiable {
        let id = UUID()
        let title: String
        let icon: String
        let shortcut: String?
        let action: () -> Void
    }
    
    var commands: [Command] {
        // Filter based on query
        let allCommands = [
            Command(title: "Add State", icon: "plus.square", shortcut: "⌘N", action: { viewModel.addState(at: .zero); close() }),
            Command(title: "Start Simulation", icon: "play.fill", shortcut: "⌘R", action: { viewModel.mode = .simulation; close() }),
            Command(title: "Stop Simulation", icon: "stop.fill", shortcut: "Esc", action: { viewModel.mode = .editing; close() }),
            Command(title: "Zoom to Fit", icon: "arrow.up.left.and.arrow.down.right", shortcut: "⇧Z", action: { viewModel.zoomToFit(viewSize: CGSize(width: 800, height: 600)); close() }), // Approximation
            Command(title: "Reset Simulation", icon: "arrow.counterclockwise", shortcut: "⌘0", action: { viewModel.resetSimulation(); close() })
        ]
        
        // Add Node Jump commands
        let nodeCommands = viewModel.nodes.map { node in
            Command(title: "Jump to \(node.label)", icon: "scope", shortcut: nil, action: {
                viewModel.selection = [node.id]
                // TODO: Pan to node
                close()
            })
        }
        
        let combined = allCommands + nodeCommands
        
        if query.isEmpty { return combined }
        return combined.filter { $0.title.localizedCaseInsensitiveContains(query) }
    }
    
    func close() {
        withAnimation(.easeOut(duration: 0.15)) {
            isPresented = false
            query = ""
            selectedIndex = 0
        }
    }
    
    var body: some View {
        ZStack(alignment: .top) {
            Color.black.opacity(0.2)
                .ignoresSafeArea()
                .onTapGesture { close() }
            
            VStack(spacing: 0) {
                // Search Field
                HStack(spacing: 12) {
                    Image(systemName: "command")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    
                    TextField("Type a command...", text: $query)
                        .font(.title3)
                        .textFieldStyle(.plain)
                        .onSubmit {
                            if !commands.isEmpty {
                                commands[selectedIndex].action()
                            }
                        }
                }
                .padding(16)
                .background(.ultraThinMaterial)
                
                Divider()
                
                // Results
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(Array(commands.enumerated()), id: \.element.id) { index, command in
                            Button(action: command.action) {
                                HStack {
                                    Image(systemName: command.icon)
                                        .frame(width: 24)
                                        .foregroundStyle(index == selectedIndex ? .white : .primary)
                                    
                                    Text(command.title)
                                        .foregroundStyle(index == selectedIndex ? .white : .primary)
                                    
                                    Spacer()
                                    
                                    if let shortcut = command.shortcut {
                                        Text(shortcut)
                                            .font(.caption)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.white.opacity(0.2), in: RoundedRectangle(cornerRadius: 4))
                                            .foregroundStyle(index == selectedIndex ? .white : .secondary)
                                    }
                                }
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(index == selectedIndex ? Theme.Colors.accent : Color.clear)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .onHover { isHovering in
                                if isHovering { selectedIndex = index }
                            }
                        }
                    }
                }
                .frame(maxHeight: 300)
            }
            .frame(width: 600)
            .background(.regularMaterial)
            .cornerRadius(12)
            .shadow(radius: 20, y: 10)
            .padding(.top, 100)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.white.opacity(0.1), lineWidth: 1)
            )
        }
    }
}
