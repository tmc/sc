import SwiftUI

struct DesignToolbar: View {
    @Bindable var viewModel: StatechartViewModel
    @State private var selectedTool: DesignTool = .select

    enum DesignTool {
        case select, pan
    }

    var body: some View {
        HStack(spacing: 8) {
            // Tool Mode Selector
            HStack(spacing: 2) {
                ToolButton(
                    icon: "cursorarrow",
                    isSelected: selectedTool == .select,
                    action: { selectedTool = .select }
                )
                .help("Selection tool (V)")

                ToolButton(
                    icon: "hand.draw",
                    isSelected: selectedTool == .pan,
                    action: { selectedTool = .pan }
                )
                .help("Pan tool (H)")
            }
            .padding(2)
            .background(Color.secondary.opacity(0.1), in: Capsule())

            Divider()
                .frame(height: 20)
                .padding(.horizontal, 4)

            // Add State
            Button(action: { viewModel.addState(at: .zero) }) {
                Image(systemName: "plus.square.fill")
                    .font(.title2)
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(Theme.Colors.accent)
                    .frame(width: 36, height: 36)
                    .background(Theme.Colors.accent.opacity(0.1), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Add State (⌘N)")

            Divider()
                .frame(height: 20)
                .padding(.horizontal, 4)

            // Zoom Controls
            HStack(spacing: 0) {
                Button(action: { 
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) { 
                        viewModel.scale = max(viewModel.scale * 0.8, 0.1) 
                    } 
                }) {
                    Image(systemName: "minus")
                        .font(.body.weight(.bold))
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                
                Text("\(Int(viewModel.scale * 100))%")
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 42)
                
                Button(action: { 
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) { 
                        viewModel.scale = min(viewModel.scale * 1.25, 5.0) 
                    } 
                }) {
                    Image(systemName: "plus")
                        .font(.body.weight(.bold))
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .padding(2)
            .background(Color.secondary.opacity(0.05), in: Capsule())

            Divider()
                .frame(height: 20)
                .padding(.horizontal, 4)

            // Simulate Button
            Button(action: {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                    viewModel.mode = .simulation
                }
                #if os(iOS)
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                #endif
            }) {
                Image(systemName: "play.fill")
                    .font(.title3)
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44) // Circular play button
                    .background(Theme.Colors.accent, in: Circle())
                    .shadow(color: Theme.Colors.accent.opacity(0.4), radius: 4, x: 0, y: 2)
            }
            .buttonStyle(.plain)
            .help("Start Simulation (⌘R)")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .glassPill()
        .padding(.horizontal)
        .padding(.bottom, 8) 
    }
}

struct ToolButton: View {
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.body.weight(.medium))
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.plain)
        .background(isSelected ? Theme.Colors.accent.opacity(0.15) : Color.clear, in: Circle())
        .foregroundStyle(isSelected ? Theme.Colors.accent : Color.secondary)
    }
}

#Preview("DesignToolbar") {
    struct ToolbarPreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Toolbar Machine")
        @State var viewModel: StatechartViewModel
        init() {
            let m = StatechartWrapper(name: "Toolbar Machine")
            let vm = StatechartViewModel(machine: m)
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        var body: some View {
            VStack { Spacer(); DesignToolbar(viewModel: viewModel).padding(); }
            #if os(iOS)
            .background(Color(uiColor: .systemBackground))
            #else
            .background(Color(nsColor: .windowBackgroundColor))
            #endif
        }
    }
    return ToolbarPreviewWrapper()
}
