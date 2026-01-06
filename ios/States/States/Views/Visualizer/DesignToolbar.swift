import SwiftUI

struct DesignToolbar: View {
    @Bindable var viewModel: StatechartViewModel
    @State private var selectedTool: DesignTool = .select

    enum DesignTool {
        case select, pan
    }

    var body: some View {
        HStack(spacing: 16) {
            // Tool Mode Selector
            HStack(spacing: 4) {
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
            .padding(4)
            .background(Color.secondary.opacity(0.1), in: Capsule())

            Divider()
                .frame(height: 24)

            // Add State
            Button(action: { viewModel.addState(at: .zero) }) {
                Image(systemName: "plus.square.fill")
                    .font(.title3)
                    .foregroundStyle(Theme.Colors.accent)
            }
            .buttonStyle(.plain)
            .help("Add State (⌘N)")

            // Zoom Controls
            HStack(spacing: 2) {
                Button(action: { 
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) { 
                        viewModel.scale = max(viewModel.scale * 0.8, 0.1) 
                    } 
                }) {
                    Image(systemName: "minus")
                        .font(.body.weight(.bold))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
                
                Text("\(Int(viewModel.scale * 100))%")
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 44)
                
                Button(action: { 
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) { 
                        viewModel.scale = min(viewModel.scale * 1.25, 5.0) 
                    } 
                }) {
                    Image(systemName: "plus")
                        .font(.body.weight(.bold))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
            }
            .padding(6)
            .background(Color.secondary.opacity(0.05), in: Capsule())

            Divider()
                .frame(height: 24)

            // Simulate Button
            Button(action: {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                    viewModel.mode = .simulation
                }
                #if os(iOS)
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                #endif
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "play.fill")
                    Text("Simulate")
                }
                .font(.headline)
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Theme.Colors.accent, in: Capsule())
            }
            .buttonStyle(.plain)
            .shadow(color: Theme.Colors.accent.opacity(0.4), radius: 8, x: 0, y: 4)
        }
        .padding(12)
        .ultraThinGlass() // Use Theme extension
        .padding(.horizontal)
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
                .frame(width: 32, height: 32)
        }
        .buttonStyle(.plain)
        .background(isSelected ? Theme.Colors.accent.opacity(0.15) : Color.clear, in: Circle())
        .foregroundStyle(isSelected ? Theme.Colors.accent : Color.primary)
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
