import SwiftUI

struct SimulationToolbar: View {
    @Bindable var viewModel: StatechartViewModel
    @State private var eventName: String = ""
    @FocusState private var isEventFieldFocused: Bool

    var body: some View {
        HStack(spacing: 16) {
            // Stop Button
            Button(action: {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                    viewModel.mode = .editing
                }
                #if os(iOS)
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                #endif
            }) {
                Image(systemName: "stop.fill")
                    .font(.title3)
                    .foregroundStyle(.red)
                    .frame(width: 44, height: 44)
                    .background(Color.red.opacity(0.15), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Stop Simulation (Esc)")

            Divider()
                .frame(height: 28)

            // Step Controls
            HStack(spacing: 12) {
                Button(action: { viewModel.resetSimulation() }) {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.body.weight(.medium))
                }
                .buttonStyle(.plain)
                .help("Reset (⌘0)")

                HStack(spacing: 4) {
                    Button(action: { withAnimation(.easeInOut(duration: 0.15)) { viewModel.stepBack() } }) {
                        Image(systemName: "chevron.left")
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.currentStepIndex == 0)
                    .padding(4)
                    
                    Text("Step \(viewModel.currentStepIndex + 1)")
                        .font(.caption.monospacedDigit().weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(minWidth: 50)

                    Button(action: { withAnimation(.easeInOut(duration: 0.15)) { viewModel.stepForward() } }) {
                        Image(systemName: "chevron.right")
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.currentStepIndex >= viewModel.simulationHistory.count - 1)
                    .padding(4)
                }
                .padding(4)
                .background(Color.secondary.opacity(0.05), in: Capsule())
            }

            Divider()
                .frame(height: 28)

            // Event Injection
            HStack(spacing: 8) {
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)

                    TextField("Event", text: $eventName)
                        .textFieldStyle(.plain)
                        .frame(width: 100)
                        .focused($isEventFieldFocused)
                        .onSubmit { sendEvent() }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.secondary.opacity(0.08), in: Capsule())

                Button(action: sendEvent) {
                    Text("Send")
                        .font(.subheadline.weight(.semibold))
                }
                .buttonStyle(.borderedProminent)
                .disabled(eventName.isEmpty)
                .controlSize(.small)
            }
            
            // Active State Indicators (Brief)
            if !viewModel.activeStateIDs.isEmpty {
                Divider().frame(height: 20)
                HStack(spacing: -8) {
                    ForEach(Array(viewModel.activeStateIDs.prefix(3)), id: \.self) { id in
                         Circle()
                            .fill(Color.green)
                            .frame(width: 8, height: 8)
                            .overlay(Circle().stroke(Theme.Colors.nodeBackground, lineWidth: 1))
                    }
                }
            }
        }
        .padding(12)
        .ultraThinGlass()
        .padding(.horizontal)
    }

    private func sendEvent() {
        guard !eventName.isEmpty else { return }
        withAnimation(.easeInOut(duration: 0.2)) {
            viewModel.sendEvent(eventName)
        }
        eventName = ""
    }
}

#Preview("SimulationToolbar - Basic") {
    struct SimulationToolbarPreview: View {
        @State var machine = StatechartWrapper(name: "Sim Machine")
        @State var viewModel: StatechartViewModel
        init() {
            let m = StatechartWrapper(name: "Sim Machine")
            let vm = StatechartViewModel(machine: m)
            // Seed some basic simulation state
            vm.simulationHistory = [[], [], []]
            vm.currentStepIndex = 1
            vm.activeStateIDs = [UUID(), UUID()]
            vm.mode = .simulation
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        var body: some View {
            VStack { Spacer(); SimulationToolbar(viewModel: viewModel).padding(); }
            #if os(iOS)
            .background(Color(uiColor: .systemBackground))
            #else
            .background(Color(nsColor: .windowBackgroundColor))
            #endif
        }
    }
    return SimulationToolbarPreview()
}
