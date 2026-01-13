import SwiftUI

struct SimulationToolbar: View {
    @Bindable var viewModel: StatechartViewModel
    @State private var eventName: String = ""
    @FocusState private var isEventFieldFocused: Bool

    var body: some View {
        HStack(spacing: 8) {
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
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44)
                    .background(AnyShapeStyle(Color.red.opacity(0.9)), in: Circle())
                    .shadow(color: Color.red.opacity(0.3), radius: 4, x: 0, y: 2)
            }
            .buttonStyle(.plain)
            .help("Stop Simulation (Esc)")

            Divider()
                .frame(height: 20)
                .padding(.horizontal, 4)

            // Step Controls
            HStack(spacing: 4) {
                Button(action: { viewModel.resetSimulation() }) {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.body.weight(.medium))
                        .frame(width: 28, height: 28)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Reset (⌘0)")

                HStack(spacing: 0) {
                    Button(action: { withAnimation(.easeInOut(duration: 0.15)) { viewModel.stepBack() } }) {
                        Image(systemName: "chevron.left")
                            .font(.body.weight(.bold))
                            .frame(width: 28, height: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.currentStepIndex == 0)
                    
                    Text("Step \(viewModel.currentStepIndex + 1)")
                        .font(.caption.monospacedDigit().weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(minWidth: 50)

                    Button(action: { withAnimation(.easeInOut(duration: 0.15)) { viewModel.stepForward() } }) {
                        Image(systemName: "chevron.right")
                            .font(.body.weight(.bold))
                            .frame(width: 28, height: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.currentStepIndex >= viewModel.simulationHistory.count - 1)
                }
                .padding(2)
                .background(AnyShapeStyle(Color.secondary.opacity(0.05)), in: Capsule())
            }

            Divider()
                .frame(height: 20)
                .padding(.horizontal, 4)

            // Event Injection
            HStack(spacing: 6) {
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)

                    TextField("Event", text: $eventName)
                        .textFieldStyle(.plain)
                        .font(.callout)
                        .frame(width: 90)
                        .focused($isEventFieldFocused)
                        .onSubmit { sendEvent() }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(AnyShapeStyle(Color.secondary.opacity(0.08)), in: Capsule())

                Button(action: sendEvent) {
                    Image(systemName: "arrow.up")
                        .font(.body.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(width: 28, height: 28)
                        .background(AnyShapeStyle(eventName.isEmpty ? Color.secondary.opacity(0.2) : Theme.Colors.accent), in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(eventName.isEmpty)
            }
            
            // Active State Indicators (Brief)
            if !viewModel.activeStateIDs.isEmpty {
                Divider().frame(height: 20).padding(.horizontal, 4)
                HStack(spacing: -8) {
                    ForEach(Array(viewModel.activeStateIDs.prefix(3)), id: \.self) { id in
                         Circle()
                            .fill(Color.green)
                            .frame(width: 8, height: 8)
                            .overlay(Circle().stroke(Theme.Colors.nodeBackground, lineWidth: 1))
                            .shadow(color: Color.green.opacity(0.4), radius: 2)
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .glassPill()
        .padding(.horizontal)
        .padding(.bottom, 8)
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

