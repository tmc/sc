import SwiftUI

struct VisualizerView: View {
    @State private var viewModel: StatechartViewModel
    let machine: StatechartWrapper // Store machine for change detection
    @Environment(\.undoManager) var undoManager
    @State private var showExtensions = false
    
    init(machine: StatechartWrapper) {
        self.machine = machine
        _viewModel = State(initialValue: StatechartViewModel(machine: machine))
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                // FlowCanvas
                FlowView(
                    nodes: $viewModel.nodes,
                    edges: $viewModel.edges,
                    activeStateIDs: $viewModel.activeStateIDs,
                    scale: $viewModel.scale,
                    offset: $viewModel.offset,
                    selection: $viewModel.selection,
                    onNodeMoveEnded: { oldPositions in
                        viewModel.registerMoveUndo(oldPositions: oldPositions)
                    }
                )
                
                // Simulation Controls
                if viewModel.mode == StatechartViewModel.EditorMode.editing {
                    DesignToolbar(viewModel: viewModel)
                        .padding(.bottom)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                } else {
                    SimulationToolbar(viewModel: viewModel)
                        .padding(.bottom)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .onAppear {
                viewModel.zoomToFit(viewSize: geometry.size)
            }
            // If the incoming machine changes (Navigation selection), reload the VM
            .onChange(of: machine) { _, newMachine in
                viewModel.load(machine: newMachine)
                // Optionally reset undo manager scope?
            }
            .onChange(of: viewModel.machine.id) { _, _ in
                // This might be redundant if we use the above, but good for internal changes
                viewModel.zoomToFit(viewSize: geometry.size)
            }
        }
        // Native Inspector
        .inspector(isPresented: .constant(true)) {
            PropertiesView(viewModel: viewModel)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                if viewModel.mode == .editing {
                    Button(action: { viewModel.addState(at: .zero) }) {
                        Label("Add State", systemImage: "plus.square")
                    }
                }
            }
            ToolbarItem(placement: .destructiveAction) {
                if viewModel.mode == .editing {
                    Button(role: .destructive, action: { viewModel.deleteSelection() }) {
                        Label("Delete", systemImage: "trash")
                    }
                    .keyboardShortcut(.delete, modifiers: [])
                    .disabled(viewModel.selection.isEmpty)
                }
            }
            ToolbarItem(placement: .primaryAction) {
                 Button(action: { showExtensions = true }) {
                     Label("Extensions", systemImage: "puzzlepiece.extension")
                 }
            }
        }
        .sheet(isPresented: $showExtensions) {
            if #available(iOS 16.0, macOS 13.0, *) {
                NavigationStack {
                    ExtensionBrowserView()
                }
            } else {
                Text("Extensions require iOS 16 / macOS 13 or newer.")
            }
        }
        .onAppear {
            viewModel.undoManager = undoManager
        }
        .onChange(of: undoManager) { _, newManager in
            viewModel.undoManager = newManager
        }
    }
}

struct DesignToolbar: View {
    @Bindable var viewModel: StatechartViewModel
    
    var body: some View {
        HStack(spacing: 12) {
            // Tools (Visual placeholders for now, except Add)
            Group {
                Button(action: { /* Selection Mode */ }) {
                    Image(systemName: "cursorarrow")
                }
                .buttonStyle(.plain)
                .padding(8)
                .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                
                Button(action: { /* Pan Mode */ }) {
                    Image(systemName: "hand.draw")
                }
                .buttonStyle(.plain)
                .padding(8)
            }

            Divider()
                .frame(height: 20)

            Button(action: { viewModel.addState(at: .zero) }) {
                Image(systemName: "plus.square")
            }
            .buttonStyle(.plain)
            .padding(8)
            .help("Add State")

            Divider()
                .frame(height: 20)
            
            Button(action: { withAnimation { viewModel.mode = StatechartViewModel.EditorMode.simulation } }) {
                HStack(spacing: 4) {
                    Image(systemName: "play.fill")
                    Text("Simulate")
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
        .padding(.horizontal)
        .padding(.bottom, 16)
    }
}

struct SimulationToolbar: View {
    @Bindable var viewModel: StatechartViewModel
    @State private var eventName: String = ""
    
    var body: some View {
        HStack {
            // Step Controls
            HStack(spacing: 4) {
                // Exit / Stop
                Button(action: { withAnimation { viewModel.mode = StatechartViewModel.EditorMode.editing } }) {
                    Image(systemName: "stop.fill")
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
                .padding(.trailing, 8)
                .help("Stop Simulation")
                
                Divider()
                    .padding(.trailing, 8)
                
                Button(action: { viewModel.resetSimulation() }) {
                    Image(systemName: "arrow.counterclockwise")
                }
                .help("Reset Simulation")
                
                Button(action: { withAnimation { viewModel.stepBack() } }) {
                    Image(systemName: "backward.frame")
                }
                .disabled(viewModel.currentStepIndex == 0)
                .help("Step Back")
                
                Button(action: { withAnimation { viewModel.stepForward() } }) {
                    Image(systemName: "forward.frame")
                }
                .disabled(viewModel.currentStepIndex >= viewModel.simulationHistory.count - 1)
                .help("Step Forward")
                
                Text("\(viewModel.currentStepIndex)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 30)
            }
            .padding(.trailing, 8)
            
            Divider()
            
            // Event Injection
            HStack {
                TextField("Event Name", text: $eventName)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .onSubmit {
                        sendEvent()
                    }
                
                Button("Send") {
                    sendEvent()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(eventName.isEmpty)
            }
            
            Spacer()
            
            // Active State Summary
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach(Array(viewModel.activeStateIDs), id: \.self) { id in
                        if let node = viewModel.nodes.first(where: { $0.id == id }) {
                            Text(node.label)
                                .font(.caption.bold())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.green.opacity(0.2), in: Capsule())
                                .overlay(Capsule().stroke(Color.green.opacity(0.5), lineWidth: 1))
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
        .padding(.horizontal)
        .padding(.bottom, 16)
    }
    
    private func sendEvent() {
        guard !eventName.isEmpty else { return }
        withAnimation {
            viewModel.sendEvent(eventName)
        }
        eventName = ""
    }
}



#Preview("Visualizer - Complex Cycle") {
    // State Injection Pattern
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Cyclic Machine")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Cyclic Machine")
            let vm = StatechartViewModel(machine: m)
            
            // Create Cycle
            let a = FlowNode(id: UUID(), position: CGPoint(x: 100, y: 200), label: "A", type: .atomic)
            let b = FlowNode(id: UUID(), position: CGPoint(x: 300, y: 100), label: "B", type: .atomic)
            let c = FlowNode(id: UUID(), position: CGPoint(x: 300, y: 300), label: "C", type: .atomic)
            
            vm.nodes = [a, b, c]
            vm.edges = [
                FlowEdge(source: a.id, target: b.id, label: "to B"),
                FlowEdge(source: b.id, target: c.id, label: "to C"),
                FlowEdge(source: c.id, target: a.id, label: "to A")
            ]
            
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        
        var body: some View {
            VisualizerView(machine: machine)
        }
    }
    return PreviewWrapper()
}

#Preview("Visualizer - Parallel States") {
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Parallel Machine")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Parallel Machine")
            let vm = StatechartViewModel(machine: m)
            
            // Parallel Parent (Large container)
            let parent = FlowNode(id: UUID(), position: CGPoint(x: 50, y: 50), label: "Upload Composite", type: .parallel, size: CGSize(width: 400, height: 300))
            
            // Children (visualized inside)
            let p1 = FlowNode(id: UUID(), position: CGPoint(x: 80, y: 100), label: "Progress", type: .atomic, parentID: parent.id)
            let p2 = FlowNode(id: UUID(), position: CGPoint(x: 80, y: 200), label: "Timer", type: .atomic, parentID: parent.id)
            
            vm.nodes = [parent, p1, p2]
            vm.edges = [
                FlowEdge(source: parent.id, target: p1.id, label: "start"),
                FlowEdge(source: parent.id, target: p2.id, label: "start")
            ]
            
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        
        var body: some View {
            VisualizerView(machine: machine)
        }
    }
    return PreviewWrapper()
}

#Preview("Visualizer - Disconnected") {
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Disconnected")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Disconnected")
            let vm = StatechartViewModel(machine: m)
            
            vm.nodes = [
                FlowNode(id: UUID(), position: CGPoint(x: 100, y: 100), label: "Cluster 1", type: .atomic),
                FlowNode(id: UUID(), position: CGPoint(x: 100, y: 400), label: "Cluster 2", type: .atomic)
            ]
            
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        
        var body: some View {
            VisualizerView(machine: machine)
        }
    }
    return PreviewWrapper()
}
