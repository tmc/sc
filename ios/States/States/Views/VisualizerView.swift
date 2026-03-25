import SwiftUI

struct VisualizerView: View {
    @State private var viewModel: StatechartViewModel
    let machine: StatechartWrapper // Store machine for change detection
    @Environment(\.undoManager) var undoManager
    @State private var showExtensions = false
    @State private var showCommandBar = false
    
    var onGenerate: ((String, [Int: Double]) -> Void)?
    
    init(machine: StatechartWrapper, onGenerate: ((String, [Int: Double]) -> Void)? = nil) {
        self.machine = machine
        self.onGenerate = onGenerate
        _viewModel = State(initialValue: StatechartViewModel(machine: machine))
    }

    var body: some View {
        GeometryReader { geometry in
            mainStack(geometry: geometry)
                .onAppear {
                    viewModel.zoomToFit(viewSize: geometry.size)
                }
                .onChange(of: machine) { _, newMachine in
                    viewModel.load(machine: newMachine)
                }
                .onChange(of: viewModel.machine.id) { _, _ in
                    viewModel.zoomToFit(viewSize: geometry.size)
                }
        }
        .modifier(InspectorOrSheetModifier(viewModel: viewModel, onGenerate: onGenerate))
        .userActivity("com.tmc.States.viewMachine") { activity in
            updateUserActivity(activity)
        }
        #if os(macOS)
        .onDeleteCommand {
            viewModel.selection.forEach { viewModel.deleteNode(id: $0) }
            viewModel.selection.removeAll()
        }
        #endif
        .focusedSceneValue(\.statechartViewModel, viewModel)
        .toolbar { makeToolbar() }
        .sheet(isPresented: $showExtensions) { extensionSheetContent }
        .onAppear { viewModel.undoManager = undoManager }
        .onChange(of: undoManager) { _, newManager in
            viewModel.undoManager = newManager
        }
        .sheet(isPresented: $viewModel.showAnalysis) {
            if let report = viewModel.analysisReport {
                NavigationStack {
                    AnalysisView(report: report) { issue in
                        viewModel.showAnalysis = false
                        viewModel.selection = [issue.nodeID]
                        viewModel.centerNode(id: issue.nodeID, viewSize: CGSize(width: 800, height: 600)) // We need a way to get viewSize properly here, or just center.
                    }
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { viewModel.showAnalysis = false }
                        }
                    }
                }
                .presentationDetents([.medium, .large])
            }
        }
        // Context Inspector moved to Native Inspector on macOS
        // On iOS, it might need a sheet if not handled by Modifier.
        // Actually, let's keep it here for iOS ONLY if Modifier doesn't handle it?
        // Reuse InspectorOrSheetModifier logic.
        // If we remove this sheet, how does iOS show Context?
        // InspectorOrSheetModifier handles PropertiesView only currently.
        // I will add ContextInspector logic to Modifier for iOS too?
        // Or keep this sheet for iOS?
        // Let's keep this sheet for iOS logic, but wrap in #if os(iOS)
        #if os(iOS)
        .sheet(isPresented: $viewModel.showContextInspector) {
            if let engine = viewModel.engine {
                NavigationStack {
                    ContextInspectorView(engine: engine)
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Done") { viewModel.showContextInspector = false }
                            }
                        }
                }
                .presentationDetents([.medium])
            } else {
                Text("Simulation not active")
                    .padding()
            }
        }
        #endif
        .overlay { commandBarOverlay }
        .background { commandBarToggle }
    }

    // Break out the main ZStack to reduce type-checker load
    @ViewBuilder
    private func mainStack(geometry: GeometryProxy) -> some View {
        ZStack(alignment: .bottom) {
            flowCanvas
            
            if viewModel.isLoading {
                ProgressView("Loading...")
                    .controlSize(.large)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.ultraThinMaterial)
            } else {
                if viewModel.mode == StatechartViewModel.EditorMode.editing {
                    DesignToolbar(viewModel: viewModel)
                        .padding(.bottom, 24)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                } else {
                    SimulationToolbar(viewModel: viewModel)
                        .padding(.bottom, 24)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
    }

    // Extract FlowView construction to a property to simplify body
    private var flowCanvas: some View {
        FlowView(
            nodes: $viewModel.nodes,
            edges: $viewModel.edges,
            activeStateIDs: $viewModel.activeStateIDs,
            scale: $viewModel.scale,
            offset: $viewModel.offset,
            selection: $viewModel.selection,
            selectionRect: $viewModel.selectionRect,
            onNodeMoveEnded: { oldPositions in
                viewModel.registerMoveUndo(oldPositions: oldPositions)
            },
            onNodeDelete: { id in
                viewModel.deleteNode(id: id)
            },
            onNodeDuplicate: { id in
                viewModel.duplicateNode(id: id)
            },
            onAddTransitionFrom: { sourceID in
                // TODO: Wire up force-connection mode if needed, 
                // or just rely on drag. FlowView handles drag-to-connect primarily.
                // If we want a button to start connection mode, FlowView needs an external trigger state.
                // For now, let's just print or ignore. The context menu has "Add Transition" maybe?
                // The current context menu doesn't have "Add Transition", only Add Sub-state.
                // Let's implement this if requested later.
            },
            onNodeRename: { id, oldLabel in
                viewModel.registerRenameUndo(id: id, oldLabel: oldLabel)
            },
            onAddStateAt: { point in
                viewModel.addState(at: point)
            },
            onAddSubstate: { parentID in
                viewModel.addSubstate(parentID: parentID)
            },
            onNodeReparent: { childID, parentID in
                viewModel.reparent(childID: childID, newParentID: parentID)
            }
        )
    }

    // Toolbar builder to keep body small
    @ToolbarContentBuilder
    private func makeToolbar() -> some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            if viewModel.mode == .editing {
                Button(action: { viewModel.addState(at: .zero) }) {
                    Label("Add State", systemImage: "plus.square")
                }
                .keyboardShortcut("n", modifiers: .command)
                .help("Add new state (⌘N)")
            }
        }
        ToolbarItem(placement: .primaryAction) {
            /*
            Button(action: { showExtensions = true }) {
                Label("Extensions", systemImage: "puzzlepiece.extension")
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .help("Browse extensions (⇧⌘E)")
            */
        }
        ToolbarItem(placement: .primaryAction) {
            Button(action: { viewModel.analyze() }) {
                Label("Analyze", systemImage: "chart.bar.doc.horizontal")
            }
            .help("Analyze Statechart")
            .help("Analyze Statechart")
        }
        
        ToolbarItem(placement: .automatic) {
            ControlGroup {
                Button(action: { withAnimation { viewModel.scale = max(0.1, viewModel.scale - 0.25) } }) {
                    Label("Zoom Out", systemImage: "minus.magnifyingglass")
                }
                Button(action: { withAnimation { viewModel.scale = 1.0; viewModel.offset = .zero } }) {
                    Label("Actual Size", systemImage: "1.magnifyingglass")
                }
                Button(action: { withAnimation { viewModel.scale = min(4.0, viewModel.scale + 0.25) } }) {
                    Label("Zoom In", systemImage: "plus.magnifyingglass")
                }
            }
            .controlGroupStyle(.navigation) // Compact style
        }
        
        if viewModel.mode == .simulation {
            ToolbarItem(placement: .secondaryAction) {
                Button(action: {
                    // Toggle Context Inspector
                    // We need a state for this.
                    // For now, let's use the same sheet but switch content?
                    // Or add a new sheet state in ViewModel?
                    viewModel.showContextInspector.toggle()
                }) {
                    Label("Context", systemImage: "curlybraces")
                }
                .help("Inspect Variables")
            }
        }
        
        ToolbarItem(placement: .automatic) {
            if viewModel.mode == .editing {
                Button(action: { withAnimation { viewModel.mode = .simulation } }) {
                    Label("Simulate", systemImage: "play.fill")
                }
                .keyboardShortcut("r", modifiers: .command)
                .help("Start simulation (⌘R)")
            } else {
                Button(action: { withAnimation { viewModel.mode = .editing } }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .keyboardShortcut(.escape, modifiers: [])
                .help("Stop simulation (Esc)")
            }
        }
    }

    // Platform-specific inspector/sheet extracted
    private func inspectorOrSheet(viewModel: StatechartViewModel) -> some ViewModifier {
        return InspectorOrSheetModifier(viewModel: viewModel, onGenerate: onGenerate)
    }

    private struct InspectorOrSheetModifier: ViewModifier {
        let viewModel: StatechartViewModel
        let onGenerate: ((String, [Int: Double]) -> Void)?

        func body(content: Content) -> some View {
            #if os(iOS)
            content.sheet(isPresented: .constant(true)) {
                PropertiesView(viewModel: viewModel, onGenerate: onGenerate)
                    .presentationDetents([.height(160), .medium, .large])
                    .presentationBackgroundInteraction(.enabled(upThrough: .medium))
                    .presentationDragIndicator(.visible)
                    .interactiveDismissDisabled()
            }
            #else
            content.inspector(isPresented: .constant(true)) {
                if viewModel.showContextInspector, let engine = viewModel.engine {
                    ContextInspectorView(engine: engine)
                        .inspectorColumnWidth(min: 220, ideal: 280, max: 350)
                } else {
                    PropertiesView(viewModel: viewModel, onGenerate: onGenerate)
                        .inspectorColumnWidth(min: 220, ideal: 280, max: 350)
                }
            }
            #endif
        }
    }
    // MARK: - Body Helpers to Reduce Complexity
    
    private func updateUserActivity(_ activity: NSUserActivity) {
        activity.title = machine.name
        activity.userInfo = ["machineID": machine.id.uuidString]
        activity.isEligibleForSearch = true
        #if os(iOS)
        activity.isEligibleForPrediction = true
        #endif
        #if os(iOS) || os(macOS)
        activity.isEligibleForHandoff = true
        #endif
    }

    @ViewBuilder
    private var extensionSheetContent: some View {
        if #available(iOS 16.0, macOS 13.0, *) {
            NavigationStack { ExtensionBrowserView() }
        } else {
            Text("Extensions require iOS 16 / macOS 13 or newer.")
        }
    }

    @ViewBuilder
    private var commandBarOverlay: some View {
         if showCommandBar { CommandBarView(isPresented: $showCommandBar, viewModel: viewModel) }
    }
    
    @ViewBuilder
    private var commandBarToggle: some View {
        Button("") { showCommandBar.toggle() }
            .keyboardShortcut("k", modifiers: .command)
            .hidden()
        
        // Keyboard Shortcuts
        Button("Duplicate") {
            viewModel.selection.forEach { viewModel.duplicateNode(id: $0) }
        }
        .keyboardShortcut("d", modifiers: .command)
        .hidden()
        
        Button("Delete") {
            viewModel.selection.forEach { viewModel.deleteNode(id: $0) }
            viewModel.selection.removeAll()
        }
        .keyboardShortcut(.delete, modifiers: [])
        .hidden()
    }
}




#Preview("Visualizer - Complex Cycle") {
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Cyclic Machine")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Cyclic Machine")
            let vm = StatechartViewModel(machine: m)
            
            let a = FlowNode(id: UUID(), position: CGPoint(x: 100, y: 200), label: "A", type: .atomic)
            let b = FlowNode(id: UUID(), position: CGPoint(x: 300, y: 100), label: "B", type: .atomic)
            let c = FlowNode(id: UUID(), position: CGPoint(x: 300, y: 300), label: "C", type: .atomic)
            
            vm.nodes = [a, b, c]
            vm.edges = [
                FlowEdge(source: a.id, target: b.id, event: "to B"),
                FlowEdge(source: b.id, target: c.id, event: "to C"),
                FlowEdge(source: c.id, target: a.id, event: "to A")
            ]
            
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        
        var body: some View {
            VisualizerView(machine: machine)
        }
    }
    return PreviewWrapper()
        .frame(width: 800, height: 600)
}

#Preview("Visualizer - Parallel States") {
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Parallel Machine")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Parallel Machine")
            let vm = StatechartViewModel(machine: m)
            
            let parent = FlowNode(id: UUID(), position: CGPoint(x: 50, y: 50), label: "Upload Composite", type: .parallel, size: CGSize(width: 400, height: 300))
            let p1 = FlowNode(id: UUID(), position: CGPoint(x: 80, y: 100), label: "Progress", type: .atomic, parentID: parent.id)
            let p2 = FlowNode(id: UUID(), position: CGPoint(x: 80, y: 200), label: "Timer", type: .atomic, parentID: parent.id)
            
            vm.nodes = [parent, p1, p2]
            vm.edges = [
                FlowEdge(source: parent.id, target: p1.id, event: "start"),
                FlowEdge(source: parent.id, target: p2.id, event: "start")
            ]
            
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        
        var body: some View {
            VisualizerView(machine: machine)
        }
    }
    return PreviewWrapper()
        .frame(width: 800, height: 600)
}

#Preview("Visualizer - Empty Machine") {
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Empty Machine")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Empty Machine")
            let vm = StatechartViewModel(machine: m)
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        
        var body: some View {
            VisualizerView(machine: machine)
        }
    }
    return PreviewWrapper()
        .frame(width: 800, height: 600)
}

#Preview("Visualizer - Large Graph") {
    struct LargeGraphPreview: View {
        @State var machine = StatechartWrapper(name: "Large Graph")
        @State var viewModel: StatechartViewModel
        init() {
            let m = StatechartWrapper(name: "Large Graph")
            let vm = StatechartViewModel(machine: m)
            var nodes: [FlowNode] = []
            var edges: [FlowEdge] = []
            // Create a grid of nodes 5x5
            let cols = 5
            let rows = 5
            let spacing: CGFloat = 150
            for r in 0..<rows {
                for c in 0..<cols {
                    let id = UUID()
                    let pos = CGPoint(x: 100 + CGFloat(c) * spacing, y: 100 + CGFloat(r) * spacing)
                    nodes.append(FlowNode(id: id, position: pos, label: "N\(r)\(c)", type: .atomic))
                }
            }
            // Connect row-wise
            for r in 0..<rows {
                for c in 0..<(cols - 1) {
                    let source = nodes[r * cols + c].id
                    let target = nodes[r * cols + (c + 1)].id
                    edges.append(FlowEdge(source: source, target: target, event: "→"))
                }
            }
            // Connect column-wise
            for r in 0..<(rows - 1) {
                for c in 0..<cols {
                    let source = nodes[r * cols + c].id
                    let target = nodes[(r + 1) * cols + c].id
                    edges.append(FlowEdge(source: source, target: target, event: "↓"))
                }
            }
            vm.nodes = nodes
            vm.edges = edges
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        var body: some View { VisualizerView(machine: machine) }
    }
    return LargeGraphPreview()
        .frame(width: 800, height: 600)
}

#Preview("Visualizer - Zoomed & Offset") {
    struct ZoomedPreview: View {
        @State var machine = StatechartWrapper(name: "Zoomed")
        @State var viewModel: StatechartViewModel
        init() {
            let m = StatechartWrapper(name: "Zoomed")
            let vm = StatechartViewModel(machine: m)
            let a = FlowNode(id: UUID(), position: CGPoint(x: 0, y: 0), label: "Root", type: .atomic)
            vm.nodes = [a]
            vm.scale = 2.0
            vm.offset = CGSize(width: -200, height: -150)
            _viewModel = State(initialValue: vm)
            _machine = State(initialValue: m)
        }
        var body: some View { VisualizerView(machine: machine) }
    }
    return ZoomedPreview()
        .frame(width: 800, height: 600)
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

#Preview("ToolButton States") {
    VStack(spacing: 16) {
        ToolButton(icon: "cursorarrow", isSelected: true, action: {})
        ToolButton(icon: "hand.draw", isSelected: false, action: {})
    }
    .padding()
}

