import SwiftUI
import UniformTypeIdentifiers

struct MachineListView: View {
    @Environment(AppViewModel.self) var viewModel
    @State private var showingAddSheet = false
    @State private var showingGenerationSheet = false
    @State private var isImporting = false
    @State private var newMachineName = ""
    @State private var searchText = ""
    // @State private var selectedMachine: StatechartWrapper? // Moved to ViewModel for deep linking

    private var filteredMachines: [StatechartWrapper] {
        if searchText.isEmpty {
            return viewModel.machines
        }
        return viewModel.machines.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        @Bindable var bindableViewModel = viewModel
        List(selection: $bindableViewModel.selectedMachine) {
            ForEach(filteredMachines) { machine in
                NavigationLink(value: machine) {
                    MachineRowView(machine: machine)
                }
#if os(macOS)
                .listRowBackground(Color.clear) // Transparent rows for vibrant sidebar
#else
                .listRowBackground(Color(uiColor: .secondarySystemGroupedBackground)) // Card-like
#endif
                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                    Button(role: .destructive) {
                        delete(machine)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
                .swipeActions(edge: .leading, allowsFullSwipe: false) {
                    Button {
                        duplicateMachine(machine)
                    } label: {
                        Label("Duplicate", systemImage: "doc.on.doc")
                    }
                    .tint(.indigo)
                }
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        #else
        .listStyle(.sidebar)
        .scrollContentBackground(.hidden)
        .background(Theme.Colors.sidebarBackground) // Custom translucent background
        #endif
        .searchable(text: $searchText, prompt: "Search machines")
        .navigationTitle("Statecharts")

        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                HStack {
                    Button(action: { isImporting = true }) {
                        Label("Load Folder", systemImage: "folder")
                    }
                    Button(action: { showingAddSheet = true }) {
                        Label("Add Machine", systemImage: "plus")
                    }
                }
            }
            
            ToolbarItem(placement: .automatic) {
                Button(action: { showingGenerationSheet = true }) {
                    Label("Generate", systemImage: "wand.and.stars")
                }
                .help("Generate with AI")
            }
        }
        .sheet(isPresented: $showingAddSheet) {
            NavigationStack {
                Form {
                    TextField("Machine Name", text: $newMachineName)
                        .onSubmit { createMachine() }
                        .font(Theme.Typography.body)
                }
                .navigationTitle("New Machine")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Cancel") { showingAddSheet = false }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Create") { createMachine() }
                        .disabled(newMachineName.isEmpty)
                    }
                }
            }
            #if os(macOS)
            .frame(width: 320, height: 160)
            #endif
        }
        .sheet(isPresented: $showingGenerationSheet) {
            GenerationSheet()
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    viewModel.loadMachines(from: url)
                }
            case .failure(let error):
                print("Import failed: \(error.localizedDescription)")
            }
        }
    }
    
    // MARK: - Actions
    
    private func createMachine() {
        guard !newMachineName.isEmpty else { return }
        viewModel.addMachine(name: newMachineName)
        newMachineName = ""
        showingAddSheet = false
        hapticFeedback(.light)
    }

    private func duplicateMachine(_ machine: StatechartWrapper) {
        let newMachine = StatechartWrapper(name: "\(machine.name) Copy", jsonContent: machine.jsonContent)
        viewModel.machines.append(newMachine)
        hapticFeedback(.light)
    }
    
    private func delete(_ machine: StatechartWrapper) {
        if let index = viewModel.machines.firstIndex(where: { $0.id == machine.id }) {
            withAnimation {
                viewModel.deleteMachine(at: IndexSet(integer: index))
            }
            hapticFeedback(.medium)
        }
    }
    
    private enum HapticStyle {
        case light, medium, heavy
    }
    
    private func hapticFeedback(_ style: HapticStyle) {
        #if os(iOS)
        let impactStyle: UIImpactFeedbackGenerator.FeedbackStyle
        switch style {
        case .light: impactStyle = .light
        case .medium: impactStyle = .medium
        case .heavy: impactStyle = .heavy
        }
        UIImpactFeedbackGenerator(style: impactStyle).impactOccurred()
        #endif
    }
}

// MARK: - Machine Row View

struct MachineRowView: View {
    let machine: StatechartWrapper
    
    @State private var isHovered = false

    private var machineIcon: String {
        if machine.jsonContent?.contains("\"type\":\"parallel\"") == true {
            return "square.stack.3d.up.fill"
        }
        return "circle.hexagongrid.fill"
    }

    var body: some View {
        HStack(spacing: 12) {
            // Icon Container
            ZStack {
                Circle()
                    .fill(Theme.Colors.accent.opacity(0.1))
                    .frame(width: 36, height: 36)
                
                Image(systemName: machineIcon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Theme.Colors.accent)
            }
            
            // Text Content
            VStack(alignment: .leading, spacing: 4) {
                Text(machine.name)
                    .font(Theme.Typography.body)
                    .foregroundStyle(.primary)
                
                HStack(spacing: 6) {
                    if let json = machine.jsonContent, !json.isEmpty {
                        Text("JSON")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(Theme.Colors.gridDot)
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Empty")
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            
            Spacer()
        }
        .padding(.vertical, 6)
        .contentShape(Rectangle())
        #if os(macOS)
        .onHover { isHovered = $0 }
        // .background(isHovered ? Color.primary.opacity(0.05) : Color.clear) // Handled by List style?
        #endif
    }
}

#Preview("Machine List") {
    NavigationSplitView {
        MachineListView()
            .environment(AppViewModel())
    } detail: {
        Text("Select a Statechart")
    }
}

