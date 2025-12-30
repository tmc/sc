import SwiftUI
import UniformTypeIdentifiers

struct MachineListView: View {
    @Environment(AppViewModel.self) var viewModel
    @State private var showingAddSheet = false
    @State private var showingGenerationSheet = false
    @State private var isImporting = false
    @State private var newMachineName = ""

    var body: some View {
        List {
            ForEach(viewModel.machines) { machine in
                NavigationLink(value: machine) {
                    Text(machine.name)
                }
            }
            .onDelete(perform: viewModel.deleteMachine)
        }
        .navigationTitle("Statecharts")
        .navigationDestination(for: StatechartWrapper.self) { machine in
            VisualizerView(machine: machine)
        }
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
            .frame(width: 300, height: 150)
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
    
    private func createMachine() {
        guard !newMachineName.isEmpty else { return }
        viewModel.addMachine(name: newMachineName)
        newMachineName = ""
        showingAddSheet = false
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
