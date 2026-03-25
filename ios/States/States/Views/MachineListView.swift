import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

struct MachineListView: View {
    @Environment(AppViewModel.self) var viewModel
    @State private var isImporting = false
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
                .simultaneousGesture(TapGesture().onEnded {
                    hapticFeedback(.light)
                })
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
                .contextMenu {
                    Button {
                        copyJSON(machine)
                    } label: {
                        Label("Copy JSON", systemImage: "doc.on.doc")
                    }
                    
                    if let json = machine.jsonContent, !json.isEmpty {
                        ShareLink(item: json, preview: SharePreview(machine.name)) {
                            Label("Share...", systemImage: "square.and.arrow.up")
                        }
                    }
                    
                    Button(role: .destructive) {
                        delete(machine)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        #else
        .listStyle(.sidebar)
        // Removing explicit background to allow native NSVisualEffectView vibrancy
        #endif
        .overlay {
            if filteredMachines.isEmpty {
                if #available(iOS 17.0, macOS 14.0, *) {
                    if !searchText.isEmpty {
                        ContentUnavailableView.search(text: searchText)
                    } else {
                        ContentUnavailableView(
                            "No Statecharts",
                            systemImage: "tray",
                            description: Text("Create a new statechart or import a folder to get started.")
                        )
                    }
                } else {
                    Text("No Statecharts found")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .searchable(text: $searchText, prompt: "Search machines")
        .navigationTitle("Statecharts")

        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                HStack {
                    Button(action: { isImporting = true }) {
                        Label("Load Folder", systemImage: "folder")
                    }
                    Button(action: { createMachine() }) {
                        Label("New Chart", systemImage: "plus")
                    }
                }
            }
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
        viewModel.addMachine(name: "New Chart")
        hapticFeedback(.light)
    }

    private func duplicateMachine(_ machine: StatechartWrapper) {
        viewModel.duplicateMachine(machine)
        hapticFeedback(.light)
    }
    
    private func copyJSON(_ machine: StatechartWrapper) {
        let json = machine.jsonContent ?? ""
        #if os(macOS)
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(json, forType: .string)
        #else
        UIPasteboard.general.string = json
        #endif
        hapticFeedback(.medium)
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
                    .accessibilityHidden(true)
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

