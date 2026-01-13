//
//  ContentView.swift
//  States
//
//  Created by tmc on 12/26/25.
//

import SwiftUI

struct ContentView: View {
    @Environment(AppViewModel.self) var viewModel


    var body: some View {
        NavigationSplitView {
            MachineListView()
#if os(macOS)
                .navigationSplitViewColumnWidth(min: 180, ideal: 200)
#endif
        } detail: {
            if let machine = viewModel.selectedMachine {
                VisualizerView(machine: machine, onGenerate: viewModel.generateRemote)
                    .id(machine.id) // Ensure view recreation on change
            } else {
                if #available(iOS 17.0, macOS 14.0, *) {
                    ContentUnavailableView("Select a Statechart", systemImage: "flowchart", description: Text("Select a statechart from the sidebar to view or edit it."))
                } else {
                    Text("Select a Statechart")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

#Preview("Content View") {
    ContentView()
        .environment(AppViewModel())
}

#Preview("Content View - Dark Mode") {
    ContentView()
        .environment(AppViewModel())
        .preferredColorScheme(.dark)
}

#Preview("Content View - Right-to-Left") {
    ContentView()
        .environment(AppViewModel())
        .environment(\.locale, Locale(identifier: "ar"))
}
#Preview("Content View - Dynamic Type Large") {
    ContentView()
        .environment(AppViewModel())
        .environment(\.sizeCategory, .accessibilityExtraExtraExtraLarge)
}

#Preview("Content View - Compact Width", traits: .fixedLayout(width: 320, height: 640)) {
    ContentView()
        .environment(AppViewModel())
}

#if os(macOS)
#Preview("Content View - macOS Window Medium") {
    ContentView()
        .environment(AppViewModel())
        .frame(width: 900, height: 600)
}
#endif

