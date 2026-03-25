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
            if !viewModel.openMachines.isEmpty {
                VStack(spacing: 0) {
                    TabBarView(viewModel: viewModel)
                    
                    if let machine = viewModel.selectedMachine {
                        VisualizerView(machine: machine, onGenerate: viewModel.generateRemote)
                            .id(machine.id) // Ensure view recreation on change
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        ContentUnavailableView("Error", systemImage: "exclamationmark.triangle")
                    }
                }
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


// MARK: - Tab Bar Components

struct TabBarView: View {
    var viewModel: AppViewModel
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(viewModel.openMachines) { machine in
                    TabItemView(machine: machine, isSelected: viewModel.selectedMachine?.id == machine.id) {
                        viewModel.selectedMachine = machine
                    } onClose: {
                        viewModel.closeTab(machine)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
        .background(Theme.Colors.canvasBackground)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }
}

struct TabItemView: View {
    let machine: StatechartWrapper
    let isSelected: Bool
    let onSelect: () -> Void
    let onClose: () -> Void
    
    @State private var isHovered = false
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "circle.hexagongrid")
                .font(.caption2)
                .foregroundStyle(isSelected ? Theme.Colors.accent : .secondary)
            
            Text(machine.name)
                .font(Theme.Typography.caption)
                .foregroundStyle(isSelected ? .primary : .secondary)
                .lineLimit(1)
            
            if isSelected || isHovered {
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .padding(2)
                .background(Color.primary.opacity(0.1), in: Circle())
            } else {
                // Spacer to keep layout stable
                Color.clear
                    .frame(width: 12, height: 12)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? isHovered ? Color.primary.opacity(0.1) : Color.primary.opacity(0.05) : isHovered ? Color.primary.opacity(0.03) : Color.clear)
        )
        .onTapGesture {
            onSelect()
            #if os(iOS)
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            #endif
        }
        #if os(macOS)
        .onHover { isHovered = $0 }
        #endif
    }
}

