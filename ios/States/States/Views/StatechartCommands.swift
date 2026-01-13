import SwiftUI

struct StatechartCommands: Commands {
    @FocusedValue(\.statechartViewModel) var viewModel

    var body: some Commands {
        CommandGroup(after: .pasteboard) {
            Button("Duplicate Node") {
                viewModel?.duplicateSelection()
            }
            .keyboardShortcut("d", modifiers: .command)
            .disabled(viewModel == nil || viewModel?.selection.isEmpty ?? true)
            
            Button("Delete Node") {
                viewModel?.deleteSelection()
            }
            .keyboardShortcut(.delete, modifiers: [])
            .disabled(viewModel == nil || viewModel?.selection.isEmpty ?? true)
        }
        
        CommandGroup(after: .newItem) {
            Button("Add State (Canvas)") {
                // Add at origin or inferred center
                // Ideally we'd map screen center to logic, but here we just use logic (0,0) offset
                // Using offset logic: -offset / scale
                if let vm = viewModel {
                    let centerX = -vm.offset.width / vm.scale
                    let centerY = -vm.offset.height / vm.scale
                    vm.addState(at: CGPoint(x: centerX + 100, y: centerY + 100))
                }
            }
            .keyboardShortcut("n", modifiers: [.command, .shift])
            .disabled(viewModel == nil)
        }
    }
}
