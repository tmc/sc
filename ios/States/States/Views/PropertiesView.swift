import SwiftUI

struct PropertiesView: View {
    var viewModel: StatechartViewModel
    
    var body: some View {
        @Bindable var viewModel = viewModel
        
        Form {
            if let selectedNodeID = viewModel.selection.first,
               let index = viewModel.nodes.firstIndex(where: { $0.id == selectedNodeID }) {
                
                Section("Node Properties") {
                    TextField("Label", text: $viewModel.nodes[index].label)
                    Picker("Type", selection: $viewModel.nodes[index].type) {
                        Text("Atomic").tag(FlowNode.NodeType.atomic)
                        Text("Compound").tag(FlowNode.NodeType.compound)
                        Text("Parallel").tag(FlowNode.NodeType.parallel)
                        Text("Final").tag(FlowNode.NodeType.final)
                        Text("History").tag(FlowNode.NodeType.history)
                    }
                }
                
                Section("Metadata") {
                    LabeledContent("ID", value: viewModel.nodes[index].id.uuidString)
                    LabeledContent("Position", value: "\(Int(viewModel.nodes[index].position.x)), \(Int(viewModel.nodes[index].position.y))")
                }
                
            } else if let selectedEdgeID = viewModel.selection.first,
                      let index = viewModel.edges.firstIndex(where: { $0.id == selectedEdgeID }) {
                
                Section("Transition Properties") {
                    TextField("Label", text: Binding(
                        get: { viewModel.edges[index].label ?? "" },
                        set: { viewModel.edges[index].label = $0.isEmpty ? nil : $0 }
                    ))
                }
                
                Section("Connection") {
                    LabeledContent("Source", value: viewModel.nodes.first(where: { $0.id == viewModel.edges[index].source })?.label ?? "Unknown")
                    LabeledContent("Target", value: viewModel.nodes.first(where: { $0.id == viewModel.edges[index].target })?.label ?? "Unknown")
                }
                
            } else {
                // Machine Inspector (Global)
                Section("Simulation History") {
                    if viewModel.simulationHistory.isEmpty {
                        Text("No history captured.")
                            .foregroundStyle(.secondary)
                    } else {
                        List {
                            ForEach(Array(viewModel.simulationHistory.enumerated()), id: \.offset) { index, stateIDs in
                                HStack {
                                    Text("Step \(index)")
                                        .font(.caption.monospaced())
                                    Spacer()
                                    if index == viewModel.currentStepIndex {
                                        Image(systemName: "arrow.left.circle.fill")
                                            .foregroundStyle(.green)
                                    }
                                }
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    withAnimation {
                                        viewModel.jumpToStep(index)
                                    }
                                }
                            }
                        }
                        .frame(minHeight: 150)
                    }
                }
                
                Section("Machine Context") {
                    if let json = viewModel.machine.jsonContent,
                       let data = json.data(using: .utf8),
                       let obj = try? JSONSerialization.jsonObject(with: data, options: []),
                       let prettyData = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]),
                       let prettyString = String(data: prettyData, encoding: .utf8) {
                        
                        Text(prettyString)
                            .font(.system(size: 11, design: .monospaced))
                            .lineLimit(nil)
                            .textSelection(.enabled)
                            .padding(4)
                            .background(Color(.systemFill))
                            .cornerRadius(6)
                    } else {
                        Text("No JSON context available.")
                            .italic()
                    }
                }
            }
        }
        .inspectorColumnWidth(min: 250, ideal: 300, max: 400)
    }
}

#Preview("Properties - Populated") {
    let vm = StatechartViewModel(machine: StatechartWrapper(name: "Preview"))
    vm.selection = [vm.nodes.first!.id]
    return PropertiesView(viewModel: vm)
}

#Preview("Properties - Empty") {
    let vm = StatechartViewModel(machine: StatechartWrapper(name: "Preview"))
    return PropertiesView(viewModel: vm)
}
