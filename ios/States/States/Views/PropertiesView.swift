import SwiftUI

struct PropertiesView: View {
    @Bindable var viewModel: StatechartViewModel
    var onGenerate: ((String, [Int: Double]) -> Void)?
    
    @State private var prompt: String = ""
    @State private var robustness: Double = 0.0
    @State private var complexity: Double = 0.0
    @State private var isGenerating: Bool = false
    
    var body: some View {
        Form {
            if let selectedNodeID = viewModel.selection.first,
               let index = viewModel.nodes.firstIndex(where: { $0.id == selectedNodeID }) {
                
                // --- Node Editing ---
                Section {
                    TextField("Label", text: $viewModel.nodes[index].label)
                        .textFieldStyle(.plain)
                        .font(Theme.Typography.body)
                        .padding(8)
                        .background(Theme.Colors.canvasBackground, in: RoundedRectangle(cornerRadius: 6))
                } header: {
                    Text("Node Properties")
                }
                
                Section {
                    Picker("Type", selection: $viewModel.nodes[index].type) {
                        Text("Atomic").tag(FlowNode.NodeType.atomic)
                        Text("Compound").tag(FlowNode.NodeType.compound)
                        Text("Parallel").tag(FlowNode.NodeType.parallel)
                        Text("Final").tag(FlowNode.NodeType.final)
                        Text("History").tag(FlowNode.NodeType.history)
                    }
                    .pickerStyle(.menu)
                    .tint(Theme.Colors.accent)
                } header: {
                    Text("Configuration")
                }
                
                Section {
                    LabeledContent {
                        Text(viewModel.nodes[index].id.uuidString)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    } label: {
                        Text("ID")
                            .font(Theme.Typography.caption)
                    }
                    
                    LabeledContent {
                        Text("\(Int(viewModel.nodes[index].position.x)), \(Int(viewModel.nodes[index].position.y))")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    } label: {
                        Text("Position")
                            .font(Theme.Typography.caption)
                    }
                } header: {
                    Text("Metadata")
                }
                
            } else if let selectedEdgeID = viewModel.selection.first,
                      let index = viewModel.edges.firstIndex(where: { $0.id == selectedEdgeID }) {
                
                // --- Edge Editing ---
                Section {
                    TextField("Label", text: Binding(
                        get: { viewModel.edges[index].event ?? "" },
                        set: { viewModel.edges[index].event = $0.isEmpty ? nil : $0 }
                    ))
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.body)
                    .padding(8)
                    .background(Theme.Colors.canvasBackground, in: RoundedRectangle(cornerRadius: 6))
                } header: {
                    Text("Transition")
                }

                Section {
                    TextField("Guard (Condition)", text: Binding(
                        get: { viewModel.edges[index].guardExpression ?? "" },
                        set: { viewModel.edges[index].guardExpression = $0.isEmpty ? nil : $0 }
                    ))
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.body.monospaced())
                    .padding(8)
                    .background(Theme.Colors.canvasBackground, in: RoundedRectangle(cornerRadius: 6))
                    
                    TextField("Action", text: Binding(
                        get: { viewModel.edges[index].action ?? "" },
                        set: { viewModel.edges[index].action = $0.isEmpty ? nil : $0 }
                    ))
                    .textFieldStyle(.plain)
                    .font(Theme.Typography.body.monospaced())
                    .padding(8)
                    .background(Theme.Colors.canvasBackground, in: RoundedRectangle(cornerRadius: 6))
                } header: {
                    Text("Logic")
                }
                
                Section {
                    LabeledContent("Source", value: viewModel.nodes.first(where: { $0.id == viewModel.edges[index].source })?.label ?? "Unknown")
                    LabeledContent("Target", value: viewModel.nodes.first(where: { $0.id == viewModel.edges[index].target })?.label ?? "Unknown")
                } header: {
                    Text("Connection")
                }
                
            } else {
                // --- Global / Machine Inspection ---
                
                // MACHINE PROPERTIES
                Section {
                    TextField("Name", text: $viewModel.machine.name)
                        .textFieldStyle(.plain)
                        .font(Theme.Typography.body.bold())
                        .padding(8)
                        .background(Theme.Colors.canvasBackground, in: RoundedRectangle(cornerRadius: 6))
                } header: {
                    Text("Machine Properties")
                }
                
                // AI GENERATION
                Section {
                   DisclosureGroup("Generative AI") {
                       VStack(alignment: .leading, spacing: 12) {
                           Text("Describe the desired chart logic:")
                               .font(.caption)
                               .foregroundStyle(.secondary)
                           
                           TextField("Prompt e.g., 'Traffic Light'", text: $prompt)
                               .textFieldStyle(.roundedBorder)
                           
                           VStack(alignment: .leading) {
                               Text("Steering Features (SAE)")
                                   .font(.caption2)
                                   .foregroundStyle(.secondary)
                               
                               SteeringControl(label: "Robustness", value: $robustness)
                               SteeringControl(label: "Complexity", value: $complexity)
                           }
                           
                           Button(action: {
                               let steering = [10: robustness, 28: complexity]
                               onGenerate?(prompt, steering)
                           }) {
                               HStack {
                                   Spacer()
                                   if isGenerating {
                                       ProgressView().controlSize(.small)
                                   } else {
                                       Label("Generate", systemImage: "sparkles")
                                   }
                                   Spacer()
                               }
                           }
                           .buttonStyle(.borderedProminent)
                           .disabled(prompt.isEmpty)
                       }
                       .padding(.vertical, 4)
                   }
                } header: {
                    Text("Assistant")
                }

                Section {
                    if viewModel.simulationHistory.isEmpty {
                        HStack {
                            Image(systemName: "clock.arrow.circlepath")
                            Text("No history captured.")
                        }
                        .font(Theme.Typography.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding()
                    } else {
                        List {
                            ForEach(Array(viewModel.simulationHistory.enumerated()), id: \.offset) { index, stateIDs in
                                HStack {
                                    Text("Step \(index)")
                                        .font(.caption.monospaced())
                                        .foregroundStyle(index == viewModel.currentStepIndex ? Theme.Colors.accent : .primary)
                                    Spacer()
                                    if index == viewModel.currentStepIndex {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(Theme.Colors.accent)
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
                } header: {
                    Text("Simulation History")
                }
                
                Section {
                    if let json = viewModel.machine.jsonContent,
                       let data = json.data(using: .utf8),
                       let obj = try? JSONSerialization.jsonObject(with: data, options: []),
                       let prettyData = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]),
                       let prettyString = String(data: prettyData, encoding: .utf8) {
                        
                        Text(prettyString)
                            .font(.system(size: 10, design: .monospaced))
                            .lineLimit(nil)
                            .textSelection(.enabled)
                            .padding(8)
                            .background(Theme.Colors.canvasBackground, in: RoundedRectangle(cornerRadius: 6))
                    } else {
                        Text("No JSON context available.")
                            .italic()
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    HStack {
                        Text("Machine Context")
                        Spacer()
                        if let _ = viewModel.machine.jsonContent {
                            Label("JSON", systemImage: "curlybraces")
                                .font(.caption2)
                        }
                    }
                }
            }
        }
        #if os(macOS)
        .formStyle(.grouped)
        #else
        .listStyle(.insetGrouped) // Use list layout for form appearance on iOS
        #endif
        .inspectorColumnWidth(min: 220, ideal: 280, max: 350)
        .background(Theme.Colors.sidebarBackground) // Consistent background
        .scrollContentBackground(.hidden)
    }
}

#Preview("Properties - Populated") {
    struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Preview")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Preview")
            let vm = StatechartViewModel(machine: m)
            
            let node = FlowNode(id: UUID(), position: .zero, label: "Scanning", type: .atomic)
            vm.nodes = [node]
            vm.selection = [node.id]
            
            _machine = State(initialValue: m)
            _viewModel = State(initialValue: vm)
        }
        
        var body: some View {
            PropertiesView(viewModel: viewModel)
        }
    }
    return PreviewWrapper()
}

#Preview("Properties - Empty") {
     struct PreviewWrapper: View {
        @State var machine = StatechartWrapper(name: "Preview")
        @State var viewModel: StatechartViewModel
        
        init() {
            let m = StatechartWrapper(name: "Preview")
            let vm = StatechartViewModel(machine: m)
            _machine = State(initialValue: m)
            _viewModel = State(initialValue: vm)
        }
        
        var body: some View {
            PropertiesView(viewModel: viewModel)
        }
    }
    return PreviewWrapper()
}

// MARK: - Helper Views

struct SteeringControl: View {
    let label: String
    @Binding var value: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f", value))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            Slider(value: $value, in: -1...1)
                .tint(Theme.Colors.accent)
        }
    }
}
