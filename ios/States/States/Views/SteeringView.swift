
import SwiftUI
import Combine

// Simple Debouncer
class Debouncer: ObservableObject {
    @Published var input: [Int: Double] = [:]
    @Published var output: [Int: Double] = [:]
    private var cancellables = Set<AnyCancellable>()
    
    init(delay: TimeInterval = 0.2) {
        $input
            .debounce(for: .seconds(delay), scheduler: RunLoop.main)
            .sink { [weak self] val in
                self?.output = val
            }
            .store(in: &cancellables)
    }
}

struct SteeringView: View {
    @Binding var steeringValues: [Int: Double]
    var onGenerate: (String, [Int: Double]) -> Void
    @State private var prompt: String = "Traffic Light"
    @Environment(\.dismiss) var dismiss
    
    @StateObject private var debouncer = Debouncer(delay: 0.15)
    
    // Defined Features
    let features: [(id: Int, name: String, desc: String)] = [
        (10, "Robustness", "Increases error handling states"),
        (28, "Complexity", "Increases structural depth")
    ]
    
    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Prompt")) {
                    TextField("Describe Statechart...", text: $prompt)
                }
                
                Section(header: Text("Steering Controls (SAE)")) {
                    ForEach(features, id: \.id) { feature in
                        VStack(alignment: .leading) {
                            HStack {
                                Text(feature.name)
                                    .font(.headline)
                                Spacer()
                                Text(String(format: "%.1f", steeringValues[feature.id] ?? 0))
                                    .monospacedDigit()
                                    .foregroundStyle(.secondary)
                            }
                            Text(feature.desc)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            
                            Slider(
                                value: Binding(
                                    get: { steeringValues[feature.id] ?? 0 },
                                    set: { val in
                                        steeringValues[feature.id] = val
                                        debouncer.input = steeringValues // Trigger debounce
                                    }
                                ),
                                in: -10...10,
                                step: 0.5
                            ) {
                                Text("Strength")
                            } minimumValueLabel: {
                                Text("-10").font(.caption2)
                            } maximumValueLabel: {
                                Text("+10").font(.caption2)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
                
                Section {
                    Button(action: {
                        // Force immediate generation
                        onGenerate(prompt, steeringValues)
                    }) {
                        HStack {
                            Spacer()
                            Label("Force Refresh", systemImage: "arrow.clockwise")
                            Spacer()
                        }
                    }
                }
            }
            .navigationTitle("Generative Steering")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .destructiveAction) {
                    Button("Reset") { steeringValues = [:] }
                }
            }
            // Watch debounced output
            .onReceive(debouncer.$output) { values in
                 if !values.isEmpty {
                     onGenerate(prompt, values)
                 }
            }
            .onAppear {
                // Initialize debouncer
                debouncer.input = steeringValues
            }
        }
        .presentationDetents([.medium, .large])
        .presentationBackgroundInteraction(.enabled) // Allow usage of app behind sheet
    }
}

#Preview {
    SteeringView(steeringValues: .constant([10: 5.0]), onGenerate: { _ in })
}
