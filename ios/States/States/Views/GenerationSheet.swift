import SwiftUI

struct GenerationSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AppViewModel.self) var viewModel
    @State private var prompt: String = ""
    @State private var isGenerating = false
    
    var body: some View {
        NavigationStack {
            Form {
                Section("Describe your machine") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 100)
                        .overlay(alignment: .topLeading) {
                            if prompt.isEmpty {
                                Text("e.g. A traffic light system with a timer...")
                                    .foregroundStyle(.tertiary)
                                    .padding(.top, 8)
                                    .padding(.leading, 5)
                                    .allowsHitTesting(false)
                            }
                        }
                }
                
                Section {
                    Button(action: generate) {
                        if isGenerating {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Text("Generate")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(prompt.isEmpty || isGenerating)
                }
            }
            .navigationTitle("Generate Statechart")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(width: 400, height: 300)
        #endif
    }
    
    private func generate() {
        isGenerating = true
        
        // Simulate network delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            viewModel.generateMachine(prompt: prompt)
            isGenerating = false
            dismiss()
        }
    }
}

#Preview {
    GenerationSheet()
        .environment(AppViewModel())
}
