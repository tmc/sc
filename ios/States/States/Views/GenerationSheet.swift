import SwiftUI

struct GenerationSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(AppViewModel.self) var viewModel
    @State private var prompt: String = ""
    @State private var isGenerating = false
    
    var body: some View {
        NavigationStack {
            Form {
                Section("Quick Prompts") {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            QuickPromptButton(label: "🚦 Traffic Light", prompt: "A traffic light system with green, yellow, red states")
                            QuickPromptButton(label: "🔐 Login Flow", prompt: "A secure login flow with idle, authenticating, loggedIn, and error states")
                            QuickPromptButton(label: "🎵 Music Player", prompt: "A music player with play, pause, and stop functionality")
                        }
                        .padding(.horizontal, 4)
                    }
                    .listRowInsets(EdgeInsets()) // Edge-to-edge scroll
                    .padding(.vertical, 8)
                }
                
                Section("Describe your machine") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 120)
                        .font(.body)
                        .overlay(alignment: .topLeading) {
                            if prompt.isEmpty {
                                Text("e.g., A multi-step checkout process with validation...")
                                    .foregroundStyle(.tertiary)
                                    .padding(.top, 8)
                                    .padding(.leading, 5)
                                    .allowsHitTesting(false)
                            }
                        }
                }
                
                Section {
                    Button(action: generate) {
                        HStack {
                            if isGenerating {
                                ProgressView()
                                    .controlSize(.small)
                                    .padding(.trailing, 8)
                                Text("Designing...")
                            } else {
                                Text("Generate Statechart")
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(prompt.isEmpty || isGenerating)
                    .listRowBackground(Color.clear)
                }
            }
            .navigationTitle("Generate")
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
        .frame(width: 450, height: 400)
        #endif
    }
    
    // Helper View for Quick Prompts
    func QuickPromptButton(label: String, prompt: String) -> some View {
        Button(action: { self.prompt = prompt }) {
            Text(label)
                .font(.subheadline)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(20)
        }
        .buttonStyle(.plain)
    }
    
    private func generate() {
        isGenerating = true
        
        // Simulate "thinking" time
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
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
