import SwiftUI
import ExtensionKit

// Sample 2: Blueprint Renderer (Detailed, technical look)
struct BlueprintRenderer: StateRenderer {
    func render(state: StateData) -> AnyView {
        AnyView(
            ZStack(alignment: .topLeading) {
                Color.blue.opacity(0.1)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(state.label.uppercased())
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(.blue)
                    
                    Divider().background(Color.blue)
                    
                    Text("ID: \(state.id.prefix(4))")
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(.secondary)
                    
                    if !state.meta.isEmpty {
                        ForEach(Array(state.meta.keys.sorted().prefix(2)), id: \.self) { key in
                            Text("\(key): \(state.meta[key] ?? "")")
                                .font(.system(size: 8))
                        }
                    }
                }
                .padding(4)
            }
            .border(Color.blue, width: 1)
            .shadow(color: .blue.opacity(0.2), radius: 2)
        )
    }
}
