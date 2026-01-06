import SwiftUI
import ExtensionKit

// Sample 1: Minimalist Renderer (High contrast, simple shapes)
struct MinimalistRenderer: StateRenderer {
    func render(state: StateData) -> AnyView {
        AnyView(
            ZStack {
                Circle()
                    .fill(Color.black)
                Text(state.label)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.white)
            }
            .frame(width: 60, height: 60)
        )
    }
}
