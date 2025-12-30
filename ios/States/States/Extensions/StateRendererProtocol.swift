import SwiftUI
import ExtensionKit

/// Protocol defining the interface for custom state renderers.
/// Use the 'AppExtension' macro to conform to this protocol in your extension.
protocol StateRenderer {
    /// Renders a custom view for the given state configuration.
    func render(state: StateData) -> AnyView
}

/// Data passed to the extension for rendering.
struct StateData: Codable, Sendable {
    let id: String
    let label: String
    let type: String
    let meta: [String: String]
}

/// The Extension Point Identifier.
let StateRendererExtensionPointID = "dev.tmc.States.renderer"
