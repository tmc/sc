import Foundation
import Combine

@Observable
class ExtensionManager {
    static let shared = ExtensionManager()

    struct DisplayableExtension: Identifiable, Hashable {
        let id: String
        let name: String

        static func == (lhs: DisplayableExtension, rhs: DisplayableExtension) -> Bool {
            lhs.id == rhs.id
        }
        func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }
    }

    var visibleExtensions: [DisplayableExtension] = [
        DisplayableExtension(id: "minimalist", name: "Minimalist (Built-in)"),
        DisplayableExtension(id: "blueprint", name: "Blueprint (Built-in)")
    ]

    @MainActor var activeRenderer: (any StateRenderer)? = nil

    init() {}

    func connect(to ext: DisplayableExtension) async throws -> any StateRenderer {
        switch ext.id {
        case "minimalist": return MinimalistRenderer()
        case "blueprint": return BlueprintRenderer()
        default: throw ExtensionError.invalidProtocol
        }
    }

    @MainActor
    func setActive(identity: DisplayableExtension?) async {
        guard let identity = identity else {
            activeRenderer = nil
            return
        }
        do {
            activeRenderer = try await connect(to: identity)
        } catch {
            print("Failed to activate: \(error)")
        }
    }
}

enum ExtensionError: Error {
    case invalidProtocol
}
