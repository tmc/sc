import Foundation
import ExtensionKit
import Combine

@Observable
class ExtensionManager {
    static let shared = ExtensionManager()
    
    struct DisplayableExtension: Identifiable, Hashable {
        let id: String
        let name: String
        let identity: AppExtensionIdentity?
        
        static func == (lhs: DisplayableExtension, rhs: DisplayableExtension) -> Bool {
            lhs.id == rhs.id
        }
        func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }
    }
    
    var visibleExtensions: [DisplayableExtension] = [
        DisplayableExtension(id: "minimalist", name: "Minimalist (Built-in)", identity: nil),
        DisplayableExtension(id: "blueprint", name: "Blueprint (Built-in)", identity: nil)
    ]
    
    @MainActor var activeRenderer: (any StateRenderer)? = nil
    
    init() { startMonitoring() }
    
    func startMonitoring() {
        Task {
            do {
                let monitor = AppExtensionPoint.Monitor()
                for await extensions in monitor {
                    let matching = extensions.filter { $0.extensionPointIdentifier == StateRendererExtensionPointID }
                    await updateExtensions(matching)
                }
            } catch { print("Monitor error: \(error)") }
        }
    }
    
    @MainActor
    private func updateExtensions(_ extensions: [AppExtensionIdentity]) {
        var current = [
             DisplayableExtension(id: "minimalist", name: "Minimalist (Built-in)", identity: nil),
             DisplayableExtension(id: "blueprint", name: "Blueprint (Built-in)", identity: nil)
        ]
        let found = extensions.map { DisplayableExtension(id: $0.bundleIdentifier, name: $0.localizedName ?? "Unknown", identity: $0) }
        current.append(contentsOf: found)
        self.visibleExtensions = current
    }
    
    func connect(to ext: DisplayableExtension) async throws -> any StateRenderer {
        if let identity = ext.identity {
            let process = try AppExtensionProcess(configuration: .init(appExtensionIdentity: identity))
            let extensionProxy = try process.makeXPCConnection().remoteObjectProxy as? (any StateRenderer)
            guard let proxy = extensionProxy else { throw ExtensionError.invalidProtocol }
            return proxy
        } else {
            switch ext.id {
            case "minimalist": return MinimalistRenderer()
            case "blueprint": return BlueprintRenderer()
            default: throw ExtensionError.invalidProtocol
            }
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
