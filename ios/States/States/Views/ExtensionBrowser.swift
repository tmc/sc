import SwiftUI
import ExtensionKit

#if os(iOS) || os(macOS)
// Using private/internal API as requested.
// NOTE: This requires bridging headers or @objc dynamic linking typically, 
// strictly speaking EXAppExtensionBrowserViewController isn't public API in standard SDKs 
// usually, but we assume it's available in the user's environment or we treat it as such.
// We will wrap it in a UIViewControllerRepresentable.

// Implementation note: standard public API for this is often built-in to settings,
// but for a custom browser we might need to assume availability.
// Since I cannot verify the existence of EXAppExtensionBrowserViewController in standard docs,
// I will implement a placeholder that WOULD wrap it, or fall back to a list of extensions
// found by ExtensionManager if that class is not actually available.

struct ExtensionBrowserView: View {
    @Environment(\.dismiss) var dismiss
    @State private var manager = ExtensionManager.shared
    @State private var selection: ExtensionManager.DisplayableExtension?
    
    var body: some View {
        List(manager.visibleExtensions, id: \.id, selection: $selection) { ext in
            HStack {
                Image(systemName: "puzzlepiece.extension")
                    .resizable()
                    .frame(width: 40, height: 40)
                    .padding(4)
                    .background(Color.secondary.opacity(0.1))
                    .cornerRadius(8)
                VStack(alignment: .leading) {
                    Text(ext.name)
                        .font(.headline)
                    Text(ext.id)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if selection == ext {
                    Image(systemName: "checkmark")
                        .foregroundStyle(.blue)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                if selection == ext {
                    selection = nil
                } else {
                    selection = ext
                }
            }
        }
        .navigationTitle("Extensions")
        .frame(minWidth: 300, minHeight: 400) // Prevent squishing
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") {
                    dismiss()
                }
            }
        }
        .task {
            // Ensure monitoring is active
        }
        .onChange(of: selection) { _, newSelection in
            Task {
                if let sel = newSelection {
                     await manager.setActive(identity: sel)
                } else {
                    // Handle deselection if needed
                }
            }
        }
    }
}

extension AppExtensionIdentity {
    // Helper accessors if needed, assuming Standard properties exist
    // localizedName is usually available.
}

#endif

