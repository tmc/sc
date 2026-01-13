import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralSettingsView()
                .tabItem {
                    Label("General", systemImage: "gear")
                }
                .tag(1)
            
            AppearanceSettingsView()
                .tabItem {
                    Label("Appearance", systemImage: "paintbrush")
                }
                .tag(2)
        }
        .frame(width: 450, height: 350) // Standard settings size
    }
}

struct GeneralSettingsView: View {
    @AppStorage("enableHaptics") private var enableHaptics = true
    @AppStorage("autoSave") private var autoSave = true
    
    var body: some View {
        Form {
            Section {
                Toggle("Enable Haptics", isOn: $enableHaptics)
                Toggle("Auto-save Changes", isOn: $autoSave)
            } header: {
                Text("Behavior")
            } footer: {
                Text("Haptics play when moving nodes or connecting edges.")
            }
        }
        .formStyle(.grouped)
    }
}

struct AppearanceSettingsView: View {
    @AppStorage("showGrid") private var showGrid = true
    @AppStorage("showMinimap") private var showMinimap = false
    
    var body: some View {
        Form {
            Section("Canvas") {
                Toggle("Show Background Grid", isOn: $showGrid)
                Toggle("Show Minimap", isOn: $showMinimap)
                    .disabled(true) // Future feature
            }
        }
        .formStyle(.grouped)
    }
}

#Preview {
    SettingsView()
}
