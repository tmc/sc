//
//  StatesApp.swift
//  States
//
//  Created by tmc on 12/26/25.
//

import SwiftUI
import UniformTypeIdentifiers

@main
struct StatesApp: App {
    @State private var appViewModel = AppViewModel()
    @State private var isImporting = false
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appViewModel)
                .onOpenURL { url in
                    appViewModel.restore(from: url)
                }
                .onContinueUserActivity("com.tmc.States.viewMachine") { activity in
                    appViewModel.continueActivity(activity)
                }
                .fileImporter(
                    isPresented: $isImporting,
                    allowedContentTypes: [.folder],
                    allowsMultipleSelection: false
                ) { result in
                    switch result {
                    case .success(let urls):
                        if let url = urls.first {
                            // Ensure we can access the folder
                            appViewModel.loadMachines(from: url)
                        }
                    case .failure(let error):
                        print("Import failed: \(error)")
                    }
                }
        }
        .commands {
            SidebarCommands()
            StatechartCommands()
            
            #if os(macOS)
            CommandGroup(replacing: .appInfo) {
                Button("About States") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            NSApplication.AboutPanelOptionKey.credits: NSAttributedString(
                                string: "The Statechart Visualizer\n\nCopyright © 2024 TMC",
                                attributes: [
                                    NSAttributedString.Key.font: NSFont.boldSystemFont(ofSize: 11),
                                    NSAttributedString.Key.foregroundColor: NSColor.secondaryLabelColor
                                ]
                            ),
                            NSApplication.AboutPanelOptionKey(rawValue: "Copyright"): "© 2024 TMC"
                        ]
                    )
                }
            }
            #endif
            
            CommandGroup(replacing: .newItem) {
                Button("New Statechart") {
                    appViewModel.addMachine(name: "New Chart")
                }
                .keyboardShortcut("n", modifiers: .command)
            }
            
            CommandGroup(after: .newItem) {
                Button("Import Folder...") {
                    isImporting = true
                }
            }
        }
        
        #if os(macOS)
        Settings {
            SettingsView()
        }
        #endif
    }
}
