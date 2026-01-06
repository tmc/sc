//
//  StatesApp.swift
//  States
//
//  Created by tmc on 12/26/25.
//

import SwiftUI

@main
struct StatesApp: App {
    @State private var appViewModel = AppViewModel()

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
        }
    }
}
