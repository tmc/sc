//
//  ContentView.swift
//  States
//
//  Created by tmc on 12/26/25.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationSplitView {
            MachineListView()
#if os(macOS)
                .navigationSplitViewColumnWidth(min: 180, ideal: 200)
#endif
        } detail: {
            Text("Select a Statechart")
        }
    }
}

#Preview("Content View") {
    ContentView()
        .environment(AppViewModel())
}


