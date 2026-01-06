//
//  AnalysisView.swift
//  States
//
//  Created by Assistant on 1/3/25.
//

import SwiftUI

struct AnalysisView: View {
    let report: AnalysisReport
    let onSelectIssue: (AnalysisIssue) -> Void
    
    var body: some View {
        List {
            Section(header: Text("Summary")) {
                HStack {
                    Text("Reachability")
                    Spacer()
                    Text(String(format: "%.1f%%", report.reachabilityPercentage))
                        .foregroundColor(report.reachabilityPercentage < 100 ? .orange : .green)
                }
                HStack {
                    Text("Total States")
                    Spacer()
                    Text("\(report.totalStates)")
                }
                HStack {
                    Text("Reachable")
                    Spacer()
                    Text("\(report.reachableStates)")
                }
            }
            
            if report.isEmpty {
                Section {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("No issues found.")
                    }
                }
            } else {
                ForEach(AnalysisIssueType.allCases) { type in
                    let issues = report.issues(of: type)
                    if !issues.isEmpty {
                        Section(header: Label(type.rawValue, systemImage: type.icon)) {
                            ForEach(issues) { issue in
                                Button(action: {
                                    onSelectIssue(issue)
                                }) {
                                    HStack {
                                        Text(issue.nodeLabel)
                                            .font(.system(.body, design: .monospaced))
                                        Spacer()
                                        Image(systemName: "chevron.right")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .foregroundColor(.primary)
                            }
                        }
                    }
                }
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        #else
        .listStyle(.inset)
        #endif
        .navigationTitle("Analysis")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}
