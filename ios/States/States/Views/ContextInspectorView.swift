//
//  ContextInspectorView.swift
//  States
//
//  Created by Assistant on 1/3/25.
//

import SwiftUI

struct ContextInspectorView: View {
    @ObservedObject var engine: StatechartEngine
    
    // Local state for editing to avoid jumpy text fields
    @State private var editingValue: String = ""
    @State private var editingKey: String?
    
    var body: some View {
        List {
            if engine.context.isEmpty {
                Text("No context variables")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(engine.context.keys.sorted(), id: \.self) { key in
                    HStack {
                        Text(key)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.secondary)
                        
                        Spacer()
                        
                        let value = engine.context[key] ?? ""
                        valueView(key: key, value: value)
                    }
                }
            }
        }
        .navigationTitle("Context")
    }
    
    @ViewBuilder
    func valueView(key: String, value: Any) -> some View {
        if let intVal = value as? Int {
            TextField("", value: Binding(
                get: { intVal },
                set: { engine.context[key] = $0 }
            ), format: .number)
            .multilineTextAlignment(.trailing)
            .frame(maxWidth: 100)
            
        } else if let doubleVal = value as? Double {
             TextField("", value: Binding(
                 get: { doubleVal },
                 set: { engine.context[key] = $0 }
             ), format: .number)
             .multilineTextAlignment(.trailing)
             .frame(maxWidth: 100)
            
        } else if let stringVal = value as? String {
            TextField("", text: Binding(
                get: { stringVal },
                set: { engine.context[key] = $0 }
            ))
            .multilineTextAlignment(.trailing)
            .frame(maxWidth: 150)
            
        } else if let boolVal = value as? Bool {
            Toggle("", isOn: Binding(
                get: { boolVal },
                set: { engine.context[key] = $0 }
            ))
            .labelsHidden()
            
        } else {
            Text("\(String(describing: value))")
                .foregroundStyle(.secondary)
        }
    }
}
