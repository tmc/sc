import SwiftUI
import Observation
import Foundation

@Observable
class AppViewModel {
    var machines: [StatechartWrapper] = []
    var selectedMachine: StatechartWrapper?
    
    init() {
        // Mock Data
        machines = []
        // Load Sample Data
        for (key, json) in SampleData.machines {
            if key == "machines" {
                // Parse the array of machines
                if let data = json.data(using: .utf8) {
                    do {
                         let importedMachines = try JSONDecoder().decode([StatelyDocsMachine].self, from: data)
                         for m in importedMachines {
                             // Encode the definition back to JSON string for StatechartWrapper
                             let defData = try JSONEncoder().encode(m.definition)
                             let defString = String(data: defData, encoding: .utf8)
                             machines.append(StatechartWrapper(name: m.name, jsonContent: defString))
                         }
                    } catch {
                        print("Failed to parse machines array: \(error)")
                    }
                }
            } else {
                 machines.append(StatechartWrapper(name: key, jsonContent: json))
            }
        }
        
        // Sort by name for consistency
        machines.sort { $0.name < $1.name }
    }
    
    func loadMachines(from folderURL: URL) {
        // Allow access to the folder securely
        let secure = folderURL.startAccessingSecurityScopedResource()
        defer {
            if secure { folderURL.stopAccessingSecurityScopedResource() }
        }
        
        let fileManager = FileManager.default
        let keys: [URLResourceKey] = [.isRegularFileKey]
        
        guard let enumerator = fileManager.enumerator(at: folderURL, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles]) else {
            print("Failed to create file enumerator")
            return
        }
        
        for case let fileURL as URL in enumerator {
            // Check if it's a JSON file
            if fileURL.pathExtension.lowercased() == "json" {
                // Use filename as machine name for now, stripping extension
                let name = fileURL.deletingPathExtension().lastPathComponent
                
                // Read content
                do {
                    let content = try String(contentsOf: fileURL, encoding: .utf8)
                    
                    if !machines.contains(where: { $0.name == name }) {
                        let newMachine = StatechartWrapper(name: name, jsonContent: content)
                        machines.append(newMachine)
                    } else {
                        // Update existing? Or skip? Let's update.
                         if let index = machines.firstIndex(where: { $0.name == name }) {
                             machines[index].jsonContent = content
                         }
                    }
                } catch {
                    print("Failed to read JSON content from \(fileURL.lastPathComponent): \(error)")
                }
            }
        }
        
        // Sort
        machines.sort { $0.name < $1.name }
    }
    
    func addMachine(name: String) {
        let newMachine = StatechartWrapper(name: name)
        machines.insert(newMachine, at: 0)
    }
    
    func deleteMachine(at offsets: IndexSet) {
        machines.remove(atOffsets: offsets)
    }
    
    // MARK: - AI Generation (Mock)
    // MARK: - AI Generation (Mock)
    func generateMachine(prompt: String) {
        let name = "Generated: \(prompt.prefix(15))..."
        var machine = StatechartWrapper(name: name)
        
        let lowerPrompt = prompt.lowercased()
        
        if lowerPrompt.contains("traffic") {
            let json = """
            {
              "name": "trafficLight",
              "root_state": {
                "label": "trafficLight",
                "type": "OR",
                "is_initial": true,
                "children": [
                  { "label": "green", "type": "BASIC", "is_initial": true },
                  { "label": "yellow", "type": "BASIC" },
                  { "label": "red", "type": "BASIC" }
                ]
              },
              "transitions": [
                { "from": ["green"], "to": ["yellow"], "event": "TIMER" },
                { "from": ["yellow"], "to": ["red"], "event": "TIMER" },
                { "from": ["red"], "to": ["green"], "event": "TIMER" }
              ]
            }
            """
            machine.jsonContent = json
        } else if lowerPrompt.contains("login") || lowerPrompt.contains("auth") {
             // Login Flow Mock
             let json = """
             {
               "name": "loginFlow",
               "root_state": {
                 "label": "loginFlow",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "idle", "type": "BASIC", "is_initial": true },
                   { "label": "authenticating", "type": "BASIC" },
                   { "label": "loggedIn", "type": "BASIC" },
                   { "label": "error", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["idle"], "to": ["authenticating"], "event": "LOGIN" },
                 { "from": ["authenticating"], "to": ["loggedIn"], "event": "SUCCESS" },
                 { "from": ["authenticating"], "to": ["error"], "event": "FAILURE" },
                 { "from": ["loggedIn"], "to": ["idle"], "event": "LOGOUT" },
                 { "from": ["error"], "to": ["authenticating"], "event": "RETRY" },
                 { "from": ["error"], "to": ["idle"], "event": "CANCEL" }
               ]
             }
             """
             machine.jsonContent = json
        } else if lowerPrompt.contains("music") || lowerPrompt.contains("player") {
             // Music Player Mock
             let json = """
             {
               "name": "musicPlayer",
               "root_state": {
                 "label": "musicPlayer",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "stopped", "type": "BASIC", "is_initial": true },
                   { "label": "playing", "type": "BASIC" },
                   { "label": "paused", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["stopped"], "to": ["playing"], "event": "PLAY" },
                 { "from": ["playing"], "to": ["paused"], "event": "PAUSE" },
                 { "from": ["playing"], "to": ["stopped"], "event": "STOP" },
                 { "from": ["paused"], "to": ["playing"], "event": "PLAY" },
                 { "from": ["paused"], "to": ["stopped"], "event": "STOP" }
               ]
             }
             """
             machine.jsonContent = json
        } else if lowerPrompt.contains("toggle") || lowerPrompt.contains("switch") {
             // Toggle Mock
             let json = """
             {
               "name": "toggle",
               "root_state": {
                 "label": "toggle",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "inactive", "type": "BASIC", "is_initial": true },
                   { "label": "active", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["inactive"], "to": ["active"], "event": "TOGGLE" },
                 { "from": ["active"], "to": ["inactive"], "event": "TOGGLE" }
               ]
             }
             """
             machine.jsonContent = json
        } else {
            // Generic Mock for unknown
             let json = """
             {
               "name": "generic",
               "root_state": {
                 "label": "generic",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "start", "type": "BASIC", "is_initial": true },
                   { "label": "process", "type": "BASIC" },
                   { "label": "end", "type": "BASIC" },
                   { "label": "fail", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["start"], "to": ["process"], "event": "NEXT" },
                 { "from": ["process"], "to": ["end"], "event": "COMPLETE" },
                 { "from": ["process"], "to": ["fail"], "event": "ERROR" },
                 { "from": ["fail"], "to": ["process"], "event": "RETRY" }
               ]
             }
             """
             machine.jsonContent = json
        }
        
        machines.insert(machine, at: 0)
        machines.insert(machine, at: 0)
    }

    // MARK: - Remote Generation (SAE Steering)
    
    func generateRemote(prompt: String, steering: [Int: Double]) {
        guard let url = URL(string: "http://localhost:8000/generate") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "prompt": prompt,
            "steering": Dictionary(uniqueKeysWithValues: steering.map { (String($0.key), $0.value) })
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            print("Failed to encode request: \(error)")
            return
        }
        
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                print("Remote Gen Error: \(error)")
                return
            }
            
            guard let data = data else { return }
            
            do {
                if let jsonResponse = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let scJSON = jsonResponse["json"] as? String {
                   
                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        
                        // Check if we are already viewing a steered version of this prompt
                        let expectedName = "Steered: \(prompt.prefix(10))"
                        
                        if let current = self.selectedMachine, current.name == expectedName {
                            // Update in place to preserve view state (if possible)
                            // StatechartWrapper's jsonContent is @Observation tracked, so this should trigger update
                            if current.jsonContent != scJSON {
                                current.jsonContent = scJSON
                            }
                        } else {
                            // Create new
                            let machine = StatechartWrapper(name: expectedName, jsonContent: scJSON)
                            self.machines.insert(machine, at: 0)
                            self.selectedMachine = machine
                        }
                    }
                }
            } catch {
               print("Failed to decode response: \(error)")
            }
        }
        task.resume()
    }
    
    // MARK: - Deep Linking & restoration
    
    func restore(from url: URL) {
        // Mock restoration logic
        // Scheme: states://machine/<ID> or <Name>
        // For simplicity, we match by name if ID isn't found
        let path = url.lastPathComponent
        if let machine = machines.first(where: { $0.id.uuidString == path || $0.name == path }) {
             selectedMachine = machine
        }
    }
    
    func continueActivity(_ activity: NSUserActivity) {
        if activity.activityType == "com.tmc.States.viewMachine",
           let machineIDString = activity.userInfo?["machineID"] as? String {
            if let machine = machines.first(where: { $0.id.uuidString == machineIDString }) {
                selectedMachine = machine
            }
        }
    }
}
