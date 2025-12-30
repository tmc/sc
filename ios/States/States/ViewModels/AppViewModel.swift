import SwiftUI
import Observation

@Observable
class AppViewModel {
    var machines: [StatechartWrapper] = []
    
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
        machines.append(newMachine)
    }
    
    func deleteMachine(at offsets: IndexSet) {
        machines.remove(atOffsets: offsets)
    }
    
    // MARK: - AI Generation (Mock)
    func generateMachine(prompt: String) {
        let name = "Generated: \(prompt.prefix(10))..."
        var machine = StatechartWrapper(name: name)
        
        let lowerPrompt = prompt.lowercased()
        
        if lowerPrompt.contains("traffic") {
            // Traffic Light Mock
            let json = """
            {
              "id": "trafficLight",
              "initial": "green",
              "states": {
                "green": { "on": { "TIMER": "yellow" } },
                "yellow": { "on": { "TIMER": "red" } },
                "red": { "on": { "TIMER": "green" } }
              }
            }
            """
            machine.jsonContent = json
        } else if lowerPrompt.contains("toggle") || lowerPrompt.contains("switch") {
             // Toggle Mock
             let json = """
             {
               "id": "toggle",
               "initial": "inactive",
               "states": {
                 "inactive": { "on": { "TOGGLE": "active" } },
                 "active": { "on": { "TOGGLE": "inactive" } }
               }
             }
             """
             machine.jsonContent = json
        } else {
            // Generic Mock
            let json = """
            {
              "id": "generic",
              "initial": "start",
              "states": {
                "start": { "on": { "NEXT": "end" } },
                "end": { "type": "final" }
              }
            }
            """
            machine.jsonContent = json
        }
        
        machines.append(machine)
    }
}
