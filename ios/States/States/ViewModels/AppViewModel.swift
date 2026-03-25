import SwiftUI
import Observation
import Foundation
import OSLog

@Observable
class AppViewModel {
    var machines: [StatechartWrapper] = []
    
    // MARK: - Tab Management
    var openMachines: [StatechartWrapper] = []
    
    var selectedMachine: StatechartWrapper? {
        didSet {
            if let machine = selectedMachine {
                // Auto-open tab for selected machine
                if !openMachines.contains(where: { $0.id == machine.id }) {
                    openMachines.append(machine)
                }
            }
        }
    }
    
    init() {
        // Load from Persistence
        machines = LibraryManager.shared.loadAll()
        
        if machines.isEmpty {
            loadSampleData()
            // Persist sample data on first run
            for machine in machines {
                LibraryManager.shared.save(machine)
            }
        }
        
        // Sort by name for consistency
        machines.sort { $0.name < $1.name }
    }
    
    private func loadSampleData() {
        // Load Sample Data (Simplified)
        for (key, json) in SampleData.machines {
             machines.append(StatechartWrapper(name: key, jsonContent: json))
        }
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
            Logger.viewModels.error("Failed to create file enumerator")
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
                        LibraryManager.shared.save(newMachine)
                    } else {
                        // Update existing? Or skip? Let's update.
                         if let index = machines.firstIndex(where: { $0.name == name }) {
                             machines[index].jsonContent = content
                             LibraryManager.shared.save(machines[index])
                         }
                    }
                } catch {
                    Logger.viewModels.error("Failed to read JSON content from \(fileURL.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
        
        // Sort
        machines.sort { $0.name < $1.name }
    }
    
    @discardableResult
    func addMachine(name: String) -> StatechartWrapper {
        let newMachine = StatechartWrapper(name: name)
        machines.insert(newMachine, at: 0)
        selectedMachine = newMachine
        LibraryManager.shared.save(newMachine)
        return newMachine
    }
    
    func deleteMachine(at offsets: IndexSet) {
        offsets.forEach { index in
            let machine = machines[index]
            // Close tab if open
            if openMachines.contains(where: { $0.id == machine.id }) {
                closeTab(machine)
            }
            LibraryManager.shared.delete(machine)
        }
        machines.remove(atOffsets: offsets)
    }
    
    func updateMachine(_ machine: StatechartWrapper) {
        if let index = machines.firstIndex(where: { $0.id == machine.id }) {
            machines[index] = machine
            LibraryManager.shared.save(machine)
        }
    }
    
    func duplicateMachine(_ machine: StatechartWrapper) {
        let newMachine = StatechartWrapper(name: "\(machine.name) Copy", jsonContent: machine.jsonContent)
        machines.insert(newMachine, at: 0)
        LibraryManager.shared.save(newMachine)
    }
    
    func closeTab(_ machine: StatechartWrapper) {
        guard let index = openMachines.firstIndex(where: { $0.id == machine.id }) else { return }
        
        openMachines.remove(at: index)
        
        // If we closed the active tab, select a neighbor
        if selectedMachine?.id == machine.id {
            if openMachines.isEmpty {
                selectedMachine = nil
            } else {
                // Select the one to the right, or the last one if at end
                let newIndex = min(index, openMachines.count - 1)
                selectedMachine = openMachines[newIndex]
            }
        }
    }
    
    // MARK: - AI Generation (Mock)
    // MARK: - AI Generation (Mock)
    func generateMachine(prompt: String) {
        let machine = SampleData.generateMachine(prompt: prompt)
        machines.insert(machine, at: 0)
        selectedMachine = machine
        LibraryManager.shared.save(machine)
    }

    // MARK: - Remote Generation (SAE Steering)
    
    // MARK: - Native & Remote Generation (SAE Steering)
    
    func generateRemote(prompt: String, steering: [Int: Double]) {
        // Phase 1: Try Native Service
        Task {
            if await SteeringService.shared.isModelLoaded == false {
                 try? await SteeringService.shared.loadModel()
            }
            
            // Map steering to native format (Mock vector for now)
            for (k, v) in steering {
                await SteeringService.shared.setSteering(featureID: k, strength: Float(v), vector: [1.0] /* Placeholder */)
            }
            
            do {
                let json = try await SteeringService.shared.generate(prompt: prompt)
                if json != "{}" {
                    await MainActor.run {
                        self.handleGeneratedJSON(prompt: prompt, json: json)
                    }
                    return
                }
            } catch {
                Logger.ai.error("Native Gen Failed: \(error.localizedDescription). Falling back to remote.")
            }
            
            // Fallback to Remote
            self.generateRemoteFallback(prompt: prompt, steering: steering)
        }
    }

    private func handleGeneratedJSON(prompt: String, json: String) {
        // Check if we are already viewing a steered version of this prompt
        let expectedName = "Steered: \(prompt.prefix(10))"
        
        if var current = self.selectedMachine, current.name == expectedName {
            // Update in place to preserve view state (if possible)
            if current.jsonContent != json {
                current.jsonContent = json
                self.selectedMachine = current
                if let index = self.machines.firstIndex(where: { $0.id == current.id }) {
                    self.machines[index] = current
                    LibraryManager.shared.save(current)
                }
            }
        } else {
            // Create new
            let machine = StatechartWrapper(name: expectedName, jsonContent: json)
            self.machines.insert(machine, at: 0)
            self.selectedMachine = machine
            LibraryManager.shared.save(machine)
        }
    }

    private func generateRemoteFallback(prompt: String, steering: [Int: Double]) {
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
            Logger.ai.error("Failed to encode request: \(error.localizedDescription)")
            return
        }
        
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                Logger.ai.error("Remote Gen Error: \(error.localizedDescription)")
                return
            }
            
            guard let data = data else { return }
            
            do {
                if let jsonResponse = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let scJSON = jsonResponse["json"] as? String {
                   
                    DispatchQueue.main.async {
                        self?.handleGeneratedJSON(prompt: prompt, json: scJSON)
                    }
                }
            } catch {
               Logger.ai.error("Failed to decode response: \(error.localizedDescription)")
            }
        }
        task.resume()
    }
    
    // MARK: - Deep Linking & restoration
    
    func restore(from url: URL) {
        if url.isFileURL {
            // Handle file import
            let secure = url.startAccessingSecurityScopedResource()
            defer { if secure { url.stopAccessingSecurityScopedResource() } }
            
            do {
                let content = try String(contentsOf: url, encoding: .utf8)
                let name = url.deletingPathExtension().lastPathComponent
                
                // Check if already exists by name (naive) or ID?
                if let index = machines.firstIndex(where: { $0.name == name }) {
                    Logger.data.info("Updating existing machine from file: \(name)")
                    machines[index].jsonContent = content
                    LibraryManager.shared.save(machines[index])
                    selectedMachine = machines[index]
                } else {
                    Logger.data.info("Importing new machine from file: \(name)")
                    let newMachine = StatechartWrapper(name: name, jsonContent: content)
                    machines.insert(newMachine, at: 0)
                    LibraryManager.shared.save(newMachine)
                    selectedMachine = newMachine
                }
            } catch {
                Logger.data.error("Failed to restore from file URL: \(error.localizedDescription)")
            }
        } else {
            // Handle Scheme: states://machine/<ID> or <Name>
            let path = url.lastPathComponent
            if let machine = machines.first(where: { $0.id.uuidString == path || $0.name == path }) {
                 selectedMachine = machine
            }
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
