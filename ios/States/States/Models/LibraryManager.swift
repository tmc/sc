import Foundation

class LibraryManager {
    static let shared = LibraryManager()
    
    // Directory: Documents (User Visible)
    // Directory: iCloud Documents (if available) -> Local Documents
    private var libraryURL: URL? {
        if let iCloudURL = FileManager.default.url(forUbiquityContainerIdentifier: nil)?.appendingPathComponent("Documents") {
            return iCloudURL
        }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    }
    
    init() {
        createLibraryDirectoryIfNeeded()
    }
    
    private func createLibraryDirectoryIfNeeded() {
        guard let url = libraryURL else { return }
        if !FileManager.default.fileExists(atPath: url.path) {
            do {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            } catch {
                print("Failed to create library directory: \(error)")
            }
        }
    }
    
    func loadAll() -> [StatechartWrapper] {
        guard let url = libraryURL else { return [] }
        var machines: [StatechartWrapper] = []
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            for fileURL in fileURLs {
                if fileURL.pathExtension == "json" {
                    if let data = try? Data(contentsOf: fileURL),
                       let machine = try? JSONDecoder().decode(StatechartWrapper.self, from: data) {
                        machines.append(machine)
                    }
                }
            }
        } catch {
            print("Failed to load library: \(error)")
        }
        return machines
    }
    
    func save(_ machine: StatechartWrapper) {
        guard let url = libraryURL else { return }
        let fileURL = url.appendingPathComponent("\(machine.id.uuidString).json")
        
        do {
            let data = try JSONEncoder().encode(machine)
            try data.write(to: fileURL)
        } catch {
            print("Failed to save machine \(machine.name): \(error)")
        }
    }
    
    func delete(_ machine: StatechartWrapper) {
         guard let url = libraryURL else { return }
         let fileURL = url.appendingPathComponent("\(machine.id.uuidString).json")
         try? FileManager.default.removeItem(at: fileURL)
    }
    
    // MARK: - Simulation Persistence
    
    private var runsURL: URL? {
        return libraryURL?.appendingPathComponent("SimulationRuns")
    }
    
    func saveRun(_ run: SimulationRun) {
        guard let baseURL = runsURL else { return }
        let machineFolder = baseURL.appendingPathComponent(run.machineID.uuidString)
        
        do {
            if !FileManager.default.fileExists(atPath: machineFolder.path) {
                try FileManager.default.createDirectory(at: machineFolder, withIntermediateDirectories: true)
            }
            
            let fileURL = machineFolder.appendingPathComponent("\(run.id.uuidString).json")
            let data = try JSONEncoder().encode(run)
            try data.write(to: fileURL)
        } catch {
            print("Failed to save simulation run: \(error)")
        }
    }
    
    func loadRuns(for machineID: UUID) -> [SimulationRun] {
        guard let baseURL = runsURL else { return [] }
        let machineFolder = baseURL.appendingPathComponent(machineID.uuidString)
        var runs: [SimulationRun] = []
        
        guard FileManager.default.fileExists(atPath: machineFolder.path) else { return [] }
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: machineFolder, includingPropertiesForKeys: nil)
            for fileURL in fileURLs {
                if fileURL.pathExtension == "json" {
                    if let data = try? Data(contentsOf: fileURL),
                       let run = try? JSONDecoder().decode(SimulationRun.self, from: data) {
                        runs.append(run)
                    }
                }
            }
        } catch {
            print("Failed to load runs: \(error)")
        }
        
        return runs.sorted { $0.timestamp > $1.timestamp }
    }
}
