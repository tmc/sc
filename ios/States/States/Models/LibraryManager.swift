import Foundation

class LibraryManager {
    static let shared = LibraryManager()
    
    // Directory: Documents (User Visible)
    private var libraryURL: URL? {
        // Use Documents directory for user visibility (Files app support)
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        // Optional: Check if we are using iCloud and use url(forUbiquityContainerIdentifier:)
        // But for now, local Documents is the baseline.
        return documents
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
}
