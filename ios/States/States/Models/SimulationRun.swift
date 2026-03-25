import Foundation

struct SimulationRun: Codable, Identifiable, Sendable {
    let id: UUID
    let machineID: UUID
    let timestamp: Date
    let steps: [[UUID]] // List of active state IDs per step
    
    // Metadata
    let name: String? // User might name a run?
    
    init(id: UUID = UUID(), machineID: UUID, timestamp: Date = Date(), steps: [[UUID]], name: String? = nil) {
        self.id = id
        self.machineID = machineID
        self.timestamp = timestamp
        self.steps = steps
        self.name = name
    }
}
