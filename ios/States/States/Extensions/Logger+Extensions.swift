import OSLog

extension Logger {
    /// Using the bundle identifier for the subsystem
    private static var subsystem = Bundle.main.bundleIdentifier ?? "dev.tmc.States"

    /// Logs for the Application lifecycle and high-level events
    static let app = Logger(subsystem: subsystem, category: "App")
    
    /// Logs for ViewModels
    static let viewModels = Logger(subsystem: subsystem, category: "ViewModels")
    
    /// Logs for Data Persistence and File Management
    static let data = Logger(subsystem: subsystem, category: "Data")
    
    /// Logs for AI and Steering Services
    static let ai = Logger(subsystem: subsystem, category: "AI")
    
    /// Logs for Statechart Logic and Execution
    static let statechart = Logger(subsystem: subsystem, category: "Statechart")
    
    /// Logs for UI and Rendering
    static let ui = Logger(subsystem: subsystem, category: "UI")
}
