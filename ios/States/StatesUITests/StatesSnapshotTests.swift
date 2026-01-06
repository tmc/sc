
import XCTest
import SwiftUI
@testable import States

final class StatesSnapshotTests: XCTestCase {
    
    override func setUpWithError() throws {
        continueAfterFailure = false
        // Landscape orientation for better canvas view
        XCUIDevice.shared.orientation = .landscapeLeft
    }

    @MainActor
    func testVisualizerSnapshot() throws {
        let app = XCUIApplication()
        app.launch()
        
        // 2) Tap the "kettle" machine to open the visualizer
        let machineButton = app.buttons["kettle"]
        XCTAssertTrue(machineButton.waitForExistence(timeout: 5), "Machine 'kettle' not found in list")
        machineButton.tap()
        
        // 3) Wait for the visualizer to load and nodes to appear
        // The "lukewarm" node should be visible if rendering works
        let simulateButton = app.buttons["Simulate"]
        XCTAssertTrue(simulateButton.waitForExistence(timeout: 5))
        let nodeText = app.staticTexts["lukewarm"]
        
        // Take Snapshot of the Canvas
        // We look for a known element or just take entire screen
        // The canvas accessibility isn't fully set yet, so screen check is best.
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = "Visualizer_Canvas_Render"
        attachment.lifetime = .keepAlways
        add(attachment)
        
        // Check for node existence by label via Accessibility Overlay
        // Check for node existence by label via Accessibility Overlay
        // "kettle" machine has "off", "heating", "boiling", "lukewarm" states
        let lukewarmState = app.staticTexts["lukewarm"]
        XCTAssertTrue(lukewarmState.waitForExistence(timeout: 5), "Node 'lukewarm' text not found")
    }
}
