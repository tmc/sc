//
//  StatesUITests.swift
//  StatesUITests
//
//  Created by tmc on 12/26/25.
//

import XCTest

final class StatesUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    override func tearDownWithError() throws {
    }

    @MainActor
    func testCreationFlow() throws {
        let app = XCUIApplication()
        app.launch()
        
        // 1. Create New Chart
        let newChartButton = app.buttons["New Chart"]
        XCTAssertTrue(newChartButton.waitForExistence(timeout: 2), "New Chart button should exist")
        newChartButton.tap()
        
        // 2. Find and Enter New Chart
        // It's usually inserted at the top or sorted. Name is "New Chart".
        let newChartCell = app.buttons["New Chart"]
        XCTAssertTrue(newChartCell.waitForExistence(timeout: 2), "New Chart row should appear")
        newChartCell.firstMatch.tap()
        
        // 3. Verify Visualizer Loaded by checking for "Add State" toolbar button
        let addStateButton = app.buttons["Add State"]
        XCTAssertTrue(addStateButton.waitForExistence(timeout: 5), "Visualizer should load")
        
        // 4. Add a State
        addStateButton.tap()
        
        // 5. Verify Node Appears
        // Nodes are accessible elements labeled by their text.
        // Default label is "New State"
        let newNode = app.buttons["New State"] // Or other element type depending on modifier
        // FlowView nodes use .accessibilityAddTraits(.isButton)
        XCTAssertTrue(newNode.waitForExistence(timeout: 2), "New State node should appear on canvas")
        
        // 6. Navigation Back (iPhone only, but safe to check existence)
        if app.navigationBars.buttons.firstMatch.exists && !app.navigationBars["Statecharts"].exists {
            app.navigationBars.buttons.firstMatch.tap()
        }
    }

    @MainActor
    func testInspectSampleData() throws {
        let app = XCUIApplication()
        app.launch()

        // 1. Verify Sidebar / Machine List
        let sidebarTitle = app.navigationBars["Statecharts"]
        XCTAssertTrue(sidebarTitle.exists, "Sidebar title should exist")
        
        // List of machines expected in SampleData
        let expectedMachines = ["Traffic Light", "Standard Machine", "Nested", "History Example"]
        
        for machineName in expectedMachines {
            // Find the cell
            let machineCell = app.buttons[machineName]
            
            if machineCell.waitForExistence(timeout: 2) {
                machineCell.firstMatch.tap()
                
                // Verify Visualizer Loaded
                let simulateButton = app.buttons["Simulate"]
                XCTAssertTrue(simulateButton.waitForExistence(timeout: 5), "Visualizer should load for \(machineName)")
                
                // Go back if needed (iPhone/collapsed split)
                if !app.navigationBars["Statecharts"].exists && app.navigationBars.buttons.count > 0 {
                     app.navigationBars.buttons.firstMatch.tap()
                }
            } else {
                 XCTContext.runActivity(named: "Check for \(machineName)") { _ in
                     // Fallback check static text if it's not a button?
                     if app.staticTexts[machineName].exists {
                         app.staticTexts[machineName].tap()
                         XCTAssertTrue(app.buttons["Simulate"].exists)
                     }
                 }
            }
        }
    }
}
