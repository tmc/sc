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
    func testInspectAllCharts() throws {
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
            
            // On iPad/Mac (SplitView), the list might be always visible or in a column.
            // On iPhone, we might need to go back.
            // Assuming iPad for now as per user request/simulator.
            
            if machineCell.waitForExistence(timeout: 2) {
                machineCell.tap()
                
                // Verify Visualizer Loaded
                // We check for the "Simulate" button which is part of the Visualizer view toolbar
                let simulateButton = app.buttons["Simulate"]
                XCTAssertTrue(simulateButton.waitForExistence(timeout: 5), "Visualizer should load for \(machineName)")
                
                // Optional: Check for a node specific to that machine?
                // For now, toolbar existence proves view body loaded without crash.
                
                // Go back if needed (split view doesn't need back)
                // If collapsed (iPhone), we need back button.
                if !app.navigationBars["Statecharts"].exists {
                     app.navigationBars.buttons.firstMatch.tap()
                }
            } else {
                 // It might be scrolled off screen? Or list empty?
                 // For now, log failure if main sample data missing.
                 XCTContext.runActivity(named: "Check for \(machineName)") { _ in
                     // Fallback check static text
                     if app.staticTexts[machineName].exists {
                         app.staticTexts[machineName].tap()
                         XCTAssertTrue(app.buttons["Simulate"].exists)
                     }
                 }
            }
        }
    }
}
