//
//  StatesTests.swift
//  StatesTests
//
//  Created by tmc on 12/26/25.
//

import Foundation
import Testing
@testable import States

struct StatesTests {

    @Test func testAddMachine() async throws {
        let viewModel = AppViewModel()
        let initialCount = viewModel.machines.count
        
        viewModel.addMachine(name: "Test Machine")
        
        #expect(viewModel.machines.count == initialCount + 1)
        #expect(viewModel.machines.last?.name == "Test Machine")
    }
    
    @Test func testDeleteMachine() async throws {
        let viewModel = AppViewModel()
        viewModel.addMachine(name: "Test Machine 1")
        let initialCount = viewModel.machines.count
        
        // Find index of the newly added machine
        guard let index = viewModel.machines.firstIndex(where: { $0.name == "Test Machine 1" }) else {
            #expect(Bool(false), "Machine not found")
            return
        }
        
        viewModel.deleteMachine(at: IndexSet(integer: index))
        
        #expect(viewModel.machines.count == initialCount - 1)
        #expect(!viewModel.machines.contains(where: { $0.name == "Test Machine 1" }))
    }

}
