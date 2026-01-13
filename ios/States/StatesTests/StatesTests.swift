//
//  StatesTests.swift
//  StatesTests
//
//  Created by tmc on 12/26/25.
//

import Foundation
import Testing
@testable import States

struct AppViewModelTests {

    @Test func testMachineCRUD() async throws {
        let viewModel = AppViewModel()
        let initialCount = viewModel.machines.count
        let name = "UnitTest Machine \(UUID())"
        
        // Create
        let newMachine = viewModel.addMachine(name: name)
        #expect(viewModel.machines.count == initialCount + 1)
        #expect(viewModel.machines.contains(where: { $0.id == newMachine.id }))
        #expect(viewModel.selectedMachine?.id == newMachine.id)
        
        // Duplicate
        viewModel.duplicateMachine(newMachine)
        #expect(viewModel.machines.count == initialCount + 2)
        guard let duped = viewModel.machines.first(where: { $0.name == "\(name) Copy" }) else {
            #expect(Bool(false), "Duplicated machine not found")
            return
        }
        #expect(duped.id != newMachine.id)
        
        // Delete original and copy
        var index = viewModel.machines.firstIndex(where: { $0.id == newMachine.id })!
        viewModel.deleteMachine(at: IndexSet(integer: index))
        
        index = viewModel.machines.firstIndex(where: { $0.id == duped.id })!
        viewModel.deleteMachine(at: IndexSet(integer: index))
        
        #expect(viewModel.machines.count == initialCount)
    }
    
    @Test func testMockGeneration() async throws {
        let viewModel = AppViewModel()
        let prompt = "traffic light"
        
        viewModel.generateMachine(prompt: prompt)
        
        guard let machine = viewModel.selectedMachine else {
            #expect(Bool(false), "Selected machine should be set after generation")
            return
        }
        
        #expect(machine.jsonContent?.contains("trafficLight") == true)
        #expect(viewModel.machines.contains(where: { $0.id == machine.id }))
        
        // Cleanup
        if let index = viewModel.machines.firstIndex(where: { $0.id == machine.id }) {
            viewModel.deleteMachine(at: IndexSet(integer: index))
        }
    }
}

struct StatechartViewModelTests {
    
    @Test func testNodeManipulation() async throws {
        let machine = StatechartWrapper(name: "TestWrapper")
        let viewModel = StatechartViewModel(machine: machine)
        
        // 1. Add Node
        let initialCount = viewModel.nodes.count
        viewModel.addState(at: CGPoint(x: 100, y: 100))
        
        #expect(viewModel.nodes.count == initialCount + 1)
        guard let node = viewModel.nodes.last else {
            #expect(Bool(false), "Node was not added")
            return
        }
        #expect(node.position.x == 100)
        #expect(node.position.y == 100)
        #expect(viewModel.selection.contains(node.id))
        
        // 2. Duplicate Node
        viewModel.selection = [node.id]
        viewModel.duplicateSelection()
        
        #expect(viewModel.nodes.count == initialCount + 2)
        let dupedNode = viewModel.nodes.last!
        #expect(dupedNode.label == "\(node.label) Copy")
        #expect(dupedNode.id != node.id)
        
        // 3. Delete Selection
        viewModel.selection = [node.id, dupedNode.id]
        viewModel.deleteSelection()
        
        #expect(viewModel.nodes.count == initialCount)
    }
    
    @Test func testMarqueeSelection() async throws {
        let machine = StatechartWrapper(name: "MarqueeTest")
        let viewModel = StatechartViewModel(machine: machine)
        
        // Create 2 nodes
        viewModel.addState(at: CGPoint(x: 0, y: 0))
        let node1 = viewModel.nodes.last!
        
        viewModel.addState(at: CGPoint(x: 200, y: 200))
        let node2 = viewModel.nodes.last!
        
        // Clear selection
        viewModel.selection = []
        
        // 1. Marquee rect covering Node 1 only
        viewModel.startMarquee(at: .zero, isAdditive: false)
        viewModel.updateMarquee(rect: CGRect(x: -10, y: -10, width: 50, height: 50)) // Covers (0,0)
        
        #expect(viewModel.selection.contains(node1.id))
        #expect(!viewModel.selection.contains(node2.id))
        
        // 2. Marquee rect covering Node 2 only
        viewModel.startMarquee(at: CGPoint(x: 200, y: 200), isAdditive: false)
        viewModel.updateMarquee(rect: CGRect(x: 190, y: 190, width: 50, height: 50))
        
        #expect(!viewModel.selection.contains(node1.id))
        #expect(viewModel.selection.contains(node2.id))
        
        viewModel.endMarquee()
    }
}
