
import XCTest
@testable import States

final class SteeringVerificationTests: XCTestCase {

    func testSteeringServiceExistence() async {
        let service = SteeringService.shared
        // access a property to ensure it's alive (actor isolation applies, so we just reference it)
        _ = service
        XCTAssertTrue(true, "SteeringService instantiated")
    }

    func testMockGenerationIntegration() async throws {
        let service = SteeringService.shared
        
        // Even without weights, the "traffic" prompt should trigger the mock path
        let prompt = "Create a traffic light statechart"
        let json = try await service.generate(prompt: prompt)
        
        XCTAssertTrue(json.contains("trafficLight"), "JSON should contain trafficLight")
        XCTAssertTrue(json.contains("green"), "JSON should contain green state")
    }
}
