
import Foundation

// --- Structs from StatechartWrapper.swift ---
struct StatelyDocsMachine: Codable {
    let id: String
    let name: String
    let definition: StatelyDefinition
}

struct StatelyDefinition: Codable {
    let id: String
    let rootNode: StatelyNode
    let edges: [StatelyEdge]
}

struct StatelyNode: Codable {
    let id: String
    let data: StatelyNodeData?
    let nodes: [StatelyNode]?
    let position: StatelyPosition?
    let size: StatelySize?
    let type: String?
}

struct StatelyNodeData: Codable {
    let key: String?
    let initial: String?
    let type: String?
}

struct StatelyPosition: Codable {
    let x: Double
    let y: Double
}

struct StatelySize: Codable {
    let width: Double
    let height: Double
}

struct StatelyEdge: Codable {
    let id: String
    let source: String
    let target: String?
    let data: StatelyEdgeData?
}

struct StatelyEdgeData: Codable {
    let eventTypeData: StatelyEventTypeData?
}

struct StatelyEventTypeData: Codable {
    let eventType: String?
}

// --- Sample Data Snippet ---
let jsonString = """
[   {     "id": "b028164c-d1b8-4bd8-a5d5-cb8e0e1fcd75",     "name": "feedback machine",     "definition": {       "id": "b028164c-d1b8-4bd8-a5d5-cb8e0e1fcd75",       "edges": [         {           "id": "feedback machine.question#feedback.good[0]",           "data": {             "actions": [               {                 "kind": "named",                 "action": {                   "type": "track",                   "params": {                     "response": "good"                   }                 }               }             ],             "metaEntries": [],             "eventTypeData": {               "type": "named",               "eventType": "feedback.good"             }           },           "size": {             "width": 176,             "height": 115           },           "source": "feedback machine.question",           "target": "feedback machine.question",           "position": {             "x": 213,             "y": 122           },           "uniqueId": "ncp7gduj9w"         }       ],       "context": {},       "schemas": {         "tags": {},         "input": null,         "actors": {},         "delays": {},         "events": {           "feedback.good": {             "type": "object",             "properties": {}           }         },         "guards": {},         "output": null,         "actions": {},         "context": {}       },       "rootNode": {         "id": "feedback machine",         "data": {           "key": "feedback machine",           "exit": [],           "tags": [],           "entry": [],           "assets": [],           "invoke": [],           "initial": "question",           "metaEntries": []         },         "size": {           "width": 546,           "height": 30         },         "nodes": [ ],         "position": {           "x": -109,           "y": 22         },         "uniqueId": "i7s5dbzu1ql"       },       "implementations": { }     },     "projectVersionId": "c447d996-cef1-421d-a422-8be695668764"   } ]
"""

// --- Test ---
if let data = jsonString.data(using: .utf8) {
    do {
        let machines = try JSONDecoder().decode([StatelyDocsMachine].self, from: data)
        print("Success! Parsed \(machines.count) machines.")
    } catch {
        print("Error: \(error)")
    }
} else {
    print("Failed to convert string to data")
}
