import Foundation

struct SampleData {
    // A minimal set of sample machines to keep the app clean.
    static let machines: [String: String] = [
        "01 Welcome": """
        {
          "name": "Welcome",
          "root_state": {
            "label": "Welcome",
            "type": "OR",
            "is_initial": true,
            "children": [
              { "label": "Start", "type": "BASIC", "is_initial": true },
              { "label": "Exploring", "type": "BASIC" },
              { "label": "Finished", "type": "BASIC" }
            ]
          },
          "transitions": [
            { "from": ["Start"], "to": ["Exploring"], "event": "TAP" },
            { "from": ["Exploring"], "to": ["Finished"], "event": "DONE" },
            { "from": ["Finished"], "to": ["Start"], "event": "RESET" }
          ]
        }
        """,
        "02 Traffic Light": """
        {
          "name": "Traffic Light",
          "root_state": {
            "label": "Root",
            "type": "OR",
            "is_initial": true,
            "children": [
              { "label": "Red", "type": "BASIC", "is_initial": true },
              { "label": "Green", "type": "BASIC" },
              { "label": "Yellow", "type": "BASIC" }
            ]
          },
          "transitions": [
            { "from": ["Red"], "to": ["Green"], "event": "TIMER" },
            { "from": ["Green"], "to": ["Yellow"], "event": "TIMER" },
            { "from": ["Yellow"], "to": ["Red"], "event": "TIMER" }
          ]
        }
        """,
        "03 Media Player": """
        {
          "name": "Media Player",
          "root_state": {
            "label": "Player",
            "type": "OR",
            "is_initial": true,
            "children": [
              { "label": "Stopped", "type": "BASIC", "is_initial": true },
              { "label": "Playing", "type": "BASIC" },
              { "label": "Paused", "type": "BASIC" }
            ]
          },
          "transitions": [
            { "from": ["Stopped"], "to": ["Playing"], "event": "PLAY" },
            { "from": ["Playing"], "to": ["Paused"], "event": "PAUSE" },
            { "from": ["Playing"], "to": ["Stopped"], "event": "STOP" },
            { "from": ["Paused"], "to": ["Playing"], "event": "PLAY" },
            { "from": ["Paused"], "to": ["Stopped"], "event": "STOP" }
          ]
        }
        """,
        "04 Authentication": """
        {
          "name": "Authentication",
          "root_state": {
            "label": "Auth",
            "type": "OR",
            "is_initial": true,
            "children": [
              { "label": "LoggedOut", "type": "BASIC", "is_initial": true },
              { "label": "Authenticating", "type": "BASIC" },
              { "label": "LoggedIn", "type": "BASIC" },
              { "label": "Error", "type": "BASIC" }
            ]
          },
          "transitions": [
            { "from": ["LoggedOut"], "to": ["Authenticating"], "event": "LOGIN" },
            { "from": ["Authenticating"], "to": ["LoggedIn"], "event": "SUCCESS" },
            { "from": ["Authenticating"], "to": ["Error"], "event": "FAILURE" },
            { "from": ["LoggedIn"], "to": ["LoggedOut"], "event": "LOGOUT" },
            { "from": ["Error"], "to": ["Authenticating"], "event": "RETRY" },
            { "from": ["Error"], "to": ["LoggedOut"], "event": "CANCEL" }
          ]
        }
        """
    ]

    static func generateMachine(prompt: String) -> StatechartWrapper {
        let name = "Generated: \(prompt.prefix(15))..."
        var machine = StatechartWrapper(name: name)
        
        let lowerPrompt = prompt.lowercased()
        
        if lowerPrompt.contains("traffic") {
            let json = """
            {
              "name": "trafficLight",
              "root_state": {
                "label": "trafficLight",
                "type": "OR",
                "is_initial": true,
                "children": [
                  { "label": "green", "type": "BASIC", "is_initial": true },
                  { "label": "yellow", "type": "BASIC" },
                  { "label": "red", "type": "BASIC" }
                ]
              },
              "transitions": [
                { "from": ["green"], "to": ["yellow"], "event": "TIMER" },
                { "from": ["yellow"], "to": ["red"], "event": "TIMER" },
                { "from": ["red"], "to": ["green"], "event": "TIMER" }
              ]
            }
            """
            machine.jsonContent = json
        } else if lowerPrompt.contains("login") || lowerPrompt.contains("auth") {
             // Login Flow Mock
             let json = """
             {
               "name": "loginFlow",
               "root_state": {
                 "label": "loginFlow",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "idle", "type": "BASIC", "is_initial": true },
                   { "label": "authenticating", "type": "BASIC" },
                   { "label": "loggedIn", "type": "BASIC" },
                   { "label": "error", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["idle"], "to": ["authenticating"], "event": "LOGIN" },
                 { "from": ["authenticating"], "to": ["loggedIn"], "event": "SUCCESS" },
                 { "from": ["authenticating"], "to": ["error"], "event": "FAILURE" },
                 { "from": ["loggedIn"], "to": ["idle"], "event": "LOGOUT" },
                 { "from": ["error"], "to": ["authenticating"], "event": "RETRY" },
                 { "from": ["error"], "to": ["idle"], "event": "CANCEL" }
               ]
             }
             """
             machine.jsonContent = json
        } else if lowerPrompt.contains("music") || lowerPrompt.contains("player") {
             // Music Player Mock
             let json = """
             {
               "name": "musicPlayer",
               "root_state": {
                 "label": "musicPlayer",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "stopped", "type": "BASIC", "is_initial": true },
                   { "label": "playing", "type": "BASIC" },
                   { "label": "paused", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["stopped"], "to": ["playing"], "event": "PLAY" },
                 { "from": ["playing"], "to": ["paused"], "event": "PAUSE" },
                 { "from": ["playing"], "to": ["stopped"], "event": "STOP" },
                 { "from": ["paused"], "to": ["playing"], "event": "PLAY" },
                 { "from": ["paused"], "to": ["stopped"], "event": "STOP" }
               ]
             }
             """
             machine.jsonContent = json
        } else if lowerPrompt.contains("toggle") || lowerPrompt.contains("switch") {
             // Toggle Mock
             let json = """
             {
               "name": "toggle",
               "root_state": {
                 "label": "toggle",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "inactive", "type": "BASIC", "is_initial": true },
                   { "label": "active", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["inactive"], "to": ["active"], "event": "TOGGLE" },
                 { "from": ["active"], "to": ["inactive"], "event": "TOGGLE" }
               ]
             }
             """
             machine.jsonContent = json
        } else {
            // Generic Mock for unknown
             let json = """
             {
               "name": "generic",
               "root_state": {
                 "label": "generic",
                 "type": "OR",
                 "is_initial": true,
                 "children": [
                   { "label": "start", "type": "BASIC", "is_initial": true },
                   { "label": "process", "type": "BASIC" },
                   { "label": "end", "type": "BASIC" },
                   { "label": "fail", "type": "BASIC" }
                 ]
               },
               "transitions": [
                 { "from": ["start"], "to": ["process"], "event": "NEXT" },
                 { "from": ["process"], "to": ["end"], "event": "COMPLETE" },
                 { "from": ["process"], "to": ["fail"], "event": "ERROR" },
                 { "from": ["fail"], "to": ["process"], "event": "RETRY" }
               ]
             }
             """
             machine.jsonContent = json
        }
        return machine
    }
}
