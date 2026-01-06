//
//  GuardEvaluator.swift
//  States
//
//  Created by Assistant on 1/3/25.
//

import Foundation
import JavaScriptCore

class GuardEvaluator {
    
    /// Evaluates a guard expression against a given context.
    /// - Parameters:
    ///   - expression: The boolean expression (e.g., "context.count > 5").
    ///   - context: A dictionary representing the machine's data context.
    /// - Returns: True if truthy, false otherwise. Returns true if expression is empty (default pass).
    func evaluate(expression: String, context: [String: Any]) -> Bool {
        let trimmed = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return true
        }
        
        let jsContext = JSContext()
        
        // Inject context safely
        jsContext?.setObject(context, forKeyedSubscript: "context" as NSString)
        
        // Error handling
        jsContext?.exceptionHandler = { context, exception in
            print("[GuardEvaluator] JS Exception: \(String(describing: exception))")
        }
        
        // Evaluate
        if let result = jsContext?.evaluateScript(trimmed) {
            return result.toBool()
        }
        
        return false // Fail safe
    }
    
    /// Executes an action script.
    /// - Returns: The modified context if changed, or nil.
    func executeAction(script: String, context: [String: Any]) -> [String: Any]? {
         let trimmed = script.trimmingCharacters(in: .whitespacesAndNewlines)
         if trimmed.isEmpty {
             return nil
         }
         
         let jsContext = JSContext()
         jsContext?.setObject(context, forKeyedSubscript: "context" as NSString)
         
         jsContext?.exceptionHandler = { context, exception in
             print("[GuardEvaluator] Action Exception: \(String(describing: exception))")
         }
         
         _ = jsContext?.evaluateScript(trimmed)
         
         // Extract modified context
         if let newContextVal = jsContext?.objectForKeyedSubscript("context") {
             return newContextVal.toDictionary() as? [String: Any]
         }
         
         return nil
    }
}
