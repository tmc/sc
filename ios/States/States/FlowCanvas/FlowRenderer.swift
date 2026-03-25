import SwiftUI

struct FlowRenderer {
    var scale: CGFloat
    var offset: CGSize
    var currentDragOffset: CGSize
    var selection: Set<UUID>
    var activeStateIDs: Set<UUID>
    var hoveredNodeID: UUID?
    var hoveredParentID: UUID?
    var connectingEdge: (source: UUID, currentPoint: CGPoint)?
    
    // MARK: - Drawing Logic
    
    func drawGrid(context: GraphicsContext, size: CGSize) {
        let gridStep: CGFloat = 20.0 * scale
        let gridColor = Color.gray.opacity(0.2)
        
        let path = Path { path in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let totalOffsetX = offset.width + currentDragOffset.width
            let totalOffsetY = offset.height + currentDragOffset.height
            
            let phaseX = (center.x + totalOffsetX).truncatingRemainder(dividingBy: gridStep)
            let phaseY = (center.y + totalOffsetY).truncatingRemainder(dividingBy: gridStep)
            
            for x in stride(from: phaseX - gridStep, to: size.width + gridStep, by: gridStep) {
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: size.height))
            }
            
            for y in stride(from: phaseY - gridStep, to: size.height + gridStep, by: gridStep) {
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
            }
        }
        
        context.stroke(path, with: .color(gridColor), lineWidth: 1)
        
        // Draw Origin Cross
        let originX = size.width / 2 + (offset.width + currentDragOffset.width)
        let originY = size.height / 2 + (offset.height + currentDragOffset.height)
        
        var originPath = Path()
        originPath.move(to: CGPoint(x: originX, y: originY - 10))
        originPath.addLine(to: CGPoint(x: originX, y: originY + 10))
        originPath.move(to: CGPoint(x: originX - 10, y: originY))
        originPath.addLine(to: CGPoint(x: originX + 10, y: originY))
        
        context.stroke(originPath, with: .color(.black.opacity(0.5)), lineWidth: 2)
    }

    func drawNode(context: GraphicsContext, node: FlowNode, phase: TimeInterval) {
        let nodeRect = CGRect(origin: node.position, size: node.size)
        
        // Use Theme
        let isSelected = selection.contains(node.id)
        let isActive = activeStateIDs.contains(node.id)
        let isHovered = (hoveredNodeID == node.id)
        
        // Corner Radius
        var cornerRadius = Theme.Layout.nodeCornerRadius
        if node.type == .compound || node.type == .parallel { cornerRadius = 12 }
        if node.type == .history || node.type == .final { cornerRadius = node.size.width / 2 }
        
        // Continuous Curvature (Squircle)
        let path = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous).path(in: nodeRect)
        
        // 1. Shadows (High Quality) - LOD: Disable at low zoom
        if !isActive && scale > 0.4 {
            var shadowContext = context
            // Dual Shadow for Depth (Diffused + Contact)
            shadowContext.addFilter(.shadow(color: Color.black.opacity(0.06), radius: 10, x: 0, y: 5)) // Ambient
            shadowContext.addFilter(.shadow(color: Color.black.opacity(0.08), radius: 2, x: 0, y: 1))  // Contact
            shadowContext.fill(path, with: .color(.white)) // Invisible fill to cast shadow
        }
        
        // 2. Active Glow & Breathing
        if isActive {
            var glowContext = context
            
            // Calculate Breathing: Oscillate between 0.0 and 1.0 roughly every 2.5s
            let breath = (sin(phase * 2.5) + 1) / 2
            // Map to opacity range [0.3, 0.7]
            let opacity = 0.3 + (breath * 0.4)

            // Layer 1: Wide, breathing glow (Fixed radius to avoid pipeline churn)
            glowContext.addFilter(.shadow(color: Theme.Colors.activeNodeGlow.opacity(opacity), radius: 10, x: 0, y: 0))
            glowContext.stroke(path, with: .color(Theme.Colors.activeNodeGlow), lineWidth: 4 / scale) 
            
            // Layer 2: Core brightness
            let coreContext = context
            coreContext.stroke(path, with: .color(.white.opacity(0.4)), lineWidth: 1 / scale)
        }
        
        // 2b. Reparent Highlight
        if hoveredParentID == node.id {
             var highlightContext = context
             highlightContext.addFilter(.shadow(color: Theme.Colors.accent.opacity(0.6), radius: 10, x: 0, y: 0))
             highlightContext.stroke(path, with: .color(Theme.Colors.accent), lineWidth: 4 / scale)
        }
        
        // 2c. Hover Highlight (Subtle)
        if isHovered && !isSelected && !isActive {
             let hoverContext = context
             // Light inner glow or border boost
             hoverContext.stroke(path, with: .color(Theme.Colors.nodeBorder.opacity(0.5)), lineWidth: 2 / scale)
        }
        
        // 3. Fill
        var fillColor: Color
        switch node.type {
        case .atomic:
            fillColor = Theme.Colors.nodeBackground
            if isActive { fillColor = Theme.Colors.activeNodeTint }
        case .compound:
            fillColor = Color.white.opacity(0.9)
        case .parallel:
            fillColor = .clear
        case .final:
            fillColor = .primary 
        case .history:
            fillColor = .yellow.opacity(0.2)
        }
        
        context.fill(path, with: .color(fillColor))
        
        // 4. Compound Header
        if node.type == .compound {
            // Header Logic
            let headerHeight: CGFloat = 30
            // Just draw line
            var headerSeparator = Path()
            headerSeparator.move(to: CGPoint(x: nodeRect.minX, y: nodeRect.minY + headerHeight))
            headerSeparator.addLine(to: CGPoint(x: nodeRect.maxX, y: nodeRect.minY + headerHeight))
            context.stroke(headerSeparator, with: .color(Theme.Colors.nodeBorder.opacity(0.5)), lineWidth: 1 / scale)
        }
        
        // 5. Borders / Selection Halo
        if isSelected {
             // Focus Ring (Double Stroke)
             let ringPath = path
             context.stroke(ringPath, with: .color(Theme.Colors.accent.opacity(0.4)), lineWidth: 6.0 / scale)
             context.stroke(ringPath, with: .color(Theme.Colors.accent), lineWidth: 2.0 / scale)
        } else {
             // Standard Border
             var strokeStyle = StrokeStyle(lineWidth: 1.0 / scale)
             if node.type == .parallel { strokeStyle.dash = [6/scale, 4/scale] }
             context.stroke(path, with: .color(Theme.Colors.nodeBorder), style: strokeStyle)
        }
        
        // 6. Typography
        if scale > 0.4 {
            let resolvedFont = Theme.Typography.nodeLabel(scale: scale)
            let textColor = Color.black.opacity(0.85)
            
            var textPoint = CGPoint(x: nodeRect.midX, y: nodeRect.midY)
            
            if node.type == .compound || node.type == .parallel {
                textPoint = CGPoint(x: nodeRect.minX + 10, y: nodeRect.minY + 14) // Adjusted for header
                context.draw(Text(node.label).font(.system(size: 13, weight: .semibold)).foregroundStyle(Color.black.opacity(0.6)), at: textPoint, anchor: .leading)
            } else if node.type != .final {
                 context.draw(Text(node.label).font(resolvedFont).foregroundStyle(textColor), at: textPoint)
            }
        }
        
        // 7. Connection Ports
        if (connectingEdge != nil || isSelected) && scale > 0.4 {
            drawPorts(context: context, rect: nodeRect, isHovered: false) 
        }
    }
    
    func drawArrow(context: GraphicsContext, endPoint: CGPoint, startPoint: CGPoint, color: Color) {
        let dx = endPoint.x - startPoint.x
        let dy = endPoint.y - startPoint.y
        let angle = atan2(dy, dx)
        
        let arrowSize: CGFloat = 8 / scale 
        
        var arrowPath = Path()
        arrowPath.move(to: endPoint)
        arrowPath.addLine(to: CGPoint(
            x: endPoint.x - arrowSize * cos(angle - .pi / 6),
            y: endPoint.y - arrowSize * sin(angle - .pi / 6)
        ))
        arrowPath.move(to: endPoint)
        arrowPath.addLine(to: CGPoint(
            x: endPoint.x - arrowSize * cos(angle + .pi / 6),
            y: endPoint.y - arrowSize * sin(angle + .pi / 6)
        ))
        
        context.stroke(arrowPath, with: .color(color), lineWidth: 2 / scale)
    }
    
    func drawEdge(context: GraphicsContext, source: FlowNode, target: FlowNode, edge: FlowEdge) {
        let sourceRect = CGRect(origin: source.position, size: source.size)
        let targetRect = CGRect(origin: target.position, size: target.size)
        
        let (startPoint, endPoint) = Self.calculateConnectionPoints(from: sourceRect, to: targetRect)
        
        var path = Path()
        path.move(to: startPoint)
        
        let isReflexive = (source.id == target.id)
        if isReflexive {
            // Self-loop
            let control1 = CGPoint(x: startPoint.x + 50, y: startPoint.y - 50)
            let control2 = CGPoint(x: startPoint.x - 50, y: startPoint.y - 50)
            path.addCurve(to: endPoint, control1: control1, control2: control2)
        } else {
             // Simple Orthogonal Fallback
             if let waypoints = edge.waypoints, !waypoints.isEmpty {
                 for point in waypoints { path.addLine(to: point) }
                 path.addLine(to: endPoint)
             } else {
                 let midY = (startPoint.y + endPoint.y) / 2
                 path.addLine(to: CGPoint(x: startPoint.x, y: midY))
                 path.addLine(to: CGPoint(x: endPoint.x, y: midY))
                 path.addLine(to: endPoint)
             }
        }
        
        let isSelected = selection.contains(edge.id)
        let color = isSelected ? Theme.Colors.accent : Color.gray.opacity(0.8)
        let width: CGFloat = (isSelected ? 3 : 2) / scale
        
        context.stroke(path, with: .color(color), lineWidth: width)
        drawArrow(context: context, endPoint: endPoint, startPoint: startPoint, color: color)
    }
    
    func drawPorts(context: GraphicsContext, rect: CGRect, isHovered: Bool) {
        let ports = [
            CGPoint(x: rect.midX, y: rect.minY), // Top
            CGPoint(x: rect.maxX, y: rect.midY), // Right
            CGPoint(x: rect.midX, y: rect.maxY), // Bottom
            CGPoint(x: rect.minX, y: rect.midY)  // Left
        ]
        
        for port in ports {
            let portPath = Path(ellipseIn: CGRect(x: port.x - 4, y: port.y - 4, width: 8, height: 8))
            context.fill(portPath, with: .color(.white))
            context.stroke(portPath, with: .color(Theme.Colors.accent), lineWidth: 1.5)
        }
    }
    
    func drawConnectionDrag(context: GraphicsContext, source: FlowNode, endPoint: CGPoint) {
        let sourceRect = CGRect(origin: source.position, size: source.size)
        let startPoint = CGPoint(x: sourceRect.midX, y: sourceRect.maxY)
        
        var path = Path()
        path.move(to: startPoint)
        path.addLine(to: endPoint)
        
        context.stroke(path, with: .color(Theme.Colors.accent), style: StrokeStyle(lineWidth: 2, dash: [5, 5]))
    }

    static func calculateConnectionPoints(from sourceSpy: CGRect, to targetSpy: CGRect) -> (CGPoint, CGPoint) {
        if targetSpy.minY >= sourceSpy.maxY {
             return (CGPoint(x: sourceSpy.midX, y: sourceSpy.maxY), CGPoint(x: targetSpy.midX, y: targetSpy.minY))
        } else if targetSpy.maxY <= sourceSpy.minY {
             return (CGPoint(x: sourceSpy.midX, y: sourceSpy.minY), CGPoint(x: targetSpy.midX, y: targetSpy.maxY))
        } else if targetSpy.minX >= sourceSpy.maxX {
             return (CGPoint(x: sourceSpy.maxX, y: sourceSpy.midY), CGPoint(x: targetSpy.minX, y: targetSpy.midY))
        } else {
             return (CGPoint(x: sourceSpy.minX, y: sourceSpy.midY), CGPoint(x: targetSpy.maxX, y: targetSpy.midY))
        }
    }
    
    // Helper for Edge Intersection
    static func intersectionPoint(rect: CGRect, center: CGPoint, other: CGPoint) -> CGPoint {
        let dx = other.x - center.x
        let dy = other.y - center.y
        
        let w = rect.width / 2
        let h = rect.height / 2
        
        if abs(dx) > 0 {
            let signX: CGFloat = dx > 0 ? 1 : -1
            let testX = w * signX
            let testY = testX * (dy/dx) 
            
            if abs(testY) <= h {
                return CGPoint(x: center.x + testX, y: center.y + testY)
            }
        }
        
        if abs(dy) > 0 {
            let signY: CGFloat = dy > 0 ? 1 : -1
            let testY = h * signY
            let testX = testY * (dx/dy) 
            
            return CGPoint(x: center.x + testX, y: center.y + testY)
        }
        
        return center
    }
}

extension CGRect {
    var center: CGPoint {
        CGPoint(x: midX, y: midY)
    }
}
