import SwiftUI

// MARK: - Flow Background
struct FlowBackground: View {
    let scale: CGFloat
    let offset: CGSize
    
    var body: some View {
        Canvas { context, size in
            let spacing: CGFloat = 20 * scale
            // Dynamic dot size based on scale, but clamped
            let dotSize: CGFloat = max(1.5, 2 * scale)
            
            let center = CGPoint(x: size.width / 2 + offset.width, y: size.height / 2 + offset.height)
            
            var starX = center.x.remainder(dividingBy: spacing)
            if starX < 0 { starX += spacing }
            var startY = center.y.remainder(dividingBy: spacing)
            if startY < 0 { startY += spacing }
            
            for x in stride(from: starX, to: size.width, by: spacing) {
                for y in stride(from: startY, to: size.height, by: spacing) {
                    let rect = CGRect(x: x - dotSize/2, y: y - dotSize/2, width: dotSize, height: dotSize)
                    context.fill(Path(ellipseIn: rect), with: .color(Theme.Colors.gridDot))
                }
            }
        }
        .background(Theme.Colors.canvasBackground)
        .drawingGroup()
    }
}
