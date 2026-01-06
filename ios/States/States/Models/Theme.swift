import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

/// Centralized Design System for the States App.
/// Focuses on a "Stripe-like" aesthetic with semantic colors, variable usage for glassmorphism,
/// and platform-specific tuning.
enum Theme {
    
    // MARK: - Typography
    // Uses standard system fonts but tuned for modern readability (Inter-like).
    
    enum Typography {
        static let titleLarge = Font.system(size: 34, weight: .bold, design: .default)
        static let titleMedium = Font.system(size: 22, weight: .semibold, design: .default)
        static let body = Font.system(size: 15, weight: .regular, design: .default)
        static let caption = Font.system(size: 12, weight: .medium, design: .default)
        static let monospace = Font.system(size: 13, weight: .regular, design: .monospaced)
        
        static func nodeLabel(scale: CGFloat) -> Font {
            // Adaptive font size for nodes
            return .system(size: 16, weight: .medium, design: .rounded)
        }
    }
    
    // MARK: - Colors
    // Semantic definitions.
    
    enum Colors {
        static let accent = Color.indigo
        
        // Backgrounds
        #if os(macOS)
        static let canvasBackground = Color(nsColor: .windowBackgroundColor)
        static let sidebarBackground = Color(nsColor: .controlBackgroundColor).opacity(0.5) // Translucent
        #else
        static let canvasBackground = Color(uiColor: .systemGroupedBackground)
        static let sidebarBackground = Color(uiColor: .secondarySystemBackground)
        #endif
        
        // Node Styling
        static let nodeBackground = Color.white
        static let nodeBorder = Color.black.opacity(0.1)
        static let activeNodeGlow = Color.green.opacity(0.6)
        static let activeNodeTint = Color.green.opacity(0.1)
        
        // Glassmorphism
        static let glassMaterial: Material = .regular
        
        static var gridDot: Color {
            #if os(iOS)
            Color(uiColor: .label).opacity(0.1)
            #else
            Color(nsColor: .labelColor).opacity(0.1)
            #endif
        }
    }
    
    // MARK: - Layout & Metrics
    
    enum Layout {
        static let cornerRadius: CGFloat = 12.0
        static let nodeCornerRadius: CGFloat = 8.0
        static let shadowRadius: CGFloat = 8.0
        static let shadowY: CGFloat = 4.0
        
        #if os(macOS)
        static let sidebarWidth: CGFloat = 260
        #endif
        
        // Animations
        static let breathingAnimation: Animation = .easeInOut(duration: 2.5).repeatForever(autoreverses: true)
    }
    
    // MARK: - Haptics
    enum Haptics {
        enum FeedbackStyle {
            case light, medium, heavy
        }
        
        enum NotificationType {
            case success, warning, error
        }
        
        static func play(_ style: FeedbackStyle) {
            #if os(iOS)
            let uiStyle: UIImpactFeedbackGenerator.FeedbackStyle
            switch style {
            case .light:
                uiStyle = .light
            case .medium:
                uiStyle = .medium
            case .heavy:
                uiStyle = .heavy
            }
            let generator = UIImpactFeedbackGenerator(style: uiStyle)
            generator.prepare()
            generator.impactOccurred()
            #endif
        }
        
        static func notification(_ type: NotificationType) {
            #if os(iOS)
            let uiType: UINotificationFeedbackGenerator.FeedbackType
            switch type {
            case .success:
                uiType = .success
            case .warning:
                uiType = .warning
            case .error:
                uiType = .error
            }
            let generator = UINotificationFeedbackGenerator()
            generator.prepare()
            generator.notificationOccurred(uiType)
            #endif
        }
        
        static func selection() {
            #if os(iOS)
            let generator = UISelectionFeedbackGenerator()
            generator.prepare()
            generator.selectionChanged()
            #endif
        }
    }
    
    // MARK: - Shadows
    
    static func applySoftShadow(to view: any View) -> any View {
        return view
    }
}

// MARK: - View Extensions

extension View {
    func ultraThinGlass() -> some View {
        #if os(macOS)
        self.background(.ultraThinMaterial)
        #else
        self.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: Theme.Layout.cornerRadius))
        #endif
    }
    
    /// Applies the standard "Premium" shadow style
    func premiumShadow() -> some View {
        self.shadow(color: Color.black.opacity(0.08), radius: Theme.Layout.shadowRadius, x: 0, y: Theme.Layout.shadowY)
    }
}

