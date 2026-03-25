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
    
    // MARK: - Colors
    // Semantic definitions.
    
    enum Colors {
        static let accent = Color.indigo
        
        // Backgrounds
        static let canvasBackground: Color = {
            return Color(red: 15/255, green: 23/255, blue: 42/255) // #0f172a (Midnight)
        }()
        
        static let sidebarBackground: Color = {
             #if os(macOS)
             return Color.clear // Use native vibrancy
             #else
             return Color(uiColor: .secondarySystemBackground)
             #endif
        }()
        
        static let separator: Color = {
            #if os(macOS)
            return Color(nsColor: .separatorColor)
            #else
            return Color(uiColor: .separator)
            #endif
        }()
        
        // Node Styling
        static let nodeBackground = Color.white
        static let nodeBorder = Color.primary.opacity(0.1) // Adaptive border
        static let activeNodeGlow = Color.green.opacity(0.6)
        static let activeNodeTint = Color.green.opacity(0.1)
        
        // Glassmorphism
        static let glassMaterial: Material = .regular
        
        static var gridDot: Color {
            return Color(red: 51/255, green: 65/255, blue: 85/255).opacity(0.3) // Slate 700
        }
    }
    
    // MARK: - Layout & Metrics
    
    enum Layout {
        static let cornerRadius: CGFloat = 12.0
        static let nodeCornerRadius: CGFloat = 8.0
        static let shadowRadius: CGFloat = 12.0
        static let shadowY: CGFloat = 6.0
        
        #if os(macOS)
        static let sidebarWidth: CGFloat = 260
        #endif
        
        // Animations
        static let breathingAnimation: Animation = .easeInOut(duration: 2.5).repeatForever(autoreverses: true)
        static let springStart: Animation = .spring(response: 0.3, dampingFraction: 0.7)
        static let springEnd: Animation = .spring(response: 0.4, dampingFraction: 0.6)
    }
    
    // MARK: - Haptics
    enum Haptics {
        enum FeedbackStyle {
            case light, medium, heavy, rigid, soft
        }
        
        enum NotificationType {
            case success, warning, error
        }
        
        static func play(_ style: FeedbackStyle) {
            #if os(iOS)
            let uiStyle: UIImpactFeedbackGenerator.FeedbackStyle
            switch style {
            case .light: uiStyle = .light
            case .medium: uiStyle = .medium
            case .heavy: uiStyle = .heavy
            case .rigid: uiStyle = .rigid
            case .soft: uiStyle = .soft
            }
            let generator = UIImpactFeedbackGenerator(style: uiStyle)
            generator.prepare()
            generator.impactOccurred()
            #elseif os(macOS)
            let feedbackPattern: NSHapticFeedbackManager.FeedbackPattern
            switch style {
            case .light, .soft: feedbackPattern = .alignment
            default: feedbackPattern = .levelChange
            }
            NSHapticFeedbackManager.defaultPerformer.perform(feedbackPattern, performanceTime: .default)
            #endif
        }
        
        static func notification(_ type: NotificationType) {
            #if os(iOS)
            let uiType: UINotificationFeedbackGenerator.FeedbackType
            switch type {
            case .success: uiType = .success
            case .warning: uiType = .warning
            case .error: uiType = .error
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
            #elseif os(macOS)
            // Subtle alignment click for selection
            NSHapticFeedbackManager.defaultPerformer.perform(.alignment, performanceTime: .default)
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
    /// Applies a standard ultra-thin material background with a corner radius.
    /// Defaults to `Theme.Layout.cornerRadius`.
    func ultraThinGlass(cornerRadius: CGFloat = Theme.Layout.cornerRadius) -> some View {
        self.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius))
             .overlay(
                 RoundedRectangle(cornerRadius: cornerRadius)
                     .strokeBorder(Theme.Colors.separator.opacity(0.3), lineWidth: 0.5)
                     .blendMode(.overlay)
             )
    }
    
    /// Applies a "Premium" pill shape with glass effect, commonly used for floating toolbars.
    func glassPill() -> some View {
        self.background(.regularMaterial, in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(.white.opacity(0.2), lineWidth: 0.5)
                    .blendMode(.screen)
            )
            .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 5)
    }
    
    /// Applies the standard "Premium" shadow style
    func premiumShadow() -> some View {
        self.shadow(color: Color.black.opacity(0.08), radius: Theme.Layout.shadowRadius, x: 0, y: Theme.Layout.shadowY)
    }
}

