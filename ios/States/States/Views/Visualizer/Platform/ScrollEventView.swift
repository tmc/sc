import SwiftUI

#if os(macOS)
import AppKit

struct ScrollEventView: NSViewRepresentable {
    @Binding var offset: CGSize
    @Binding var scale: CGFloat
    
    func makeNSView(context: Context) -> ScrollEventHandlingView {
        let view = ScrollEventHandlingView()
        view.offsetBinding = $offset
        view.scaleBinding = $scale
        return view
    }
    
    func updateNSView(_ nsView: ScrollEventHandlingView, context: Context) {
        nsView.offsetBinding = $offset
        nsView.scaleBinding = $scale
    }
    
    class ScrollEventHandlingView: NSView {
        var offsetBinding: Binding<CGSize>?
        var scaleBinding: Binding<CGFloat>?
        
        override var acceptsFirstResponder: Bool { true }
        
        override func magnify(with event: NSEvent) {
            handleZoom(delta: event.magnification, at: event.locationInWindow)
        }
        
        override func scrollWheel(with event: NSEvent) {
            if event.modifierFlags.contains(.command) {
                // Zoom with Scroll (Mouse usually, or Trackpad with Cmd)
                // Sensitivity factor
                let factor: CGFloat = 0.01
                handleZoom(delta: event.scrollingDeltaY * factor, at: event.locationInWindow)
            } else {
                // Pan
                guard let offsetBinding = offsetBinding else { return }
                let currentOffset = offsetBinding.wrappedValue
                
                let newOffset = CGSize(
                    width: currentOffset.width + event.scrollingDeltaX,
                    height: currentOffset.height + event.scrollingDeltaY
                )
                
                offsetBinding.wrappedValue = newOffset
            }
        }
        
        private func handleZoom(delta: CGFloat, at locationInWindow: CGPoint) {
            guard let scaleBinding = scaleBinding,
                  let offsetBinding = offsetBinding else { return }
            
            let currentScale = scaleBinding.wrappedValue
            let currentOffset = offsetBinding.wrappedValue
            
            // Calculate new scale
            // For pinch (magnify), delta is the change factor (e.g. 0.01).
            // newScale = currentScale * (1 + delta) is typical for magnification events? 
            // verifying: event.magnification is 0 if no change, +1 if doubled.
            let newScaleRaw = currentScale + (currentScale * delta)
            let newScale = min(max(newScaleRaw, 0.1), 5.0)
            
            // Calculate Ratio
            let ratio = newScale / currentScale
            
            // Convert window location to local view coordinates (which centers 0,0 typically? No, NSView coords).
            // This view is an overlay filling the area. Midpoint of this view corresponds to the "center" used in drawing.
            let localPoint = self.convert(locationInWindow, from: nil)
            let viewCenter = CGPoint(x: self.bounds.midX, y: self.bounds.midY)
            
            // Vector from center to cursor
            let v = CGSize(width: localPoint.x - viewCenter.x, height: localPoint.y - viewCenter.y)
            
            // Math: newOffset = v * (1 - ratio) + oldOffset * ratio
            let newOffset = CGSize(
                width: v.width * (1 - ratio) + currentOffset.width * ratio,
                height: v.height * (1 - ratio) + currentOffset.height * ratio
            )
            
            scaleBinding.wrappedValue = newScale
            // Only update offset if we actually scaled (bounds check)
            if newScale != currentScale {
                offsetBinding.wrappedValue = newOffset
            }
        }
    }
}
#else
import UIKit

struct ScrollEventView: UIViewRepresentable {
    @Binding var offset: CGSize
    @Binding var scale: CGFloat
    
    func makeUIView(context: Context) -> IOSScrollEventHandlingView {
        let view = IOSScrollEventHandlingView()
        view.offsetBinding = $offset
        view.scaleBinding = $scale
        return view
    }
    
    func updateUIView(_ uiView: IOSScrollEventHandlingView, context: Context) {
        uiView.offsetBinding = $offset
        uiView.scaleBinding = $scale
    }
    
    class IOSScrollEventHandlingView: UIView, UIGestureRecognizerDelegate {
        var offsetBinding: Binding<CGSize>?
        var scaleBinding: Binding<CGFloat>?
        
        private var initialPinchScale: CGFloat = 1.0
        private var lastPanLocation: CGPoint = .zero
        
        override init(frame: CGRect) {
            super.init(frame: frame)
            setupGestures()
        }
        
        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }
        
        private func setupGestures() {
            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            pinch.delegate = self
            self.addGestureRecognizer(pinch)
            
            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            // Pan requires 2 touches to differentiate from drag-to-connect or node drag
            // But if those are on nodes, maybe 2 fingers is good for canvas pan?
            // "Trackpad style" usually implies 2 finger pan on iPad, but 1 finger pan for canvas is also common mobile UX.
            // FlowView handles single finger drag. So we need 2 fingers here to avoid conflict.
            pan.minimumNumberOfTouches = 2
            pan.maximumNumberOfTouches = 2 
            pan.delegate = self
            self.addGestureRecognizer(pan)
        }
        
        // Allow simultaneous gestures (Pan + Pinch)
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            return true
        }
        
        @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let scaleBinding = scaleBinding,
                  let offsetBinding = offsetBinding else { return }
            
            switch gesture.state {
            case .began:
                initialPinchScale = scaleBinding.wrappedValue
            case .changed:
                let currentScale = scaleBinding.wrappedValue
                let scaleDelta = gesture.scale
                
                // Calculate new scale
                let newScaleRaw = initialPinchScale * scaleDelta
                let newScale = min(max(newScaleRaw, 0.1), 5.0)
                
                // Zoom-to-point logic
                let ratio = newScale / currentScale
                
                // Pinch center in view coordinates
                let pinchCenter = gesture.location(in: self)
                let viewCenter = CGPoint(x: self.bounds.midX, y: self.bounds.midY)
                
                let v = CGSize(width: pinchCenter.x - viewCenter.x, height: pinchCenter.y - viewCenter.y)
                
                let currentOffset = offsetBinding.wrappedValue
                let newOffset = CGSize(
                    width: v.width * (1 - ratio) + currentOffset.width * ratio,
                    height: v.height * (1 - ratio) + currentOffset.height * ratio
                )
                
                scaleBinding.wrappedValue = newScale
                if newScale != currentScale {
                    offsetBinding.wrappedValue = newOffset
                }
                
                // Reset scale to 1 to accumulate changes incrementally? 
                // No, sticking with initialPinchScale * gesture.scale is standard for UIPinch
                // But typically for granular updates we might reset:
                // gesture.scale = 1.0
                // initialPinchScale = newScale
                // Let's try incremental to match standard patterns better if "initial" approach drifts.
                // Resetting is safer for accumulation:
                gesture.scale = 1.0
                initialPinchScale = newScale
                
            default: break
            }
        }
        
        @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let offsetBinding = offsetBinding else { return }
            
            switch gesture.state {
            case .began:
                lastPanLocation = gesture.translation(in: self)
            case .changed:
                let translation = gesture.translation(in: self)
                let deltaX = translation.x - lastPanLocation.x
                let deltaY = translation.y - lastPanLocation.y
                
                let currentOffset = offsetBinding.wrappedValue
                offsetBinding.wrappedValue = CGSize(
                    width: currentOffset.width + deltaX,
                    height: currentOffset.height + deltaY
                )
                
                lastPanLocation = translation
            default: break
            }
        }
    }
}
#endif
