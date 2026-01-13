
import SwiftUI

#if os(macOS)
import AppKit

struct ScrollEventView: NSViewRepresentable {
    @Binding var offset: CGSize
    @Binding var scale: CGFloat
    @Binding var nodes: [FlowNode] // Shared API

    
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
        var monitor: Any?
        
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            setupMonitor()
        }
        
        private func setupMonitor() {
            if monitor != nil { NSEvent.removeMonitor(monitor!) }
            
            // Capture Scroll and Magnify globally in the window, but filter for our bounds
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.scrollWheel, .magnify]) { [weak self] event in
                guard let self = self, self.window != nil else { return event }
                
                let locationInWindow = event.locationInWindow
                let localPoint = self.convert(locationInWindow, from: nil)
                
                if self.bounds.contains(localPoint) {
                    if event.type == .scrollWheel {
                        self.handleScroll(event)
                        return nil 
                    } else if event.type == .magnify {
                         self.handleZoom(delta: event.magnification, at: locationInWindow)
                         return nil
                    }
                }
                return event
            }
        }
        
        deinit {
            if let monitor = monitor { NSEvent.removeMonitor(monitor) }
        }
        
        override func hitTest(_ point: NSPoint) -> NSView? {
            return nil
        }
        
        private func handleScroll(_ event: NSEvent) {
            if event.modifierFlags.contains(.command) {
                // Command + Scroll = Zoom
                // Logarithmic stepping for natural feel
                // Standard mouse wheel delta is usually around 0.1 to 10.0
                // Trackpad delta is smaller.
                let sensitivity: CGFloat = 0.01 
                let delta = event.scrollingDeltaY * sensitivity
                handleZoom(delta: delta, at: event.locationInWindow)
            } else {
                guard let offsetBinding = offsetBinding else { return }
                
                var multiplier: CGFloat = 1.0
                if event.modifierFlags.contains(.option) {
                    // Option + Scroll = Precision Panning (Smoother/Slower)
                    multiplier = 0.2
                }
                
                // Trackpad momentum is handled automatically by the system sending
                // events with phase == .momentum or .ended.
                // We just apply the deltas.
                
                let currentOffset = offsetBinding.wrappedValue
                let newOffset = CGSize(
                    width: currentOffset.width + (event.scrollingDeltaX * multiplier),
                    height: currentOffset.height + (event.scrollingDeltaY * multiplier)
                )
                
                // Immediate update
                // For 120Hz ProMotion, this needs to be fast.
                offsetBinding.wrappedValue = newOffset
            }
        }
        
        private func handleZoom(delta: CGFloat, at locationInWindow: CGPoint) {
            guard let scaleBinding = scaleBinding,
                  let offsetBinding = offsetBinding else { return }
            
            let currentScale = scaleBinding.wrappedValue
            let currentOffset = offsetBinding.wrappedValue
            
            // Logarithmic / Exponential Zoom
            // scale = oldScale * (1 + delta) is linear approximation of exp
            // For true momentum feeling, we trust the delta curve.
            
            let newScaleRaw = currentScale * (1 + delta)
            let newScale = min(max(newScaleRaw, 0.1), 5.0) // Clamp 0.1x to 5.0x
            
            if newScale == currentScale { return }
            
            let ratio = newScale / currentScale
            
            // Anchor at Cursor
            let localPoint = self.convert(locationInWindow, from: nil)
            
            // Convert local mouse point to "Canvas Space" relative to center (0,0) of view
            // The canvas is centered at view center.
            let viewCenter = CGPoint(x: self.bounds.midX, y: self.bounds.midY)
            
            // P_screen = P_world * scale + offset + center
            // P_world = (P_screen - center - offset) / scale
            
            // We want P_world under cursor to remain constant.
            // P_screen_new = P_world * newScale + newOffset + center
            // P_screen_old = P_screen_new (cursor didn't move)
            
            // (P_s - c - o_old) / s_old = (P_s - c - o_new) / s_new
            // Let V = P_s - c (vector from center to cursor)
            // (V - o_old) / s_old = (V - o_new) / s_new
            // (V - o_old) * (s_new/s_old) = V - o_new
            // o_new = V - (V - o_old) * ratio
            
            let v = CGSize(width: localPoint.x - viewCenter.x, height: localPoint.y - viewCenter.y)
            let newOffset = CGSize(
                width: v.width - (v.width - currentOffset.width) * ratio,
                height: v.height - (v.height - currentOffset.height) * ratio
            )
            
            scaleBinding.wrappedValue = newScale
            offsetBinding.wrappedValue = newOffset
        }
    }
}
#else
import UIKit

struct ScrollEventView: UIViewRepresentable {
    @Binding var offset: CGSize
    @Binding var scale: CGFloat
    @Binding var nodes: [FlowNode] // Need nodes for hit testing
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    func makeUIView(context: Context) -> UIScrollView {
        let scrollView = PassThroughScrollView()
        scrollView.parent = self // Link for hit testing
        scrollView.delegate = context.coordinator
        scrollView.minimumZoomScale = 0.1
        scrollView.maximumZoomScale = 5.0
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.bounces = true
        scrollView.decelerationRate = .normal
        
        // Massive content size
        let keyspace: CGFloat = 200_000
        scrollView.contentSize = CGSize(width: keyspace, height: keyspace)
        
        // Settings for pass-through
        scrollView.delaysContentTouches = false
        scrollView.canCancelContentTouches = true
        
        let zoomView = UIView()
        zoomView.tag = 999
        zoomView.frame = CGRect(origin: .zero, size: scrollView.contentSize)
        zoomView.isUserInteractionEnabled = false 
        scrollView.addSubview(zoomView)
        
        return scrollView
    }
    
    func updateUIView(_ uiView: UIScrollView, context: Context) {
        if let scrollView = uiView as? PassThroughScrollView {
            scrollView.parent = self
            // Note: Updating 'nodes' binding doesn't need to do anything to the view
            // The hitTest uses the capture 'self' or updated via 'parent' ref
        }
        
        // Sync Binding -> ScrollView
        if uiView.isDragging || uiView.isDecelerating || uiView.isZooming { return }
        
        let keyspace: CGFloat = 200_000
        let center = keyspace / 2
        let viewSize = uiView.bounds.size
        if viewSize.width == 0 { return }
        
        if abs(uiView.zoomScale - scale) > 0.001 {
             uiView.setZoomScale(scale, animated: false)
        }
        
        let targetX = center - offset.width - (viewSize.width / 2)
        let targetY = center - offset.height - (viewSize.height / 2)
        
        let currentPos = uiView.contentOffset
        if abs(currentPos.x - targetX) > 1.0 || abs(currentPos.y - targetY) > 1.0 {
            uiView.setContentOffset(CGPoint(x: targetX, y: targetY), animated: false)
        }
    }
    
    // Custom ScrollView for Hit Testing
    class PassThroughScrollView: UIScrollView {
        var parent: ScrollEventView?
        
        override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
            // Check if we hit a node
            // Point is in ScrollView bounds (screen coordinates mostly, since we are fullscreen)
            // We need to map to Canvas Space
            guard let parent = parent else { return super.hitTest(point, with: event) }
            
            // Transform point to Canvas Space
            // Canvas Space (P_world)
            // Screen Point (P_screen) = point
            // center = size/2
            // P_world = (P_screen - center - offset) / scale
            
            let center = CGPoint(x: bounds.width / 2, y: bounds.height / 2)
            let offset = parent.offset
            let scale = parent.scale
            
            let worldPoint = CGPoint(
                x: (point.x - center.x - offset.width) / scale,
                y: (point.y - center.y - offset.height) / scale
            )
            
            // Check nodes (Iterate in reverse Z-order - top first)
            // Since nodes array is usually drawn in order, last == top?
            // FlowView sorts them by hierarchy. We should check roughly.
            // A simple "contains" check is sufficient for pass-through.
            
            // Optimization: Only check if nodes count is reasonable
            // For 1000 nodes, this linear scan in hitTest is fast enough (on CPU, native code)
            
            for node in parent.nodes.reversed() {
                let nodeRect = CGRect(origin: node.position, size: node.size)
                // Add padding for touch targets?
                if nodeRect.contains(worldPoint) {
                    return nil // Passthrough to SwiftUI views below
                }
            }
            
            return super.hitTest(point, with: event)
        }
    }
    
    class Coordinator: NSObject, UIScrollViewDelegate {
        var parent: ScrollEventView
        
        init(_ parent: ScrollEventView) {
            self.parent = parent
        }
        
        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            return scrollView.viewWithTag(999)
        }
        
        func scrollViewDidScroll(_ scrollView: UIScrollView) {
            updateBindings(scrollView)
        }
        
        func scrollViewDidZoom(_ scrollView: UIScrollView) {
             updateBindings(scrollView)
        }
        
        private func updateBindings(_ scrollView: UIScrollView) {
            let keyspace: CGFloat = 200_000
            let center = keyspace / 2
            let viewSize = scrollView.bounds.size
            if viewSize.width == 0 { return }
            
            // Reverse logic:
            // ContentOffset = KeyspaceCenter - offset - ViewSize/2
            // offset = KeyspaceCenter - ContentOffset - ViewSize/2
            
            let ox = center - scrollView.contentOffset.x - (viewSize.width / 2)
            let oy = center - scrollView.contentOffset.y - (viewSize.height / 2)
            
            // Publish
            // We need to route this back to binding.
            // CAUTION: This triggers State Update -> updateUIView.
            // We added guards in updateUIView to active interaction.
            
            DispatchQueue.main.async {
                self.parent.offset = CGSize(width: ox, height: oy)
                self.parent.scale = scrollView.zoomScale
            }
        }
    }
}
#endif
