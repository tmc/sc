/**
 * Performance Optimizer for Large Statechart Visualization
 * 
 * Implements advanced rendering optimizations including:
 * - Level-of-detail (LOD) rendering
 * - Viewport culling
 * - Selective rendering
 * - Virtualization for large datasets
 * - Performance monitoring and adaptive optimization
 */

class PerformanceOptimizer {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.renderingConfig = {
            maxVisibleNodes: 100,
            minZoomForLabels: 0.5,
            minZoomForDetails: 0.8,
            cullPadding: 50,
            useVirtualization: true,
            adaptiveQuality: true,
            animationThreshold: 200, // nodes
            transitionThreshold: 300  // transitions
        };
        
        this.performance = {
            frameTime: 0,
            renderTime: 0,
            nodesRendered: 0,
            transitionsRendered: 0,
            fps: 60,
            targetFps: 60
        };
        
        this.levelOfDetail = {
            current: 'high',
            levels: {
                low: { nodeDetail: 0.3, transitionDetail: 0.2, labelThreshold: 1.0 },
                medium: { nodeDetail: 0.6, transitionDetail: 0.5, labelThreshold: 0.7 },
                high: { nodeDetail: 1.0, transitionDetail: 1.0, labelThreshold: 0.3 }
            }
        };
        
        this.viewportBounds = { x: 0, y: 0, width: 0, height: 0 };
        this.visibleNodes = new Set();
        this.visibleTransitions = new Set();
        
        this.performanceMonitor = new PerformanceMonitor(this);
        this.setupPerformanceMonitoring();
    }

    setupPerformanceMonitoring() {
        // Monitor frame rate and adapt quality
        this.performanceMonitor.onFpsChange((fps) => {
            if (fps < this.performance.targetFps * 0.8) {
                this.degradeQuality();
            } else if (fps > this.performance.targetFps * 0.95) {
                this.improveQuality();
            }
        });

        // Monitor render time
        this.performanceMonitor.onRenderTimeChange((renderTime) => {
            this.performance.renderTime = renderTime;
            if (renderTime > 16.67) { // 60fps = 16.67ms per frame
                this.optimizeRendering();
            }
        });
    }

    optimizeStatechartRendering(statechart, svg, transform) {
        const startTime = performance.now();
        
        // Update viewport bounds
        this.updateViewportBounds(svg, transform);
        
        // Determine appropriate level of detail
        const lodLevel = this.calculateLevelOfDetail(statechart, transform);
        
        // Perform viewport culling
        const visibleElements = this.cullElements(statechart, transform);
        
        // Apply optimized rendering strategy
        const renderStrategy = this.selectRenderStrategy(visibleElements);
        
        // Execute optimized rendering
        const renderResult = this.executeOptimizedRender(
            visibleElements, 
            svg, 
            transform, 
            lodLevel, 
            renderStrategy
        );
        
        // Update performance metrics
        const renderTime = performance.now() - startTime;
        this.updatePerformanceMetrics(renderTime, renderResult);
        
        return renderResult;
    }

    updateViewportBounds(svg, transform) {
        const svgRect = svg.node().getBoundingClientRect();
        const scale = transform ? transform.k : 1;
        const offsetX = transform ? transform.x : 0;
        const offsetY = transform ? transform.y : 0;

        this.viewportBounds = {
            x: (-offsetX / scale) - this.renderingConfig.cullPadding,
            y: (-offsetY / scale) - this.renderingConfig.cullPadding,
            width: (svgRect.width / scale) + (this.renderingConfig.cullPadding * 2),
            height: (svgRect.height / scale) + (this.renderingConfig.cullPadding * 2)
        };
    }

    calculateLevelOfDetail(statechart, transform) {
        const scale = transform ? transform.k : 1;
        const nodeCount = this.countNodes(statechart);
        
        // Determine LOD based on zoom level and node count
        if (scale < 0.3 || nodeCount > 500) {
            return 'low';
        } else if (scale < 0.7 || nodeCount > 200) {
            return 'medium';
        } else {
            return 'high';
        }
    }

    cullElements(statechart, transform) {
        const visibleNodes = [];
        const visibleTransitions = [];
        
        // Traverse the statechart hierarchy and check visibility
        this.traverseAndCull(statechart.root_state, visibleNodes, transform);
        
        // Cull transitions based on visible nodes
        if (statechart.transitions) {
            statechart.transitions.forEach(transition => {
                if (this.isTransitionVisible(transition, visibleNodes)) {
                    visibleTransitions.push(transition);
                }
            });
        }
        
        return {
            nodes: visibleNodes,
            transitions: visibleTransitions,
            totalNodes: this.countNodes(statechart),
            totalTransitions: statechart.transitions ? statechart.transitions.length : 0
        };
    }

    traverseAndCull(state, visibleNodes, transform, depth = 0, coordinates = null) {
        if (!state) return;
        
        // Calculate or use provided coordinates
        const nodeCoords = coordinates || this.calculateNodeCoordinates(state, depth);
        
        // Check if node is within viewport
        if (this.isNodeInViewport(nodeCoords)) {
            visibleNodes.push({
                state: state,
                coordinates: nodeCoords,
                depth: depth
            });
        }
        
        // Recursively check children (with depth limiting for performance)
        if (state.children && depth < 5) { // Limit depth to prevent excessive recursion
            state.children.forEach((child, index) => {
                const childCoords = this.calculateChildCoordinates(nodeCoords, index, state.children.length);
                this.traverseAndCull(child, visibleNodes, transform, depth + 1, childCoords);
            });
        }
    }

    isNodeInViewport(nodeCoords) {
        return nodeCoords.x + nodeCoords.width >= this.viewportBounds.x &&
               nodeCoords.x <= this.viewportBounds.x + this.viewportBounds.width &&
               nodeCoords.y + nodeCoords.height >= this.viewportBounds.y &&
               nodeCoords.y <= this.viewportBounds.y + this.viewportBounds.height;
    }

    isTransitionVisible(transition, visibleNodes) {
        const fromVisible = visibleNodes.some(node => 
            transition.from && transition.from.includes(node.state.label)
        );
        const toVisible = visibleNodes.some(node => 
            transition.to && transition.to.includes(node.state.label)
        );
        return fromVisible && toVisible;
    }

    selectRenderStrategy(visibleElements) {
        const nodeCount = visibleElements.nodes.length;
        const transitionCount = visibleElements.transitions.length;
        
        if (nodeCount > this.renderingConfig.animationThreshold) {
            return {
                type: 'static',
                animations: false,
                simplifiedNodes: true,
                batchRender: true
            };
        } else if (nodeCount > 50) {
            return {
                type: 'optimized',
                animations: true,
                simplifiedNodes: false,
                batchRender: true
            };
        } else {
            return {
                type: 'full',
                animations: true,
                simplifiedNodes: false,
                batchRender: false
            };
        }
    }

    executeOptimizedRender(visibleElements, svg, transform, lodLevel, renderStrategy) {
        const lodConfig = this.levelOfDetail.levels[lodLevel];
        
        // Clear previous renders
        this.clearOptimizedGroups(svg);
        
        // Create optimized rendering groups
        const nodeGroup = svg.append('g').attr('class', 'optimized-nodes');
        const transitionGroup = svg.append('g').attr('class', 'optimized-transitions');
        
        // Render transitions first (behind nodes)
        this.renderOptimizedTransitions(
            transitionGroup, 
            visibleElements.transitions, 
            lodConfig, 
            renderStrategy
        );
        
        // Render nodes
        this.renderOptimizedNodes(
            nodeGroup, 
            visibleElements.nodes, 
            lodConfig, 
            renderStrategy
        );
        
        return {
            nodesRendered: visibleElements.nodes.length,
            transitionsRendered: visibleElements.transitions.length,
            lodLevel: lodLevel,
            strategy: renderStrategy.type
        };
    }

    renderOptimizedNodes(container, visibleNodes, lodConfig, renderStrategy) {
        if (renderStrategy.batchRender) {
            // Batch render for performance
            this.batchRenderNodes(container, visibleNodes, lodConfig, renderStrategy);
        } else {
            // Individual render for quality
            this.individualRenderNodes(container, visibleNodes, lodConfig, renderStrategy);
        }
    }

    batchRenderNodes(container, visibleNodes, lodConfig, renderStrategy) {
        // Group nodes by type for batch processing
        const nodesByType = this.groupNodesByType(visibleNodes);
        
        Object.entries(nodesByType).forEach(([type, nodes]) => {
            const typeGroup = container.append('g').attr('class', `nodes-${type}`);
            
            // Create all nodes of this type at once
            const nodeElements = typeGroup.selectAll('.optimized-node')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', 'optimized-node')
                .attr('transform', d => `translate(${d.coordinates.x}, ${d.coordinates.y})`);

            // Add simplified geometry for performance
            nodeElements.append('rect')
                .attr('class', 'node-rect')
                .attr('x', -20)
                .attr('y', -10)
                .attr('width', 40)
                .attr('height', 20)
                .attr('rx', 3)
                .attr('fill', d => this.getNodeColor(d.state, type))
                .attr('stroke', '#2c3e50')
                .attr('stroke-width', 1);

            // Add labels only if zoom level is sufficient
            if (lodConfig.labelThreshold <= 1.0) {
                nodeElements.append('text')
                    .attr('class', 'node-label')
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'central')
                    .attr('font-size', '10px')
                    .attr('fill', 'white')
                    .text(d => this.truncateLabel(d.state.label, 8))
                    .style('pointer-events', 'none');
            }
        });
    }

    individualRenderNodes(container, visibleNodes, lodConfig, renderStrategy) {
        visibleNodes.forEach(nodeData => {
            const nodeGroup = container.append('g')
                .attr('class', 'optimized-node')
                .attr('transform', `translate(${nodeData.coordinates.x}, ${nodeData.coordinates.y})`);

            // Full quality rendering
            this.renderFullQualityNode(nodeGroup, nodeData, lodConfig);
        });
    }

    renderOptimizedTransitions(container, visibleTransitions, lodConfig, renderStrategy) {
        const transitionElements = container.selectAll('.optimized-transition')
            .data(visibleTransitions)
            .enter()
            .append('g')
            .attr('class', 'optimized-transition');

        // Simplified transition rendering for performance
        transitionElements.append('path')
            .attr('class', 'transition-path')
            .attr('d', d => this.calculateOptimizedTransitionPath(d))
            .attr('stroke', '#3498db')
            .attr('stroke-width', lodConfig.transitionDetail * 2)
            .attr('fill', 'none')
            .attr('marker-end', lodConfig.transitionDetail > 0.5 ? 'url(#arrowhead)' : null);

        // Add labels only for high detail level
        if (lodConfig.transitionDetail > 0.7) {
            transitionElements.append('text')
                .attr('class', 'transition-label')
                .attr('text-anchor', 'middle')
                .attr('font-size', '8px')
                .attr('fill', '#2c3e50')
                .text(d => d.event || d.label)
                .style('pointer-events', 'none');
        }
    }

    // Performance monitoring and adaptation methods

    degradeQuality() {
        const currentLevel = this.levelOfDetail.current;
        if (currentLevel === 'high') {
            this.levelOfDetail.current = 'medium';
        } else if (currentLevel === 'medium') {
            this.levelOfDetail.current = 'low';
        }
        
        // Reduce rendering limits
        this.renderingConfig.maxVisibleNodes = Math.max(50, this.renderingConfig.maxVisibleNodes * 0.8);
        this.renderingConfig.animationThreshold = Math.max(100, this.renderingConfig.animationThreshold * 0.8);
        
        console.log(`Performance degraded to ${this.levelOfDetail.current} quality`);
    }

    improveQuality() {
        const currentLevel = this.levelOfDetail.current;
        if (currentLevel === 'low') {
            this.levelOfDetail.current = 'medium';
        } else if (currentLevel === 'medium') {
            this.levelOfDetail.current = 'high';
        }
        
        // Increase rendering limits
        this.renderingConfig.maxVisibleNodes = Math.min(200, this.renderingConfig.maxVisibleNodes * 1.2);
        this.renderingConfig.animationThreshold = Math.min(300, this.renderingConfig.animationThreshold * 1.2);
        
        console.log(`Performance improved to ${this.levelOfDetail.current} quality`);
    }

    optimizeRendering() {
        // Enable more aggressive optimizations
        this.renderingConfig.useVirtualization = true;
        this.renderingConfig.cullPadding = Math.max(25, this.renderingConfig.cullPadding * 0.9);
        
        // Simplify animations
        if (this.renderingConfig.animationThreshold > 150) {
            this.renderingConfig.animationThreshold *= 0.9;
        }
    }

    // Utility methods

    countNodes(statechart) {
        if (!statechart || !statechart.root_state) return 0;
        return this.countNodesRecursive(statechart.root_state);
    }

    countNodesRecursive(state) {
        if (!state) return 0;
        let count = 1;
        if (state.children) {
            count += state.children.reduce((sum, child) => sum + this.countNodesRecursive(child), 0);
        }
        return count;
    }

    calculateNodeCoordinates(state, depth) {
        // Simplified coordinate calculation - would use actual layout in practice
        return {
            x: depth * 100,
            y: Math.random() * 500, // Would use proper layout algorithm
            width: 80,
            height: 30
        };
    }

    calculateChildCoordinates(parentCoords, index, totalChildren) {
        const spacing = 120;
        return {
            x: parentCoords.x + 150,
            y: parentCoords.y + (index - totalChildren/2) * spacing,
            width: 80,
            height: 30
        };
    }

    groupNodesByType(visibleNodes) {
        const groups = {};
        visibleNodes.forEach(nodeData => {
            const type = this.getNodeType(nodeData.state);
            if (!groups[type]) groups[type] = [];
            groups[type].push(nodeData);
        });
        return groups;
    }

    getNodeType(state) {
        if (state.is_final) return 'final';
        if (state.is_initial) return 'initial';
        if (state.type === 3) return 'parallel';
        if (state.type === 2) return 'compound';
        return 'atomic';
    }

    getNodeColor(state, type) {
        const colors = {
            final: '#e74c3c',
            initial: '#2ecc71',
            parallel: '#f39c12',
            compound: '#9b59b6',
            atomic: '#3498db'
        };
        return colors[type] || colors.atomic;
    }

    truncateLabel(label, maxLength) {
        if (label.length <= maxLength) return label;
        return label.substring(0, maxLength - 3) + '...';
    }

    calculateOptimizedTransitionPath(transition) {
        // Simplified path calculation for performance
        return 'M 0 0 L 100 0'; // Would calculate actual path
    }

    clearOptimizedGroups(svg) {
        svg.selectAll('.optimized-nodes').remove();
        svg.selectAll('.optimized-transitions').remove();
    }

    renderFullQualityNode(nodeGroup, nodeData, lodConfig) {
        // Full quality node rendering with all details
        const state = nodeData.state;
        
        nodeGroup.append('rect')
            .attr('class', 'node-rect')
            .attr('x', -40)
            .attr('y', -15)
            .attr('width', 80)
            .attr('height', 30)
            .attr('rx', 5)
            .attr('fill', this.getNodeColor(state, this.getNodeType(state)))
            .attr('stroke', '#2c3e50')
            .attr('stroke-width', 1);

        nodeGroup.append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('font-size', '12px')
            .attr('fill', 'white')
            .text(state.label);
    }

    updatePerformanceMetrics(renderTime, renderResult) {
        this.performance.renderTime = renderTime;
        this.performance.nodesRendered = renderResult.nodesRendered;
        this.performance.transitionsRendered = renderResult.transitionsRendered;
        
        this.performanceMonitor.recordFrame(renderTime);
    }

    getPerformanceReport() {
        return {
            ...this.performance,
            levelOfDetail: this.levelOfDetail.current,
            renderingConfig: this.renderingConfig,
            optimizationsActive: this.getActiveOptimizations()
        };
    }

    getActiveOptimizations() {
        return {
            culling: true,
            levelOfDetail: this.levelOfDetail.current !== 'high',
            virtualization: this.renderingConfig.useVirtualization,
            adaptiveQuality: this.renderingConfig.adaptiveQuality
        };
    }
}

/**
 * Performance Monitor
 * Tracks rendering performance and triggers optimizations
 */
class PerformanceMonitor {
    constructor(optimizer) {
        this.optimizer = optimizer;
        this.frameTimes = [];
        this.maxFrameHistory = 60; // 1 second at 60fps
        this.fpsCallbacks = [];
        this.renderTimeCallbacks = [];
        this.lastFrameTime = performance.now();
    }

    recordFrame(renderTime) {
        const now = performance.now();
        const frameTime = now - this.lastFrameTime;
        
        this.frameTimes.push(frameTime);
        if (this.frameTimes.length > this.maxFrameHistory) {
            this.frameTimes.shift();
        }
        
        // Calculate average FPS
        const avgFrameTime = this.frameTimes.reduce((sum, time) => sum + time, 0) / this.frameTimes.length;
        const fps = 1000 / avgFrameTime;
        
        // Trigger callbacks
        this.fpsCallbacks.forEach(callback => callback(fps));
        this.renderTimeCallbacks.forEach(callback => callback(renderTime));
        
        this.lastFrameTime = now;
    }

    onFpsChange(callback) {
        this.fpsCallbacks.push(callback);
    }

    onRenderTimeChange(callback) {
        this.renderTimeCallbacks.push(callback);
    }

    getCurrentFps() {
        if (this.frameTimes.length === 0) return 60;
        const avgFrameTime = this.frameTimes.reduce((sum, time) => sum + time, 0) / this.frameTimes.length;
        return 1000 / avgFrameTime;
    }

    reset() {
        this.frameTimes = [];
        this.lastFrameTime = performance.now();
    }
}

// Export for use in visualizer
if (typeof window !== 'undefined') {
    window.PerformanceOptimizer = PerformanceOptimizer;
    window.PerformanceMonitor = PerformanceMonitor;
}

export { PerformanceOptimizer, PerformanceMonitor };