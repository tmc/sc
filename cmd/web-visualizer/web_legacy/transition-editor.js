/**
 * TransitionEditor - Handles transition drawing and editing
 *
 * Features:
 * - Click-to-connect transition creation
 * - Bezier curve rendering with control points
 * - Self-transition loops
 * - Transition label positioning
 * - Control point dragging for curve adjustment
 * - Arrow marker positioning
 */

class TransitionEditor extends EventTarget {
    constructor(options = {}) {
        super();

        // Dependencies
        this.svg = options.svg; // D3 selection of SVG element
        this.stateManager = options.stateManager;
        this.orchestrator = options.orchestrator;

        // Drawing state
        this.isDrawing = false;
        this.sourceState = null;
        this.tempLine = null;
        this.mousePosition = { x: 0, y: 0 };

        // Visual settings
        this.curveOffset = 50; // Control point offset for bezier curves
        this.selfLoopRadius = 30;
        this.arrowSize = 8;
        this.labelOffset = 10;

        // Groups for rendering
        this.transitionsGroup = null;
        this.tempGroup = null;

        // Bind methods
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleClick = this.handleClick.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);

        this.initialize();
    }

    initialize() {
        if (!this.svg) return;

        // Create groups for transitions
        this.transitionsGroup = this.svg.select('.transitions-group');
        if (this.transitionsGroup.empty()) {
            this.transitionsGroup = this.svg.insert('g', '.states-group')
                .attr('class', 'transitions-group');
        }

        // Create group for temporary drawing elements
        this.tempGroup = this.svg.select('.temp-group');
        if (this.tempGroup.empty()) {
            this.tempGroup = this.svg.append('g')
                .attr('class', 'temp-group');
        }

        // Create arrow marker definition
        this.createArrowMarker();

        // Set up event listeners
        this.setupEventListeners();
    }

    createArrowMarker() {
        let defs = this.svg.select('defs');
        if (defs.empty()) {
            defs = this.svg.append('defs');
        }

        // Remove existing marker if present
        defs.select('#arrow-marker').remove();
        defs.select('#arrow-marker-highlight').remove();

        // Normal arrow
        defs.append('marker')
            .attr('id', 'arrow-marker')
            .attr('viewBox', '0 0 10 10')
            .attr('refX', 9)
            .attr('refY', 5)
            .attr('markerWidth', this.arrowSize)
            .attr('markerHeight', this.arrowSize)
            .attr('orient', 'auto-start-reverse')
            .append('path')
            .attr('d', 'M 0 0 L 10 5 L 0 10 z')
            .attr('fill', '#666');

        // Highlighted arrow
        defs.append('marker')
            .attr('id', 'arrow-marker-highlight')
            .attr('viewBox', '0 0 10 10')
            .attr('refX', 9)
            .attr('refY', 5)
            .attr('markerWidth', this.arrowSize)
            .attr('markerHeight', this.arrowSize)
            .attr('orient', 'auto-start-reverse')
            .append('path')
            .attr('d', 'M 0 0 L 10 5 L 0 10 z')
            .attr('fill', '#2196F3');

        // Enabled transition arrow (for simulation)
        defs.append('marker')
            .attr('id', 'arrow-marker-enabled')
            .attr('viewBox', '0 0 10 10')
            .attr('refX', 9)
            .attr('refY', 5)
            .attr('markerWidth', this.arrowSize)
            .attr('markerHeight', this.arrowSize)
            .attr('orient', 'auto-start-reverse')
            .append('path')
            .attr('d', 'M 0 0 L 10 5 L 0 10 z')
            .attr('fill', '#4CAF50');
    }

    setupEventListeners() {
        if (this.svg) {
            this.svg.on('mousemove.transition', this.handleMouseMove);
            this.svg.on('click.transition', this.handleClick);
        }
        document.addEventListener('keydown', this.handleKeyDown);
    }

    // Start drawing a transition from a source state
    startDrawing(stateElement, stateData) {
        if (this.isDrawing) return;

        this.isDrawing = true;
        this.sourceState = {
            element: stateElement,
            data: stateData,
            center: this.getStateCenter(stateData)
        };

        // Create temporary line
        this.tempLine = this.tempGroup.append('path')
            .attr('class', 'temp-transition')
            .attr('stroke', '#2196F3')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5')
            .attr('fill', 'none')
            .attr('marker-end', 'url(#arrow-marker-highlight)');

        this.updateTempLine();

        this.dispatchEvent(new CustomEvent('drawStart', {
            detail: { source: stateData.label }
        }));
    }

    // Complete the transition to a target state
    completeDrawing(stateElement, stateData) {
        if (!this.isDrawing || !this.sourceState) return;

        const sourceLabel = this.sourceState.data.label;
        const targetLabel = stateData.label;

        // Clean up
        this.cancelDrawing();

        // Notify orchestrator to create the transition
        if (this.orchestrator) {
            this.orchestrator.completeTransitionDraw(targetLabel);
        } else {
            // Fallback: dispatch event
            this.dispatchEvent(new CustomEvent('drawComplete', {
                detail: {
                    source: sourceLabel,
                    target: targetLabel
                }
            }));
        }
    }

    // Cancel the current drawing operation
    cancelDrawing() {
        if (this.tempLine) {
            this.tempLine.remove();
            this.tempLine = null;
        }
        this.isDrawing = false;
        this.sourceState = null;

        this.dispatchEvent(new CustomEvent('drawCancel'));
    }

    handleMouseMove(event) {
        const [x, y] = d3.pointer(event, this.svg.node());
        this.mousePosition = { x, y };

        if (this.isDrawing) {
            this.updateTempLine();
        }
    }

    handleClick(event) {
        // Only handle clicks on the canvas background when drawing
        if (this.isDrawing && event.target === this.svg.node()) {
            // Clicked on empty space - cancel drawing
            this.cancelDrawing();
        }
    }

    handleKeyDown(event) {
        if (event.key === 'Escape' && this.isDrawing) {
            this.cancelDrawing();
        }
    }

    updateTempLine() {
        if (!this.tempLine || !this.sourceState) return;

        const path = this.createLinePath(
            this.sourceState.center,
            this.mousePosition
        );
        this.tempLine.attr('d', path);
    }

    // Render all transitions for the current machine
    renderTransitions(machine) {
        if (!this.transitionsGroup || !machine?.statechart?.transitions) return;

        const transitions = machine.statechart.transitions;
        const statePositions = this.collectStatePositions(machine.statechart.root_state);
        const configuration = machine.configuration?.states?.map(s => s.label) || [];

        // Bind data and update
        const transitionSelection = this.transitionsGroup.selectAll('.transition')
            .data(transitions, d => d.label || `${d.from[0]}_${d.to[0]}_${d.event}`);

        // Remove old transitions
        transitionSelection.exit().remove();

        // Add new transitions
        const enterSelection = transitionSelection.enter()
            .append('g')
            .attr('class', 'transition')
            .attr('data-label', d => d.label);

        // Add path
        enterSelection.append('path')
            .attr('class', 'transition-path')
            .attr('fill', 'none')
            .attr('stroke', '#666')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrow-marker)');

        // Add invisible hit area for easier clicking
        enterSelection.append('path')
            .attr('class', 'transition-hit-area')
            .attr('fill', 'none')
            .attr('stroke', 'transparent')
            .attr('stroke-width', 15);

        // Add label background
        enterSelection.append('rect')
            .attr('class', 'transition-label-bg')
            .attr('fill', 'white')
            .attr('rx', 3)
            .attr('ry', 3);

        // Add label
        enterSelection.append('text')
            .attr('class', 'transition-label')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', '11px')
            .attr('fill', '#333');

        // Update all transitions
        const allTransitions = this.transitionsGroup.selectAll('.transition');

        allTransitions.each((d, i, nodes) => {
            const group = d3.select(nodes[i]);
            this.updateTransitionVisuals(group, d, statePositions, configuration);
        });

        // Set up click handlers
        allTransitions
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                event.stopPropagation();
                this.dispatchEvent(new CustomEvent('transitionClick', {
                    detail: { transition: d }
                }));
            })
            .on('dblclick', (event, d) => {
                event.stopPropagation();
                this.dispatchEvent(new CustomEvent('transitionDblClick', {
                    detail: { transition: d }
                }));
            });
    }

    updateTransitionVisuals(group, transition, statePositions, configuration) {
        const fromLabel = transition.from[0];
        const toLabel = transition.to[0];

        const fromPos = statePositions.get(fromLabel);
        const toPos = statePositions.get(toLabel);

        if (!fromPos || !toPos) return;

        const isSelfLoop = fromLabel === toLabel;
        const path = isSelfLoop
            ? this.createSelfLoopPath(fromPos)
            : this.createCurvePath(fromPos, toPos, transition);

        // Check if transition is enabled (for simulation)
        const isFromActive = configuration.includes(fromLabel);

        // Update path
        group.select('.transition-path')
            .attr('d', path)
            .attr('stroke', isFromActive ? '#4CAF50' : '#666')
            .attr('marker-end', isFromActive ? 'url(#arrow-marker-enabled)' : 'url(#arrow-marker)');

        group.select('.transition-hit-area')
            .attr('d', path);

        // Update label
        const labelPos = this.getLabelPosition(fromPos, toPos, isSelfLoop);
        const labelText = transition.event || transition.label || '';

        const textElement = group.select('.transition-label')
            .attr('x', labelPos.x)
            .attr('y', labelPos.y)
            .text(labelText);

        // Update label background
        const bbox = textElement.node()?.getBBox();
        if (bbox) {
            group.select('.transition-label-bg')
                .attr('x', bbox.x - 4)
                .attr('y', bbox.y - 2)
                .attr('width', bbox.width + 8)
                .attr('height', bbox.height + 4);
        }
    }

    collectStatePositions(state, positions = new Map()) {
        if (!state) return positions;

        positions.set(state.label, {
            x: state.x || 0,
            y: state.y || 0,
            width: state.width || 120,
            height: state.height || 60
        });

        if (state.children) {
            state.children.forEach(child => {
                this.collectStatePositions(child, positions);
            });
        }

        return positions;
    }

    getStateCenter(stateData) {
        return {
            x: (stateData.x || 0) + (stateData.width || 120) / 2,
            y: (stateData.y || 0) + (stateData.height || 60) / 2
        };
    }

    // Create a simple line path
    createLinePath(from, to) {
        return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
    }

    // Create a curved path between two states
    createCurvePath(fromPos, toPos, transition) {
        const fromCenter = {
            x: fromPos.x + fromPos.width / 2,
            y: fromPos.y + fromPos.height / 2
        };
        const toCenter = {
            x: toPos.x + toPos.width / 2,
            y: toPos.y + toPos.height / 2
        };

        // Calculate edge points
        const fromEdge = this.getEdgePoint(fromPos, toCenter);
        const toEdge = this.getEdgePoint(toPos, fromCenter);

        // Calculate control points for bezier curve
        const dx = toEdge.x - fromEdge.x;
        const dy = toEdge.y - fromEdge.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Perpendicular offset for curve
        const offset = Math.min(this.curveOffset, dist / 4);
        const nx = -dy / dist * offset;
        const ny = dx / dist * offset;

        const midX = (fromEdge.x + toEdge.x) / 2;
        const midY = (fromEdge.y + toEdge.y) / 2;

        // Use quadratic bezier for cleaner curves
        return `M ${fromEdge.x} ${fromEdge.y} Q ${midX + nx} ${midY + ny} ${toEdge.x} ${toEdge.y}`;
    }

    // Create a self-loop path
    createSelfLoopPath(statePos) {
        const centerX = statePos.x + statePos.width / 2;
        const topY = statePos.y;
        const r = this.selfLoopRadius;

        // Arc from top-left to top-right of state
        const startX = centerX - 15;
        const startY = topY;
        const endX = centerX + 15;
        const endY = topY;

        return `M ${startX} ${startY}
                C ${startX - r} ${startY - r * 1.5}
                  ${endX + r} ${endY - r * 1.5}
                  ${endX} ${endY}`;
    }

    // Get the point on the edge of a state rectangle
    getEdgePoint(statePos, targetPoint) {
        const cx = statePos.x + statePos.width / 2;
        const cy = statePos.y + statePos.height / 2;
        const hw = statePos.width / 2;
        const hh = statePos.height / 2;

        const dx = targetPoint.x - cx;
        const dy = targetPoint.y - cy;

        if (dx === 0 && dy === 0) {
            return { x: cx, y: cy - hh }; // Default to top
        }

        // Calculate intersection with rectangle edges
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);

        let scale;
        if (absDx * hh > absDy * hw) {
            // Intersects left or right edge
            scale = hw / absDx;
        } else {
            // Intersects top or bottom edge
            scale = hh / absDy;
        }

        return {
            x: cx + dx * scale,
            y: cy + dy * scale
        };
    }

    // Get the position for the transition label
    getLabelPosition(fromPos, toPos, isSelfLoop) {
        if (isSelfLoop) {
            return {
                x: fromPos.x + fromPos.width / 2,
                y: fromPos.y - this.selfLoopRadius - this.labelOffset
            };
        }

        const fromCenter = {
            x: fromPos.x + fromPos.width / 2,
            y: fromPos.y + fromPos.height / 2
        };
        const toCenter = {
            x: toPos.x + toPos.width / 2,
            y: toPos.y + toPos.height / 2
        };

        // Position label at midpoint, slightly offset
        return {
            x: (fromCenter.x + toCenter.x) / 2,
            y: (fromCenter.y + toCenter.y) / 2 - this.labelOffset
        };
    }

    // Highlight a transition
    highlightTransition(transitionLabel, highlight = true) {
        const transition = this.transitionsGroup?.select(`[data-label="${transitionLabel}"]`);
        if (transition) {
            transition.select('.transition-path')
                .attr('stroke', highlight ? '#2196F3' : '#666')
                .attr('stroke-width', highlight ? 3 : 2)
                .attr('marker-end', highlight ? 'url(#arrow-marker-highlight)' : 'url(#arrow-marker)');

            transition.classed('highlighted', highlight);
        }
    }

    // Highlight enabled transitions (for simulation mode)
    highlightEnabledTransitions(enabledLabels) {
        this.transitionsGroup?.selectAll('.transition').each((d, i, nodes) => {
            const group = d3.select(nodes[i]);
            const isEnabled = enabledLabels.includes(d.label);

            group.select('.transition-path')
                .classed('enabled', isEnabled)
                .attr('stroke', isEnabled ? '#4CAF50' : '#666')
                .attr('stroke-width', isEnabled ? 3 : 2);

            if (isEnabled) {
                group.raise(); // Bring enabled transitions to front
            }
        });
    }

    // Clean up
    destroy() {
        this.cancelDrawing();
        document.removeEventListener('keydown', this.handleKeyDown);
        if (this.svg) {
            this.svg.on('mousemove.transition', null);
            this.svg.on('click.transition', null);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TransitionEditor };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.TransitionEditor = TransitionEditor;
}
