/**
 * State Editor - Direct manipulation of statechart states
 * Handles state creation, editing, positioning, and visual representation
 */
class StateEditor {
    constructor(canvasControls) {
        this.canvas = canvasControls;
        this.states = new Map();
        this.dragState = null;
        this.editingState = null;
        this.stateCounter = 0;

        // State type definitions
        this.stateTypes = {
            BASIC: 1,
            NORMAL: 2,
            PARALLEL: 3,
            INITIAL: 4,
            FINAL: 5,
            HISTORY: 6
        };

        // Visual settings
        this.settings = {
            defaultWidth: 120,
            defaultHeight: 60,
            minWidth: 80,
            minHeight: 40,
            borderRadius: 8,
            fontSize: 14,
            fontFamily: 'var(--font-family-primary)',
            colors: {
                basic: {
                    fill: 'var(--color-bg-tertiary)',
                    stroke: 'var(--color-border-accent)',
                    text: 'var(--color-text-primary)'
                },
                parallel: {
                    fill: 'var(--color-bg-tertiary)',
                    stroke: 'var(--color-info)',
                    text: 'var(--color-text-primary)'
                },
                initial: {
                    fill: 'var(--color-success)',
                    stroke: 'var(--color-success)',
                    text: 'white'
                },
                final: {
                    fill: 'var(--color-error)',
                    stroke: 'var(--color-error)',
                    text: 'white'
                },
                active: {
                    fill: 'var(--color-accent-primary)',
                    stroke: 'var(--color-accent-active)',
                    text: 'white'
                }
            }
        };

        this.setupEventListeners();
        this.initializeArrowMarkers();
    }

    setupEventListeners() {
        // Listen for canvas events
        this.canvas.container.addEventListener('createstate', (e) => {
            this.createState(e.detail.x, e.detail.y);
        });

        this.canvas.container.addEventListener('deleteelements', (e) => {
            this.deleteStates(e.detail.elements);
        });

        // Listen for keyboard events
        document.addEventListener('keydown', (e) => {
            if (document.activeElement && document.activeElement.tagName.match(/input|textarea/i)) {
                return;
            }

            if (e.key === 'Enter' && this.editingState) {
                this.finishEditing();
            } else if (e.key === 'Escape' && this.editingState) {
                this.cancelEditing();
            }
        });
    }

    initializeArrowMarkers() {
        // Create arrow markers for different state types
        const defs = this.canvas.svg.select('defs');
        if (defs.empty()) {
            this.canvas.svg.append('defs');
        }

        // Standard arrow marker
        this.canvas.svg.select('defs').append('marker')
            .attr('id', 'state-arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', 'var(--color-accent-primary)');
    }

    // State creation and management
    createState(x, y, options = {}) {
        const stateId = options.id || `state_${++this.stateCounter}`;
        const stateType = options.type || this.stateTypes.BASIC;
        const label = options.label || this.generateStateLabel(stateType);

        const state = {
            id: stateId,
            type: stateType,
            label: label,
            x: x,
            y: y,
            width: options.width || this.settings.defaultWidth,
            height: options.height || this.settings.defaultHeight,
            children: [],
            parent: options.parent || null,
            active: false,
            data: options.data || {}
        };

        this.states.set(stateId, state);
        this.renderState(state);

        // Start editing if it's a new user-created state
        if (!options.skipEdit) {
            setTimeout(() => this.startEditing(stateId), 100);
        }

        return state;
    }

    generateStateLabel(type) {
        const labels = {
            [this.stateTypes.BASIC]: 'State',
            [this.stateTypes.NORMAL]: 'Compound',
            [this.stateTypes.PARALLEL]: 'Parallel',
            [this.stateTypes.INITIAL]: 'Initial',
            [this.stateTypes.FINAL]: 'Final',
            [this.stateTypes.HISTORY]: 'History'
        };

        return labels[type] || 'State';
    }

    renderState(state) {
        const stateGroup = this.canvas.statesGroup.append('g')
            .attr('class', 'state-group')
            .attr('data-state-id', state.id)
            .attr('transform', `translate(${state.x}, ${state.y})`);

        // Render based on state type
        switch (state.type) {
            case this.stateTypes.INITIAL:
                this.renderInitialState(stateGroup, state);
                break;
            case this.stateTypes.FINAL:
                this.renderFinalState(stateGroup, state);
                break;
            case this.stateTypes.HISTORY:
                this.renderHistoryState(stateGroup, state);
                break;
            default:
                this.renderNormalState(stateGroup, state);
        }

        this.setupStateInteractions(stateGroup, state);
    }

    renderNormalState(group, state) {
        const colors = this.getStateColors(state);
        const isDashed = state.type === this.stateTypes.PARALLEL;

        // Main rectangle
        const rect = group.append('rect')
            .attr('class', 'state-node')
            .attr('data-state-id', state.id)
            .attr('x', -state.width / 2)
            .attr('y', -state.height / 2)
            .attr('width', state.width)
            .attr('height', state.height)
            .attr('rx', this.settings.borderRadius)
            .attr('ry', this.settings.borderRadius)
            .style('fill', colors.fill)
            .style('stroke', colors.stroke)
            .style('stroke-width', '2')
            .style('cursor', 'pointer');

        if (isDashed) {
            rect.style('stroke-dasharray', '8,4');
        }

        // State label
        group.append('text')
            .attr('class', 'state-label')
            .attr('data-state-id', state.id)
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .style('fill', colors.text)
            .style('font-size', `${this.settings.fontSize}px`)
            .style('font-family', this.settings.fontFamily)
            .style('font-weight', '500')
            .style('pointer-events', 'none')
            .style('user-select', 'none')
            .text(state.label);

        // Add connection handles
        this.addConnectionHandles(group, state);

        // Add state type indicator
        this.addStateTypeIndicator(group, state);
    }

    renderInitialState(group, state) {
        const colors = this.getStateColors(state);
        const radius = 15;

        // Circle
        group.append('circle')
            .attr('class', 'state-node initial-state')
            .attr('data-state-id', state.id)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', radius)
            .style('fill', colors.fill)
            .style('stroke', colors.stroke)
            .style('stroke-width', '3')
            .style('cursor', 'pointer');

        // Arrow pointing right
        group.append('path')
            .attr('d', 'M -5 0 L 5 0 M 0 -5 L 5 0 L 0 5')
            .style('stroke', colors.text)
            .style('stroke-width', '2')
            .style('fill', 'none')
            .style('pointer-events', 'none');

        // Update state dimensions for initial state
        state.width = radius * 2;
        state.height = radius * 2;

        this.addConnectionHandles(group, state);
    }

    renderFinalState(group, state) {
        const colors = this.getStateColors(state);
        const outerRadius = 18;
        const innerRadius = 12;

        // Outer circle
        group.append('circle')
            .attr('class', 'state-node final-state')
            .attr('data-state-id', state.id)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', outerRadius)
            .style('fill', 'none')
            .style('stroke', colors.stroke)
            .style('stroke-width', '3')
            .style('cursor', 'pointer');

        // Inner circle
        group.append('circle')
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', innerRadius)
            .style('fill', colors.fill)
            .style('stroke', 'none')
            .style('pointer-events', 'none');

        // Update state dimensions
        state.width = outerRadius * 2;
        state.height = outerRadius * 2;

        this.addConnectionHandles(group, state);
    }

    renderHistoryState(group, state) {
        const colors = this.getStateColors(state);
        const radius = 15;

        // Circle
        group.append('circle')
            .attr('class', 'state-node history-state')
            .attr('data-state-id', state.id)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', radius)
            .style('fill', colors.fill)
            .style('stroke', colors.stroke)
            .style('stroke-width', '2')
            .style('cursor', 'pointer');

        // H symbol
        group.append('text')
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .style('fill', colors.text)
            .style('font-size', '16px')
            .style('font-family', this.settings.fontFamily)
            .style('font-weight', 'bold')
            .style('pointer-events', 'none')
            .text('H');

        // Update state dimensions
        state.width = radius * 2;
        state.height = radius * 2;

        this.addConnectionHandles(group, state);
    }

    addConnectionHandles(group, state) {
        const handles = [
            { x: 0, y: -state.height / 2, position: 'top' },
            { x: state.width / 2, y: 0, position: 'right' },
            { x: 0, y: state.height / 2, position: 'bottom' },
            { x: -state.width / 2, y: 0, position: 'left' }
        ];

        const handleGroup = group.append('g')
            .attr('class', 'connection-handles')
            .style('opacity', '0');

        handles.forEach(handle => {
            handleGroup.append('circle')
                .attr('class', 'connection-handle')
                .attr('data-position', handle.position)
                .attr('cx', handle.x)
                .attr('cy', handle.y)
                .attr('r', 6)
                .style('fill', 'var(--color-accent-primary)')
                .style('stroke', 'white')
                .style('stroke-width', '2')
                .style('cursor', 'crosshair')
                .on('mouseenter', function () {
                    d3.select(this).attr('r', 8);
                })
                .on('mouseleave', function () {
                    d3.select(this).attr('r', 6);
                });
        });
    }

    addStateTypeIndicator(group, state) {
        if (state.type === this.stateTypes.BASIC) return;

        const indicators = {
            [this.stateTypes.NORMAL]: '∘',
            [this.stateTypes.PARALLEL]: '∥'
        };

        const indicator = indicators[state.type];
        if (!indicator) return;

        group.append('text')
            .attr('class', 'state-type-indicator')
            .attr('x', state.width / 2 - 10)
            .attr('y', -state.height / 2 + 10)
            .attr('text-anchor', 'middle')
            .style('fill', 'var(--color-text-secondary)')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .style('pointer-events', 'none')
            .text(indicator);
    }

    setupStateInteractions(group, state) {
        const stateNode = group.select('.state-node');

        // Mouse events
        stateNode
            .on('mouseenter', () => this.handleStateHover(state.id, true))
            .on('mouseleave', () => this.handleStateHover(state.id, false))
            .on('click', (event) => this.handleStateClick(event, state.id))
            .on('dblclick', (event) => this.handleStateDoubleClick(event, state.id))
            .on('contextmenu', (event) => this.handleStateContextMenu(event, state.id));

        // Drag behavior
        const drag = d3.drag()
            .on('start', (event) => this.handleDragStart(event, state.id))
            .on('drag', (event) => this.handleDrag(event, state.id))
            .on('end', (event) => this.handleDragEnd(event, state.id));

        group.call(drag);
    }

    // State interactions
    handleStateHover(stateId, isEntering) {
        const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);
        const handles = stateGroup.select('.connection-handles');

        handles.transition()
            .duration(200)
            .style('opacity', isEntering ? '1' : '0');

        // Visual feedback
        stateGroup.select('.state-node')
            .classed('hovered', isEntering);
    }

    handleStateClick(event, stateId) {
        event.stopPropagation();

        if (this.canvas.currentTool === 'select') {
            this.canvas.toggleSelection(stateId);
        } else if (this.canvas.currentTool === 'add-transition') {
            this.handleTransitionStart(stateId);
        }
    }

    handleStateDoubleClick(event, stateId) {
        event.stopPropagation();
        this.startEditing(stateId);
    }

    handleStateContextMenu(event, stateId) {
        event.preventDefault();
        event.stopPropagation();

        this.showStateContextMenu(stateId, event.clientX, event.clientY);
    }

    // Drag and drop
    handleDragStart(event, stateId) {
        if (this.canvas.currentTool !== 'select') return;

        this.dragState = {
            id: stateId,
            startX: event.x,
            startY: event.y,
            originalX: this.states.get(stateId).x,
            originalY: this.states.get(stateId).y
        };

        // Visual feedback
        this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`)
            .classed('dragging', true)
            .style('opacity', '0.8');
    }

    handleDrag(event, stateId) {
        if (!this.dragState || this.dragState.id !== stateId) return;

        const state = this.states.get(stateId);
        const dx = event.x - this.dragState.startX;
        const dy = event.y - this.dragState.startY;

        const newX = this.dragState.originalX + dx;
        const newY = this.dragState.originalY + dy;

        const snapped = this.canvas.snapToGrid(newX, newY);

        // Update state position
        state.x = snapped.x;
        state.y = snapped.y;

        // Update visual position
        this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`)
            .attr('transform', `translate(${snapped.x}, ${snapped.y})`);

        // Update connections
        this.updateConnections(stateId);

        // Drop zone detection for reparenting
        this.detectDropTarget(stateId, snapped.x, snapped.y);
    }

    // Detect potential drop targets for reparenting
    detectDropTarget(draggedStateId, x, y) {
        const draggedState = this.states.get(draggedStateId);
        if (!draggedState) return;

        // Clear previous drop target highlighting
        this.canvas.statesGroup.selectAll('.state-group')
            .classed('drop-target', false);

        this.dragState.dropTarget = null;

        // Find compound states that the dragged state could be dropped into
        this.states.forEach((state, id) => {
            if (id === draggedStateId) return;

            // Only compound states (type 2) or parallel states (type 3) can be drop targets
            if (state.type !== 2 && state.type !== 3) return;

            // Check if dragged state is inside this state's bounds
            const halfWidth = (state.width || this.settings.defaultWidth) / 2;
            const halfHeight = (state.height || this.settings.defaultHeight) / 2;

            const isInside = x >= state.x - halfWidth &&
                x <= state.x + halfWidth &&
                y >= state.y - halfHeight &&
                y <= state.y + halfHeight;

            if (isInside) {
                // Highlight as potential drop target
                this.canvas.statesGroup.select(`[data-state-id="${id}"]`)
                    .classed('drop-target', true);
                this.dragState.dropTarget = id;
            }
        });
    }

    handleDragEnd(event, stateId) {
        if (!this.dragState || this.dragState.id !== stateId) return;

        // Check if we should reparent the state
        const dropTarget = this.dragState.dropTarget;

        // Remove visual feedback
        this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`)
            .classed('dragging', false)
            .style('opacity', '1');

        // Clear drop target highlighting
        this.canvas.statesGroup.selectAll('.state-group')
            .classed('drop-target', false);

        // Perform reparenting if dropped on a valid target
        if (dropTarget && dropTarget !== stateId) {
            this.reparentState(stateId, dropTarget);
        }

        this.dragState = null;

        // Dispatch state moved event
        this.dispatchEvent('statemoved', {
            stateId: stateId,
            x: this.states.get(stateId).x,
            y: this.states.get(stateId).y
        });
    }

    // Reparent a state to a new parent
    reparentState(stateId, newParentId) {
        const state = this.states.get(stateId);
        const newParent = this.states.get(newParentId);

        if (!state || !newParent) return;

        // Get the current parent
        const currentParent = state.parent;

        // Don't reparent to the same parent
        if (currentParent === newParentId) return;

        // Update the state's parent reference
        const oldParent = state.parent;
        state.parent = newParentId;

        // Dispatch reparent event for undo/redo and state manager
        this.dispatchEvent('statereparented', {
            stateId: stateId,
            fromParent: oldParent,
            toParent: newParentId
        });

        // Auto-resize parent to contain children
        this.autoResizeParent(newParentId);
    }

    // Auto-resize a parent state to contain all its children
    autoResizeParent(parentId) {
        const parent = this.states.get(parentId);
        if (!parent) return;

        // Find all children of this parent
        const children = [];
        this.states.forEach((state, id) => {
            if (state.parent === parentId) {
                children.push(state);
            }
        });

        if (children.length === 0) return;

        // Calculate bounding box of children
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        children.forEach(child => {
            const halfWidth = (child.width || this.settings.defaultWidth) / 2;
            const halfHeight = (child.height || this.settings.defaultHeight) / 2;

            minX = Math.min(minX, child.x - halfWidth);
            minY = Math.min(minY, child.y - halfHeight);
            maxX = Math.max(maxX, child.x + halfWidth);
            maxY = Math.max(maxY, child.y + halfHeight);
        });

        // Add padding
        const padding = 20;
        const headerHeight = 25; // Space for parent label

        // Calculate new parent size
        const newWidth = Math.max(maxX - minX + padding * 2, this.settings.minWidth);
        const newHeight = Math.max(maxY - minY + padding * 2 + headerHeight, this.settings.minHeight);

        // Update parent size
        parent.width = newWidth;
        parent.height = newHeight;

        // Update parent position to center on children
        parent.x = (minX + maxX) / 2;
        parent.y = (minY + maxY) / 2 + headerHeight / 2;

        // Update visual representation
        this.updateStateVisual(parentId);
    }

    // Text editing
    startEditing(stateId) {
        if (this.editingState) {
            this.finishEditing();
        }

        const state = this.states.get(stateId);
        if (!state) return;

        this.editingState = stateId;

        const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);
        const label = stateGroup.select('.state-label');

        // Hide the original label
        label.style('display', 'none');

        // Create foreign object for text input
        const fo = stateGroup.append('foreignObject')
            .attr('class', 'text-editor')
            .attr('x', -state.width / 2)
            .attr('y', -10)
            .attr('width', state.width)
            .attr('height', 20);

        const input = fo.append('xhtml:input')
            .attr('type', 'text')
            .attr('value', state.label)
            .style('width', '100%')
            .style('height', '100%')
            .style('border', 'none')
            .style('background', 'transparent')
            .style('color', 'var(--color-text-primary)')
            .style('font-size', `${this.settings.fontSize}px`)
            .style('font-family', this.settings.fontFamily)
            .style('text-align', 'center')
            .style('outline', 'none')
            .on('blur', () => this.finishEditing())
            .on('keydown', (event) => {
                if (event.key === 'Enter') {
                    this.finishEditing();
                } else if (event.key === 'Escape') {
                    this.cancelEditing();
                }
                event.stopPropagation();
            });

        // Focus and select text
        input.node().focus();
        input.node().select();
    }

    finishEditing() {
        if (!this.editingState) return;

        const stateId = this.editingState;
        const state = this.states.get(stateId);
        const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);

        // Get new text value
        const input = stateGroup.select('.text-editor input');
        const newLabel = input.node().value.trim() || state.label;

        // Update state
        state.label = newLabel;

        // Update label text
        stateGroup.select('.state-label')
            .text(newLabel)
            .style('display', 'block');

        // Remove editor
        stateGroup.select('.text-editor').remove();

        this.editingState = null;

        // Dispatch event
        this.dispatchEvent('statelabeled', {
            stateId: stateId,
            label: newLabel
        });
    }

    cancelEditing() {
        if (!this.editingState) return;

        const stateId = this.editingState;
        const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);

        // Restore original label
        stateGroup.select('.state-label').style('display', 'block');

        // Remove editor
        stateGroup.select('.text-editor').remove();

        this.editingState = null;
    }

    // State management
    updateState(stateId, updates) {
        const state = this.states.get(stateId);
        if (!state) return false;

        Object.assign(state, updates);
        this.rerenderState(stateId);
        return true;
    }

    rerenderState(stateId) {
        const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);
        stateGroup.remove();

        const state = this.states.get(stateId);
        if (state) {
            this.renderState(state);
        }
    }

    deleteStates(stateIds) {
        stateIds.forEach(stateId => {
            const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);
            stateGroup.remove();

            this.states.delete(stateId);

            // Remove from selection
            this.canvas.selectedElements.delete(stateId);
        });

        // Update connections
        this.updateAllConnections();

        // Dispatch event
        this.dispatchEvent('statesdeleted', { stateIds });
    }

    setStateActive(stateId, active) {
        const state = this.states.get(stateId);
        if (!state) return;

        state.active = active;

        const stateGroup = this.canvas.statesGroup.select(`[data-state-id="${stateId}"]`);
        stateGroup.select('.state-node').classed('active', active);
        stateGroup.select('.state-label').classed('active', active);
    }

    // Utility methods
    getStateColors(state) {
        if (state.active) {
            return this.settings.colors.active;
        }

        const typeColors = {
            [this.stateTypes.PARALLEL]: this.settings.colors.parallel,
            [this.stateTypes.INITIAL]: this.settings.colors.initial,
            [this.stateTypes.FINAL]: this.settings.colors.final
        };

        return typeColors[state.type] || this.settings.colors.basic;
    }

    updateConnections(stateId) {
        // This will be implemented when we add the transition system
        this.dispatchEvent('updateconnections', { stateId });
    }

    updateAllConnections() {
        // This will be implemented when we add the transition system
        this.dispatchEvent('updateallconnections');
    }

    showStateContextMenu(stateId, screenX, screenY) {
        const state = this.states.get(stateId);

        const menu = document.createElement('div');
        menu.className = 'context-menu state-context-menu';
        menu.style.position = 'fixed';
        menu.style.left = screenX + 'px';
        menu.style.top = screenY + 'px';
        menu.style.zIndex = '9999';

        const menuItems = [
            { text: 'Edit Label', action: () => this.startEditing(stateId) }
        ];

        // Add specific options based on state type
        if (state.type === this.stateTypes.HISTORY) {
            const isDeep = state.history_type === 2; // HISTORY_TYPE_DEEP = 2
            menuItems.push({
                text: isDeep ? 'Switch to Shallow (H)' : 'Switch to Deep (H*)',
                action: () => this.toggleHistoryType(stateId)
            });
        }

        menuItems.push(
            { text: 'Make Initial', action: () => this.setStateType(stateId, this.stateTypes.INITIAL) },
            { text: 'Make Final', action: () => this.setStateType(stateId, this.stateTypes.FINAL) },
            { text: 'Make Parallel', action: () => this.setStateType(stateId, this.stateTypes.PARALLEL) },
            { text: 'Make Normal', action: () => this.setStateType(stateId, this.stateTypes.NORMAL) },
            { text: 'Make History', action: () => this.setStateType(stateId, this.stateTypes.HISTORY) },
            { text: '---', action: null },
            { text: 'Delete', action: () => this.deleteStates([stateId]) }
        );

        menuItems.forEach(item => {
            if (item.text === '---') {
                const separator = document.createElement('hr');
                separator.className = 'context-menu-separator';
                menu.appendChild(separator);
                return;
            }

            const menuItem = document.createElement('div');
            menuItem.className = 'context-menu-item';
            menuItem.textContent = item.text;
            menuItem.addEventListener('click', () => {
                if (item.action) item.action();
                this.hideContextMenu();
            });
            menu.appendChild(menuItem);
        });

        document.body.appendChild(menu);

        // Hide menu on click outside
        const hideMenu = (e) => {
            if (!menu.contains(e.target)) {
                this.hideContextMenu();
                document.removeEventListener('click', hideMenu);
            }
        };

        setTimeout(() => {
            document.addEventListener('click', hideMenu);
        }, 0);
    }

    hideContextMenu() {
        const menu = document.querySelector('.state-context-menu');
        if (menu) {
            menu.remove();
        }
    }

    setStateType(stateId, type) {
        const state = this.states.get(stateId);
        if (!state) return;

        state.type = type;
        state.label = this.generateStateLabel(type);

        // Initialize default history type if switching to history
        if (type === this.stateTypes.HISTORY && !state.history_type) {
            state.history_type = 1; // Default to SHALLOW
        }

        this.rerenderState(stateId);

        this.dispatchEvent('statetypechanged', {
            stateId: stateId,
            type: type
        });
    }

    toggleHistoryType(stateId) {
        const state = this.states.get(stateId);
        if (!state || state.type !== this.stateTypes.HISTORY) return;

        // Toggle between 1 (SHALLOW) and 2 (DEEP)
        // If undefined/0, set to 2 (DEEP) because existing might be default shallow
        state.history_type = (state.history_type === 2) ? 1 : 2;

        this.rerenderState(stateId);

        this.dispatchEvent('statehistorytypechanged', {
            stateId: stateId,
            historyType: state.history_type
        });
    }

    handleTransitionStart(stateId) {
        // This will be implemented in the transition editor
        this.dispatchEvent('transitionstart', { fromStateId: stateId });
    }

    // Event system
    dispatchEvent(type, detail) {
        const event = new CustomEvent(type, { detail });
        this.canvas.container.dispatchEvent(event);
    }

    // Public API
    getAllStates() {
        return Array.from(this.states.values());
    }

    getState(stateId) {
        return this.states.get(stateId);
    }

    clearStates() {
        this.canvas.statesGroup.selectAll('*').remove();
        this.states.clear();
        this.canvas.clearSelection();
    }

    loadStates(statesData) {
        this.clearStates();

        statesData.forEach(stateData => {
            this.createState(stateData.x, stateData.y, {
                id: stateData.id,
                type: stateData.type,
                label: stateData.label,
                width: stateData.width,
                height: stateData.height,
                data: stateData.data,
                skipEdit: true
            });
        });
    }
}

// Export for use in other modules
window.StateEditor = StateEditor;