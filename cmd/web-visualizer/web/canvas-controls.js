/**
 * Canvas Controls - Interactive canvas with zoom, pan, and viewport management
 * Provides the foundation for the interactive statechart editor
 */
class CanvasControls {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            minZoom: 0.1,
            maxZoom: 10,
            zoomSpeed: 0.1,
            panSpeed: 1,
            gridSize: 20,
            showGrid: true,
            snapToGrid: true,
            ...options
        };
        
        // Transform state
        this.transform = {
            x: 0,
            y: 0,
            scale: 1
        };
        
        // Interaction state
        this.isDragging = false;
        this.dragStart = null;
        this.currentTool = 'select';
        this.selectedElements = new Set();
        
        // D3 zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([this.options.minZoom, this.options.maxZoom])
            .on('zoom', this.handleZoom.bind(this))
            .filter(this.zoomFilter.bind(this));
        
        this.initializeCanvas();
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
    }
    
    initializeCanvas() {
        // Clear existing content
        this.container.innerHTML = '';
        
        // Create SVG canvas
        const rect = this.container.getBoundingClientRect();
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .style('cursor', 'default')
            .call(this.zoom);
        
        // Create main groups
        this.createCanvasGroups();
        
        // Add grid if enabled
        if (this.options.showGrid) {
            this.createGrid();
        }
        
        // Add zoom controls overlay
        this.createZoomControls();
        
        // Initialize viewport
        this.resetViewport();
    }
    
    createCanvasGroups() {
        // Background group (grid, etc.)
        this.backgroundGroup = this.svg.append('g')
            .attr('class', 'canvas-background');
        
        // Main content group (transforms with zoom/pan)
        this.mainGroup = this.svg.append('g')
            .attr('class', 'canvas-main');
        
        // Transitions group (behind states)
        this.transitionsGroup = this.mainGroup.append('g')
            .attr('class', 'transitions-group');
        
        // States group
        this.statesGroup = this.mainGroup.append('g')
            .attr('class', 'states-group');
        
        // Overlay group (selection boxes, handles, etc.)
        this.overlayGroup = this.mainGroup.append('g')
            .attr('class', 'overlay-group');
        
        // UI overlay group (doesn't transform)
        this.uiGroup = this.svg.append('g')
            .attr('class', 'ui-overlay');
    }
    
    createGrid() {
        const gridSize = this.options.gridSize;
        const rect = this.container.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        // Create grid pattern
        const defs = this.svg.append('defs');
        
        // Small grid pattern
        const smallGrid = defs.append('pattern')
            .attr('id', 'small-grid')
            .attr('width', gridSize)
            .attr('height', gridSize)
            .attr('patternUnits', 'userSpaceOnUse');
        
        smallGrid.append('path')
            .attr('d', `M ${gridSize} 0 L 0 0 0 ${gridSize}`)
            .attr('fill', 'none')
            .attr('stroke', 'var(--color-border-secondary)')
            .attr('stroke-width', 0.5)
            .attr('opacity', '0.3');
        
        // Large grid pattern
        const largeGrid = defs.append('pattern')
            .attr('id', 'large-grid')
            .attr('width', gridSize * 5)
            .attr('height', gridSize * 5)
            .attr('patternUnits', 'userSpaceOnUse');
        
        largeGrid.append('path')
            .attr('d', `M ${gridSize * 5} 0 L 0 0 0 ${gridSize * 5}`)
            .attr('fill', 'none')
            .attr('stroke', 'var(--color-border-primary)')
            .attr('stroke-width', 1)
            .attr('opacity', '0.2');
        
        // Apply grid to background
        this.backgroundGroup.append('rect')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('fill', 'url(#small-grid)');
        
        this.backgroundGroup.append('rect')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('fill', 'url(#large-grid)');
    }
    
    createZoomControls() {
        // Create zoom control buttons
        const controls = this.container.querySelector('.zoom-controls');
        if (!controls) return;
        
        const zoomInBtn = controls.querySelector('.zoom-button[title="Zoom In"]');
        const zoomOutBtn = controls.querySelector('.zoom-button[title="Zoom Out"]');
        const resetBtn = controls.querySelector('.zoom-button[title="Reset Zoom"]');
        
        if (zoomInBtn) {
            zoomInBtn.addEventListener('click', () => this.zoomIn());
        }
        
        if (zoomOutBtn) {
            zoomOutBtn.addEventListener('click', () => this.zoomOut());
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetViewport());
        }
        
        // Add fit to content button
        const fitBtn = this.container.parentElement.querySelector('.panel-action[title="Fit to Screen"]');
        if (fitBtn) {
            fitBtn.addEventListener('click', () => this.fitToContent());
        }
    }
    
    setupEventListeners() {
        // Mouse events for canvas interaction
        this.svg.on('click', this.handleCanvasClick.bind(this));
        this.svg.on('dblclick', this.handleCanvasDoubleClick.bind(this));
        this.svg.on('contextmenu', this.handleCanvasContextMenu.bind(this));
        
        // Prevent default drag behavior on certain elements
        this.svg.on('dragstart', () => d3.event.preventDefault());
        
        // Tool button events
        document.querySelectorAll('.tool-button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tool = e.currentTarget.dataset.tool;
                if (tool) {
                    this.setTool(tool);
                }
            });
        });
        
        // Resize observer for responsive behavior
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(entries => {
                this.handleResize();
            });
            this.resizeObserver.observe(this.container);
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when the canvas is focused or no input is active
            if (document.activeElement && document.activeElement.tagName.match(/input|textarea|select/i)) {
                return;
            }
            
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    this.setTool(this.currentTool === 'pan' ? 'select' : 'pan');
                    break;
                case 'v':
                case 'Escape':
                    this.setTool('select');
                    break;
                case 'h':
                    this.setTool('pan');
                    break;
                case 's':
                    if (!e.ctrlKey && !e.metaKey) {
                        this.setTool('add-state');
                    }
                    break;
                case 't':
                    this.setTool('add-transition');
                    break;
                case 'Delete':
                case 'Backspace':
                    this.deleteSelected();
                    break;
                case '=':
                case '+':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.zoomIn();
                    }
                    break;
                case '-':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.zoomOut();
                    }
                    break;
                case '0':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.resetViewport();
                    }
                    break;
                case 'a':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.selectAll();
                    }
                    break;
            }
        });
    }
    
    // Zoom and pan methods
    handleZoom(event) {
        const { transform } = event;
        this.transform = {
            x: transform.x,
            y: transform.y,
            scale: transform.k
        };
        
        this.mainGroup.attr('transform', transform);
        this.updateGridVisibility();
        this.updateUI();
    }
    
    zoomFilter(event) {
        // Allow zooming with wheel or specific tools
        if (event.type === 'wheel') {
            return this.currentTool !== 'add-state' && this.currentTool !== 'add-transition';
        }
        
        // Allow zoom gestures
        if (event.type === 'touchstart') {
            return event.touches.length > 1;
        }
        
        // Allow pan tool or space+drag
        return this.currentTool === 'pan' || event.button === 1 || 
               (event.type === 'mousedown' && (event.shiftKey || event.ctrlKey));
    }
    
    zoomIn() {
        const rect = this.container.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        this.svg.transition()
            .duration(300)
            .call(this.zoom.scaleBy, 1 + this.options.zoomSpeed);
    }
    
    zoomOut() {
        const rect = this.container.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        this.svg.transition()
            .duration(300)
            .call(this.zoom.scaleBy, 1 - this.options.zoomSpeed);
    }
    
    resetViewport() {
        const rect = this.container.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        this.svg.transition()
            .duration(500)
            .call(this.zoom.transform, d3.zoomIdentity.translate(centerX, centerY));
    }
    
    fitToContent() {
        const contentBounds = this.getContentBounds();
        if (!contentBounds) {
            this.resetViewport();
            return;
        }
        
        const rect = this.container.getBoundingClientRect();
        const padding = 50;
        
        const contentWidth = contentBounds.width + padding * 2;
        const contentHeight = contentBounds.height + padding * 2;
        
        const scaleX = rect.width / contentWidth;
        const scaleY = rect.height / contentHeight;
        const scale = Math.min(scaleX, scaleY, this.options.maxZoom);
        
        const translateX = rect.width / 2 - (contentBounds.x + contentBounds.width / 2) * scale;
        const translateY = rect.height / 2 - (contentBounds.y + contentBounds.height / 2) * scale;
        
        this.svg.transition()
            .duration(500)
            .call(this.zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
    }
    
    // Canvas interaction methods
    handleCanvasClick(event) {
        const [x, y] = d3.pointer(event, this.mainGroup.node());
        
        if (this.currentTool === 'add-state') {
            this.createState(x, y);
        } else if (this.currentTool === 'select') {
            this.handleSelection(event);
        }
    }
    
    handleCanvasDoubleClick(event) {
        const [x, y] = d3.pointer(event, this.mainGroup.node());
        
        if (this.currentTool === 'select') {
            // Create state on double-click
            this.createState(x, y);
        }
    }
    
    handleCanvasContextMenu(event) {
        event.preventDefault();
        const [x, y] = d3.pointer(event, this.mainGroup.node());
        
        // Show context menu
        this.showContextMenu(x, y, event.clientX, event.clientY);
    }
    
    handleSelection(event) {
        const target = event.target;
        const isMultiSelect = event.ctrlKey || event.metaKey;
        
        if (!isMultiSelect) {
            this.clearSelection();
        }
        
        // Handle element selection
        if (target.classList.contains('state-node')) {
            const stateId = target.dataset.stateId;
            if (stateId) {
                this.toggleSelection(stateId);
            }
        }
    }
    
    // Tool management
    setTool(tool) {
        this.currentTool = tool;
        
        // Update tool buttons
        document.querySelectorAll('.tool-button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tool === tool);
        });
        
        // Update cursor
        this.updateCursor();
        
        // Dispatch tool change event
        this.dispatchEvent('toolchange', { tool });
    }
    
    updateCursor() {
        const cursors = {
            'select': 'default',
            'pan': 'grab',
            'zoom': 'zoom-in',
            'add-state': 'crosshair',
            'add-transition': 'crosshair'
        };
        
        this.svg.style('cursor', cursors[this.currentTool] || 'default');
    }
    
    // Selection management
    toggleSelection(elementId) {
        if (this.selectedElements.has(elementId)) {
            this.selectedElements.delete(elementId);
            this.updateSelectionUI(elementId, false);
        } else {
            this.selectedElements.add(elementId);
            this.updateSelectionUI(elementId, true);
        }
        
        this.dispatchEvent('selectionchange', { 
            selected: Array.from(this.selectedElements) 
        });
    }
    
    clearSelection() {
        this.selectedElements.forEach(id => {
            this.updateSelectionUI(id, false);
        });
        this.selectedElements.clear();
        
        this.dispatchEvent('selectionchange', { selected: [] });
    }
    
    selectAll() {
        // Select all states
        this.statesGroup.selectAll('.state-node').each((d, i, nodes) => {
            const stateId = nodes[i].dataset.stateId;
            if (stateId) {
                this.selectedElements.add(stateId);
                this.updateSelectionUI(stateId, true);
            }
        });
        
        this.dispatchEvent('selectionchange', { 
            selected: Array.from(this.selectedElements) 
        });
    }
    
    updateSelectionUI(elementId, selected) {
        const element = this.svg.select(`[data-state-id="${elementId}"]`);
        element.classed('selected', selected);
        
        if (selected) {
            this.showSelectionHandles(elementId);
        } else {
            this.hideSelectionHandles(elementId);
        }
    }
    
    showSelectionHandles(elementId) {
        const element = this.svg.select(`[data-state-id="${elementId}"]`);
        const bbox = element.node().getBBox();
        
        // Create selection box
        const selectionBox = this.overlayGroup.append('rect')
            .attr('class', 'selection-box')
            .attr('data-element-id', elementId)
            .attr('x', bbox.x - 3)
            .attr('y', bbox.y - 3)
            .attr('width', bbox.width + 6)
            .attr('height', bbox.height + 6)
            .style('fill', 'none')
            .style('stroke', 'var(--color-accent-primary)')
            .style('stroke-width', '2')
            .style('stroke-dasharray', '5,5')
            .style('opacity', '0.8');
        
        // Add resize handles
        this.addResizeHandles(elementId, bbox);
    }
    
    hideSelectionHandles(elementId) {
        this.overlayGroup.selectAll(`[data-element-id="${elementId}"]`).remove();
    }
    
    addResizeHandles(elementId, bbox) {
        const handles = [
            { x: bbox.x - 3, y: bbox.y - 3, cursor: 'nw-resize' },
            { x: bbox.x + bbox.width / 2 - 3, y: bbox.y - 3, cursor: 'n-resize' },
            { x: bbox.x + bbox.width + 3, y: bbox.y - 3, cursor: 'ne-resize' },
            { x: bbox.x + bbox.width + 3, y: bbox.y + bbox.height / 2 - 3, cursor: 'e-resize' },
            { x: bbox.x + bbox.width + 3, y: bbox.y + bbox.height + 3, cursor: 'se-resize' },
            { x: bbox.x + bbox.width / 2 - 3, y: bbox.y + bbox.height + 3, cursor: 's-resize' },
            { x: bbox.x - 3, y: bbox.y + bbox.height + 3, cursor: 'sw-resize' },
            { x: bbox.x - 3, y: bbox.y + bbox.height / 2 - 3, cursor: 'w-resize' }
        ];
        
        handles.forEach((handle, i) => {
            this.overlayGroup.append('rect')
                .attr('class', 'resize-handle')
                .attr('data-element-id', elementId)
                .attr('data-handle-index', i)
                .attr('x', handle.x)
                .attr('y', handle.y)
                .attr('width', 6)
                .attr('height', 6)
                .style('fill', 'var(--color-accent-primary)')
                .style('stroke', 'white')
                .style('stroke-width', '1')
                .style('cursor', handle.cursor);
        });
    }
    
    // Utility methods
    getContentBounds() {
        const states = this.statesGroup.selectAll('.state-node');
        if (states.empty()) return null;
        
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        states.each(function() {
            const bbox = this.getBBox();
            minX = Math.min(minX, bbox.x);
            minY = Math.min(minY, bbox.y);
            maxX = Math.max(maxX, bbox.x + bbox.width);
            maxY = Math.max(maxY, bbox.y + bbox.height);
        });
        
        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }
    
    updateGridVisibility() {
        const scale = this.transform.scale;
        const gridOpacity = Math.max(0, Math.min(1, (scale - 0.3) / 0.7));
        
        this.backgroundGroup.selectAll('rect')
            .style('opacity', gridOpacity);
    }
    
    updateUI() {
        // Update zoom level display
        const zoomLevel = Math.round(this.transform.scale * 100);
        const zoomDisplay = document.querySelector('.zoom-level');
        if (zoomDisplay) {
            zoomDisplay.textContent = `${zoomLevel}%`;
        }
    }
    
    handleResize() {
        // Handle container resize
        this.updateUI();
    }
    
    // Snap to grid functionality
    snapToGrid(x, y) {
        if (!this.options.snapToGrid) return { x, y };
        
        const gridSize = this.options.gridSize;
        return {
            x: Math.round(x / gridSize) * gridSize,
            y: Math.round(y / gridSize) * gridSize
        };
    }
    
    // Context menu
    showContextMenu(canvasX, canvasY, screenX, screenY) {
        // Create context menu
        const menu = document.createElement('div');
        menu.className = 'context-menu';
        menu.style.position = 'fixed';
        menu.style.left = screenX + 'px';
        menu.style.top = screenY + 'px';
        menu.style.zIndex = '9999';
        
        const menuItems = [
            { text: 'Add State', action: () => this.createState(canvasX, canvasY) },
            { text: 'Paste', action: () => this.paste(canvasX, canvasY) },
            { text: 'Select All', action: () => this.selectAll() }
        ];
        
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.className = 'context-menu-item';
            menuItem.textContent = item.text;
            menuItem.addEventListener('click', () => {
                item.action();
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
        const menu = document.querySelector('.context-menu');
        if (menu) {
            menu.remove();
        }
    }
    
    // State creation placeholder
    createState(x, y) {
        const snapped = this.snapToGrid(x, y);
        
        // Dispatch event for state creation
        this.dispatchEvent('createstate', {
            x: snapped.x,
            y: snapped.y
        });
    }
    
    // Utility methods
    deleteSelected() {
        if (this.selectedElements.size === 0) return;
        
        this.dispatchEvent('deleteelements', {
            elements: Array.from(this.selectedElements)
        });
    }
    
    paste(x, y) {
        // Placeholder for paste functionality
        this.dispatchEvent('paste', { x, y });
    }
    
    // Event system
    dispatchEvent(type, detail) {
        const event = new CustomEvent(type, { detail });
        this.container.dispatchEvent(event);
    }
    
    // Public API
    getTransform() {
        return { ...this.transform };
    }
    
    setTransform(x, y, scale) {
        this.svg.call(this.zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
    }
    
    getSelectedElements() {
        return Array.from(this.selectedElements);
    }
    
    destroy() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        
        this.container.innerHTML = '';
    }
}

// Export for use in other modules
window.CanvasControls = CanvasControls;