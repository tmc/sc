// Import the new tab system components
import stateManager from './state-manager.js';
// import TabSystem from './tab-system.js';
// import KeyboardShortcuts from './keyboard-shortcuts.js';
import webSocketClient from './websocket-client.js';
import { PerformanceOptimizer } from './performance-optimizer.js';
import { fromJson } from "@bufbuild/protobuf";
import { StatechartSchema } from "./gen/statecharts/v1/statecharts_pb.js";

class StatechartVisualizer {
    constructor() {
        this.currentMachine = null;
        this.examples = {};
        this.svg = null;
        this.simulation = null;

        // Enhanced visualization controls
        this.zoom = null;
        this.transform = null;
        this.mainGroup = null;
        this.stateNodes = null;
        this.transitionLines = null;

        // Visualization state
        this.currentStatechart = null;
        this.nodeMap = new Map();
        this.bounds = { width: 0, height: 0, minX: 0, minY: 0, maxX: 0, maxY: 0 };

        // Performance optimization
        this.performanceOptimizer = null;

        // Tab system (disabled)
        this.tabSystem = null;
        this.keyboardShortcuts = null;

        // Bound methods
        this.handleResize = this.handleResize.bind(this);
        this.handleZoom = this.handleZoom.bind(this);
        this.initializeLayout = this.initializeLayout.bind(this);
    }

    // ... lines 41-182 skipped ...

    initializeTabSystem() {
        // Tab system temporarily disabled due to missing modules
        console.warn('TabSystem and KeyboardShortcuts modules are missing. Tab system disabled.');
        /*
        // Replace the right panel with the tab system
        const rightPanel = document.querySelector('.panel-right');
        if (rightPanel) {
            // Initialize tab system in the right panel
            this.tabSystem = new TabSystem(rightPanel);

            // Initialize keyboard shortcuts
            this.keyboardShortcuts = new KeyboardShortcuts(this.tabSystem);

            // Bind tab system events
            this.bindTabSystemEvents();
        }
        */
    }

        // New tab system components
        this.stateManager = stateManager;
this.tabSystem = null;
this.keyboardShortcuts = null;
this.webSocketClient = webSocketClient;
this.eventInputSystem = null;
this.exportManager = null;

this.initializeUI();
this.initializeTabSystem();
this.initializeEventSystem();
this.initializeExportSystem();
this.initializePerformanceOptimizer();
this.initializeSync();
this.loadExamples();
    }

initializeSync() {
    if (!window.EventSource) {
        console.warn('EventSource not supported');
        return;
    }

    const eventSource = new EventSource('/api/v1/sync');

    eventSource.onopen = () => {
        console.log('Sync connection established');
    };

    eventSource.onerror = (err) => {
        console.error('Sync connection error:', err);
    };

    eventSource.addEventListener('file_change', (e) => {
        try {
            const data = JSON.parse(e.data);
            console.log('File changed:', data);
            // Reload examples to get fresh data
            this.loadExamples();

            // If the current machine is based on this file, we could reload it
            // Logic: check if current machine ID matches. 
            // Currently generated IDs are random, so we can't easily map back without extra metadata.
            // For now, just reloading the dropdown is good.
        } catch (err) {
            console.error('Failed to parse sync event:', err);
        }
    });
}

initializeUI() {
    // Example selection
    document.getElementById('load-example').addEventListener('click', () => {
        this.loadSelectedExample();
    });

    // Machine controls
    document.getElementById('reset-machine').addEventListener('click', () => {
        this.resetMachine();
    });

    // Save button
    const saveBtn = document.getElementById('save-machine');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            this.saveMachine();
        });
    }

    // Add Machine button (Left Panel)
    const addMachineBtn = document.querySelector('.panel-action[title="Add Machine"]');
    if (addMachineBtn) {
        addMachineBtn.addEventListener('click', () => {
            this.createNewMachine();
        });
    }

    // Refresh button
    const refreshBtn = document.querySelector('.panel-action[title="Refresh"]');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            this.loadExamples();
        });
    }

    // Event controls
    document.getElementById('send-event').addEventListener('click', () => {
        this.sendEvent();
    });

    document.getElementById('event-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            this.sendEvent();
        }
    });

    // Enhanced visualization controls
    this.initializeVisualizationControls();

    // Update status
    this.updateConnectionStatus('disconnected');
    this.updateMachineInfo('No machine loaded');

    // Initialize notifications
    this.initializeNotifications();

    // Initialize connection status click handler
    this.initializeConnectionStatusHandler();

    // Listen for WebSocket status updates
    window.addEventListener('websocketStatusUpdate', (e) => {
        const { status, attempts, clientId } = e.detail;
        let message = null;
        if (status === 'connecting') {
            message = `Connecting...`;
            if (attempts > 0) message = `Reconnecting... (attempt ${attempts})`;
        } else if (status === 'connected') {
            message = clientId ? `Connected (${clientId})` : `Connected...`;
        } else if (status === 'error') {
            message = `Connection error`;
        }
        this.updateConnectionStatus(status, message);
    });

    // Check initial status
    if (this.webSocketClient && typeof this.webSocketClient.getConnectionState === 'function') {
        const status = this.webSocketClient.getConnectionState();
        const clientId = this.webSocketClient.getClientId();
        let message = null;
        if (status === 'connected') {
            message = clientId ? `Connected (${clientId})` : `Connected...`;
        } else if (status === 'connecting') {
            message = `Connecting...`;
        }
        this.updateConnectionStatus(status, message);
    }
}

initializeVisualizationControls() {
    // Zoom controls
    const zoomInBtn = document.querySelector('.zoom-controls .zoom-button[title="Zoom In"]');
    const zoomOutBtn = document.querySelector('.zoom-controls .zoom-button[title="Zoom Out"]');
    const resetZoomBtn = document.querySelector('.zoom-controls .zoom-button[title="Reset Zoom"]');

    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => this.zoomIn());
    }
    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => this.zoomOut());
    }
    if (resetZoomBtn) {
        resetZoomBtn.addEventListener('click', () => this.resetZoom());
    }

    // Fit to screen control
    const fitToScreenBtn = document.querySelector('.panel-action[title="Fit to Screen"]');
    if (fitToScreenBtn) {
        fitToScreenBtn.addEventListener('click', () => this.fitToScreen());
    }

    // Export control
    const exportBtn = document.querySelector('.panel-action[title="Export"]');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => this.exportVisualization());
    }

    // Tool selection
    document.querySelectorAll('.tool-button').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tool = e.currentTarget.dataset.tool;
            this.setVisualizationTool(tool);
        });
    });
}

initializeTabSystem() {
    // Replace the right panel with the tab system
    const rightPanel = document.querySelector('.panel-right');
    if (rightPanel) {
        // Initialize tab system in the right panel
        this.tabSystem = new TabSystem(rightPanel);

        // Initialize keyboard shortcuts
        this.keyboardShortcuts = new KeyboardShortcuts(this.tabSystem);

        // Bind tab system events
        this.bindTabSystemEvents();
    }
}

initializeEventSystem() {
    // Initialize event input system
    if (typeof EventInputSystem !== 'undefined') {
        this.eventInputSystem = new EventInputSystem();

        // Make it available globally for integration
        window.eventInputSystem = this.eventInputSystem;
    }
}

initializeExportSystem() {
    // Initialize export manager
    if (typeof ExportManager !== 'undefined') {
        this.exportManager = new ExportManager();

        // Make it available globally for integration
        window.exportManager = this.exportManager;
    }
}

initializePerformanceOptimizer() {
    // Initialize performance optimizer
    this.performanceOptimizer = new PerformanceOptimizer(this);

    // Add performance status display
    this.addPerformanceDisplay();

    console.log('✅ Performance optimizer initialized');
}

addPerformanceDisplay() {
    // Add performance indicator to status bar
    const statusBar = document.getElementById('status-bar');
    if (statusBar) {
        const perfIndicator = document.createElement('div');
        perfIndicator.className = 'status-item performance-indicator';
        perfIndicator.innerHTML = `
                <span class="status-text" id="performance-fps">60 FPS</span>
                <span class="status-text" id="performance-nodes">0 nodes</span>
                <span class="status-text" id="performance-lod">High</span>
            `;

        const statusRight = statusBar.querySelector('.status-right');
        if (statusRight) {
            statusRight.insertBefore(perfIndicator, statusRight.firstChild);
        }
    }
}

bindTabSystemEvents() {
    // Handle keyboard shortcuts
    this.keyboardShortcuts.addEventListener('newMachine', () => {
        this.createNewMachine();
    });

    this.keyboardShortcuts.addEventListener('saveMachine', () => {
        this.saveMachine();
    });

    this.keyboardShortcuts.addEventListener('resetMachine', () => {
        this.resetMachine();
    });

    // Handle WebSocket events
    this.webSocketClient.addEventListener('eventProcessed', (e) => {
        this.handleEventProcessed(e.detail);
    });

    this.webSocketClient.addEventListener('machineStateUpdate', (e) => {
        this.handleMachineStateUpdate(e.detail);
    });
}

createNewMachine() {
    // Create a new blank machine
    const machineId = `machine-${Date.now()}`;
    const blankMachine = {
        id: machineId,
        statechart: this.getBlankStatechart(),
        configuration: ['idle'],
        context: {},
        created_at: new Date().toISOString()
    };

    this.stateManager.addMachine(blankMachine);

    // Set as current machine
    this.currentMachine = blankMachine;

    // Connect to systems
    if (this.eventInputSystem) {
        this.eventInputSystem.setMachine(blankMachine);
    }
    if (this.exportManager) {
        this.exportManager.setMachine(blankMachine);
    }

    this.visualizeStatechart(blankMachine.statechart);
}

getBlankStatechart() {
    return {
        rootState: {
            label: 'root',
            type: 2, // compound
            children: [
                {
                    label: 'idle',
                    type: 1, // atomic
                    children: []
                }
            ]
        },
        transitions: [],
        events: []
    };
}

    async saveMachine() {
    const currentMachine = this.stateManager.getCurrentMachine();
    if (!currentMachine) {
        alert('No machine to save');
        return;
    }

    // If it's a file-backed chart, save to file
    if (currentMachine.filename) {
        try {
            console.log(`Saving to chart file: ${currentMachine.filename}`);
            const response = await fetch(`/api/v1/charts/${currentMachine.filename}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentMachine.statechart), // Save just the statechart definition
            });

            if (response.ok) {
                console.log('Chart saved successfully to disk');
                // Also update the running machine instance
                this.saveToPersistence(currentMachine);
                this.showNotification('Chart saved to disk');
            } else {
                const error = await response.text();
                throw new Error(error);
            }
        } catch (error) {
            console.error('Save file error:', error);
            alert('Failed to save chart file: ' + error.message);
        }
        return;
    }

    // Legacy/Runtime persistence
    this.saveToPersistence(currentMachine);
}

async saveToPersistence(machine) {
    try {
        const response = await fetch(`/api/v1/machines/${machine.id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(machine),
        });

        if (response.ok) {
            console.log('Machine saved successfully');
            this.showNotification('Machine saved');
        } else {
            throw new Error('Failed to save machine');
        }
    } catch (error) {
        console.error('Save error:', error);
        alert('Failed to save machine: ' + error.message);
    }
}

handleEventProcessed(eventData) {
    const { machineId, data } = eventData;

    if (this.currentMachine && this.currentMachine.id === machineId) {
        // Update machine state
        this.currentMachine.configuration = data.configuration;
        this.currentMachine.context = data.context;

        // Update state manager
        this.stateManager.updateMachine(machineId, {
            configuration: data.configuration,
            context: data.context
        });

        // Update visualization
        this.updateVisualizationHighlights();

        // Add to history
        this.addToHistory(data.event, data.last_step);
    }
}

handleMachineStateUpdate(stateData) {
    const { machineId, data } = stateData;

    if (this.currentMachine && this.currentMachine.id === machineId) {
        // Update machine state
        this.currentMachine.configuration = data.configuration;
        this.currentMachine.context = data.context;

        // Update state manager
        this.stateManager.updateMachine(machineId, {
            configuration: data.configuration,
            context: data.context,
            state: data.state
        });

        // Update visualization
        this.updateVisualizationHighlights();
    }
}

    async loadExamples() {
    try {
        const selectElement = document.getElementById('example-select');
        selectElement.innerHTML = '<option value="">Select an example...</option>';
        this.examples = {};

        // 1. Load Charts from Disk (Bidi)
        try {
            const chartResp = await fetch(`/api/v1/charts?t=${Date.now()}`);
            if (chartResp.ok) {
                const chartData = await chartResp.json();
                if (chartData.charts && chartData.charts.length > 0) {
                    const group = document.createElement('optgroup');
                    group.label = "Local Charts (Editable)";

                    for (const chart of chartData.charts) {
                        const option = document.createElement('option');
                        option.value = `chart:${chart.filename}`;
                        option.textContent = chart.filename;
                        group.appendChild(option);
                        // We load content lazily in loadSelectedExample
                    }
                    selectElement.appendChild(group);
                }
            }
        } catch (e) {
            console.warn("Failed to load charts:", e);
        }

        // 2. Load Legacy Examples
        console.log('Fetching examples...');
        const response = await fetch(`/api/v1/examples?t=${Date.now()}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const examplesGroup = document.createElement('optgroup');
        examplesGroup.label = "Examples (Read-only)";

        const keys = Object.keys(data);
        for (const [key, val] of Object.entries(data)) {
            this.examples[key] = fromJson(StatechartSchema, val);
            const option = document.createElement('option');
            option.value = key;
            const displayName = key.replace(/^file_/, '').replace(/_/g, ' ');
            option.textContent = displayName.charAt(0).toUpperCase() + displayName.slice(1);
            examplesGroup.appendChild(option);
        }
        selectElement.appendChild(examplesGroup);

    } catch (error) {
        console.error('Failed to load examples:', error);
        const selectElement = document.getElementById('example-select');
        selectElement.innerHTML = '<option value="">Error loading examples</option>';
        const errorOption = document.createElement('option');
        errorOption.disabled = true;
        errorOption.textContent = error.message;
        selectElement.appendChild(errorOption);
    }
}

    async loadSelectedExample() {
    const selectedValue = document.getElementById('example-select').value;
    if (!selectedValue) return;

    let statechart = null;
    let filename = null;

    if (selectedValue.startsWith('chart:')) {
        // Load from Chart API
        filename = selectedValue.substring(6);
        try {
            const res = await fetch(`/api/v1/charts/${filename}`);
            if (!res.ok) throw new Error("Failed to fetch chart");
            const json = await res.json();
            // TODO: Validate/Migrate if needed. Assuming ProtoJSON compatible structure or standard JSON
            // If it's pure JSON, we might need to be careful with Proto fields. 
            // For now assuming the format matches.
            statechart = json;
        } catch (e) {
            alert("Error loading chart: " + e.message);
            return;
        }
    } else {
        // Load from Examples map
        statechart = this.examples[selectedValue];
    }

    if (!statechart) return;

    try {
        const machineId = filename
            ? `chart-${filename.replace(/\./g, '-')}`
            : `example-${selectedValue}-${Date.now()}`;

        // Create the runtime machine
        const response = await fetch('/api/v1/machines', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                id: machineId,
                statechart: statechart,
                context: {}
            }),
        });

        if (response.ok) {
            const machine = await response.json();
            if (filename) {
                machine.filename = filename;
            }
            this.currentMachine = machine;

            // Add to state manager
            this.stateManager.addMachine(machine);

            // Connect to event input system
            if (this.eventInputSystem) {
                this.eventInputSystem.setMachine(machine);
            }

            // Connect to export manager
            if (this.exportManager) {
                this.exportManager.setMachine(machine);
            }

            this.visualizeStatechart(statechart);
            this.updateMachineInfo(`Machine: ${machineId}`);
        } else {
            const error = await response.text();
            console.error('Failed to create machine:', error);
            alert('Failed to create machine: ' + error);
        }
    } catch (error) {
        console.error('Failed to load example:', error);
        alert('Failed to load example: ' + error.message);
    }
}

    async resetMachine() {
    if (!this.currentMachine) {
        return;
    }

    try {
        const response = await fetch(`/api/v1/machines/${this.currentMachine.id}/reset`, {
            method: 'POST',
        });

        if (response.ok) {
            const machine = await response.json();
            this.currentMachine = machine;

            // Update state manager
            this.stateManager.updateMachine(machine.id, machine);

            this.updateMachineDisplay();
        } else {
            const error = await response.text();
            console.error('Failed to reset machine:', error);
            alert('Failed to reset machine: ' + error);
        }
    } catch (error) {
        console.error('Failed to reset machine:', error);
        alert('Failed to reset machine: ' + error.message);
    }
}

    async sendEvent() {
    if (!this.currentMachine) {
        alert('No machine loaded');
        return;
    }

    const eventInput = document.getElementById('event-input');
    const eventName = eventInput.value.trim();

    if (!eventName) {
        alert('Please enter an event name');
        return;
    }

    try {
        const response = await fetch(`/api/v1/machines/${this.currentMachine.id}/events`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                event: eventName,
            }),
        });

        if (response.ok) {
            const result = await response.json();
            this.currentMachine.configuration = result.configuration;
            this.currentMachine.context = result.context;

            // Update state manager
            this.stateManager.updateMachine(this.currentMachine.id, {
                configuration: result.configuration,
                context: result.context
            });

            // Add to event history
            this.stateManager.addEventToHistory(eventName, result.last_step);

            this.updateMachineDisplay();
            eventInput.value = '';
        } else {
            const error = await response.text();
            console.error('Failed to send event:', error);
            alert('Failed to send event: ' + error);
        }
    } catch (error) {
        console.error('Failed to send event:', error);
        alert('Failed to send event: ' + error.message);
    }
}

updateMachineDisplay() {
    if (!this.currentMachine) {
        document.getElementById('current-states').innerHTML = '<p class="placeholder">No machine loaded</p>';
        document.getElementById('context-json').textContent = '{}';
        document.getElementById('event-buttons').innerHTML = '';
        return;
    }

    // Update current states
    const statesContainer = document.getElementById('current-states');
    if (this.currentMachine.configuration && this.currentMachine.configuration.length > 0) {
        statesContainer.innerHTML = this.currentMachine.configuration
            .map(state => `<span class="current-state">${state}</span>`)
            .join('');
    } else {
        statesContainer.innerHTML = '<p class="placeholder">No active states</p>';
    }

    // Update context
    document.getElementById('context-json').textContent =
        JSON.stringify(this.currentMachine.context || {}, null, 2);

    // Update visualization
    this.updateVisualizationHighlights();
}

visualizeStatechart(statechart) {
    if (!statechart) return;

    this.currentStatechart = statechart;
    const container = document.getElementById('statechart-container');
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = container.clientHeight;
    this.bounds.width = width;
    this.bounds.height = height;

    // Performance check - determine if we need optimization
    const nodeCount = this.performanceOptimizer ? this.performanceOptimizer.countNodes(statechart) : 0;
    const useOptimizedRendering = nodeCount > 100;

    // Create SVG with zoom/pan support
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('cursor', 'default');

    this.svg = svg;

    // Setup zoom behavior with performance optimization
    this.zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            this.transform = event.transform;
            this.onZoomChanged(event.transform);
        });

    svg.call(this.zoom);

    // Add definitions for markers and patterns
    this.setupSVGDefinitions(svg);

    // Create main group for all content
    this.mainGroup = svg.append('g')
        .attr('class', 'main-group');

    if (useOptimizedRendering && this.performanceOptimizer) {
        // Use performance-optimized rendering
        this.renderWithOptimization(statechart, svg);
    } else {
        // Use standard rendering for smaller statecharts
        this.renderStandard(statechart, svg);
    }

    // Initialize minimap
    this.initializeMinimap();

    // Create event buttons
    this.createEventButtons(statechart.events);

    // Fit to screen initially
    this.fitToScreen();

    // Update performance display
    this.updatePerformanceDisplay(nodeCount, useOptimizedRendering);
}

onZoomChanged(transform) {
    this.transform = transform;

    // Apply transform to main group
    this.mainGroup.attr('transform', transform);

    // Update minimap
    this.updateMinimap();

    // If using performance optimization, re-render with new viewport
    if (this.performanceOptimizer && this.currentStatechart) {
        const nodeCount = this.performanceOptimizer.countNodes(this.currentStatechart);
        if (nodeCount > 100) {
            this.performanceOptimizer.optimizeStatechartRendering(
                this.currentStatechart,
                this.svg,
                transform
            );
        }
    }
}

renderWithOptimization(statechart, svg) {
    console.log('🚀 Using performance-optimized rendering');

    // Use performance optimizer for large statecharts
    const renderResult = this.performanceOptimizer.optimizeStatechartRendering(
        statechart,
        svg,
        this.transform
    );

    console.log(`Rendered ${renderResult.nodesRendered} nodes, ${renderResult.transitionsRendered} transitions (${renderResult.strategy})`);
}

renderStandard(statechart, svg) {
    console.log('📊 Using standard rendering');

    // Create hierarchical layout
    const root = this.createHierarchy(statechart.rootState);
    const treeLayout = d3.tree().size([this.bounds.width - 200, this.bounds.height - 200]);
    const treeData = treeLayout(root);

    // Calculate content bounds
    this.calculateContentBounds(treeData);

    // Draw states and transitions using standard methods
    this.drawStates(this.mainGroup, treeData);
    this.drawTransitions(this.mainGroup, statechart.transitions, treeData);
}

updatePerformanceDisplay(nodeCount, optimized) {
    const fpsElement = document.getElementById('performance-fps');
    const nodesElement = document.getElementById('performance-nodes');
    const lodElement = document.getElementById('performance-lod');

    if (nodesElement) {
        nodesElement.textContent = `${nodeCount} nodes`;
    }

    if (lodElement) {
        const lod = optimized ?
            (this.performanceOptimizer ? this.performanceOptimizer.levelOfDetail.current : 'Medium') :
            'High';
        lodElement.textContent = lod;
    }

    // Update FPS display (will be updated by performance monitor)
    if (this.performanceOptimizer) {
        const fps = Math.round(this.performanceOptimizer.performanceMonitor.getCurrentFps());
        if (fpsElement) {
            fpsElement.textContent = `${fps} FPS`;
            fpsElement.className = fps < 30 ? 'status-text fps-low' :
                fps < 50 ? 'status-text fps-medium' :
                    'status-text fps-high';
        }
    }
}

createHierarchy(state) {
    const hierarchy = d3.hierarchy(state, (d) => d.children);
    return hierarchy;
}

drawStates(container, treeData) {
    // Clear existing node map and rebuild it
    this.nodeMap.clear();
    treeData.descendants().forEach(d => {
        this.nodeMap.set(d.data.label, d);
    });

    const nodes = container.selectAll('.state-group')
        .data(treeData.descendants())
        .enter()
        .append('g')
        .attr('class', d => {
            let classes = 'state-group';
            // Compound states (have children)
            if (d.children && d.children.length > 0) classes += ' compound';
            if (d.data.type === 3) classes += ' parallel'; // STATE_TYPE_PARALLEL
            if (d.data.isInitial) classes += ' initial';
            if (d.data.isFinal) classes += ' final';
            if (d.data.isHistory) {
                classes += ' history';
                if (d.data.historyType === 2) classes += ' history-deep'; // DEEP history
            }
            return classes;
        })
        .attr('data-nesting-level', d => d.depth)
        .attr('transform', d => `translate(${d.y}, ${d.x})`)
        .style('cursor', 'pointer');

    // Draw state rectangles with enhanced styling
    nodes.append('rect')
        .attr('class', d => {
            let classes = 'state-node';
            if (d.data.type === 3) classes += ' parallel'; // STATE_TYPE_PARALLEL
            if (d.data.isInitial) classes += ' initial';
            if (d.data.isFinal) classes += ' final';
            return classes;
        })
        .attr('x', -40)
        .attr('y', -15)
        .attr('width', 80)
        .attr('height', 30)
        .attr('rx', 5)
        .attr('fill', d => {
            if (d.data.isFinal) return '#e74c3c';
            if (d.data.isInitial) return '#2ecc71';
            if (d.data.type === 3) return '#f39c12'; // Parallel states
            return '#3498db';
        })
        .attr('stroke', '#2c3e50')
        .attr('stroke-width', 1)
        .on('mouseover', function (event, d) {
            d3.select(this)
                .attr('stroke-width', 2)
                .attr('stroke', '#e74c3c');

            // Show tooltip
            const tooltip = d3.select('body').append('div')
                .attr('class', 'tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '5px 10px')
                .style('border-radius', '3px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('opacity', 0);

            tooltip.transition().duration(200).style('opacity', 1);
            tooltip.html(`
                    <strong>${d.data.label}</strong><br/>
                    Type: ${d.data.type || 'basic'}<br/>
                    ${d.data.isInitial ? 'Initial State<br/>' : ''}
                    ${d.data.isFinal ? 'Final State<br/>' : ''}
                    Children: ${d.children ? d.children.length : 0}
                `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseout', function (event, d) {
            d3.select(this)
                .attr('stroke-width', 1)
                .attr('stroke', '#2c3e50');

            d3.selectAll('.tooltip').remove();
        })
        .on('click', function (event, d) {
            // Handle state selection
            nodes.selectAll('rect').attr('stroke-width', 1);
            d3.select(this).attr('stroke-width', 3);

            console.log('Selected state:', d.data.label);
        });

    // Add state labels with better positioning
    nodes.append('text')
        .attr('class', 'state-label')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('fill', 'white')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .text(d => d.data.label)
        .style('pointer-events', 'none');

    // Add type indicators for special states
    nodes.filter(d => d.data.isInitial)
        .append('circle')
        .attr('cx', -50)
        .attr('cy', 0)
        .attr('r', 3)
        .attr('fill', '#2ecc71');

    nodes.filter(d => d.data.isFinal)
        .append('circle')
        .attr('cx', 50)
        .attr('cy', 0)
        .attr('r', 3)
        .attr('fill', '#e74c3c');

    // Add history state indicator (H or H*)
    nodes.filter(d => d.data.isHistory)
        .append('text')
        .attr('class', 'history-indicator')
        .attr('x', 0)
        .attr('y', 0)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('fill', 'white')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text(d => d.data.historyType === 2 ? 'H*' : 'H');

    // Add compound state indicator (dashed border is in CSS)
    nodes.filter(d => d.children && d.children.length > 0)
        .append('text')
        .attr('class', 'compound-indicator')
        .attr('x', 30)
        .attr('y', -10)
        .attr('text-anchor', 'end')
        .attr('fill', '#888')
        .attr('font-size', '10px')
        .text(d => `(${d.children.length})`);

    // Store nodes for later highlighting
    this.stateNodes = nodes;
}

drawTransitions(container, transitions, treeData) {
    if (!transitions || transitions.length === 0) return;

    const transitionGroup = container.append('g').attr('class', 'transitions');

    transitions.forEach((transition, index) => {
        if (transition.from && transition.to && transition.from.length > 0 && transition.to.length > 0) {
            const fromNode = this.nodeMap.get(transition.from[0]);
            const toNode = this.nodeMap.get(transition.to[0]);

            if (fromNode && toNode) {
                // Create transition group
                const transitionElement = transitionGroup.append('g')
                    .attr('class', 'transition')
                    .style('cursor', 'pointer');

                // Draw transition line with arrow
                const path = transitionElement.append('path')
                    .attr('class', 'transition-line')
                    .attr('d', this.createTransitionPath(fromNode, toNode))
                    .attr('stroke', '#3498db')
                    .attr('stroke-width', 2)
                    .attr('fill', 'none')
                    .attr('marker-end', 'url(#arrowhead)')
                    .on('mouseover', function () {
                        d3.select(this)
                            .attr('stroke', '#e74c3c')
                            .attr('stroke-width', 3);
                    })
                    .on('mouseout', function () {
                        d3.select(this)
                            .attr('stroke', '#3498db')
                            .attr('stroke-width', 2);
                    })
                    .on('click', function (event) {
                        event.stopPropagation();
                        console.log('Selected transition:', transition);
                    });

                // Add transition label with background
                const midX = (fromNode.y + toNode.y) / 2;
                const midY = (fromNode.x + toNode.x) / 2;

                const eventText = transition.event || transition.label || 'event';

                // Background for label
                const labelBg = transitionElement.append('rect')
                    .attr('class', 'transition-label-bg')
                    .attr('x', midX - eventText.length * 3)
                    .attr('y', midY - 15)
                    .attr('width', eventText.length * 6)
                    .attr('height', 14)
                    .attr('rx', 3)
                    .attr('fill', 'white')
                    .attr('stroke', '#3498db')
                    .attr('stroke-width', 1);

                // Label text
                const label = transitionElement.append('text')
                    .attr('class', 'transition-label')
                    .attr('x', midX)
                    .attr('y', midY - 8)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '10px')
                    .attr('font-weight', 'bold')
                    .attr('fill', '#2c3e50')
                    .text(eventText)
                    .style('pointer-events', 'none');

                // Add guard condition if present
                if (transition.guard) {
                    transitionElement.append('text')
                        .attr('class', 'transition-guard')
                        .attr('x', midX)
                        .attr('y', midY + 5)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '8px')
                        .attr('font-style', 'italic')
                        .attr('fill', '#7f8c8d')
                        .text(`[${transition.guard}]`)
                        .style('pointer-events', 'none');
                }
            }
        }
    });

    // Store transition lines for highlighting
    this.transitionLines = transitionGroup.selectAll('.transition');
}

createTransitionPath(fromNode, toNode) {
    const fromX = fromNode.y;
    const fromY = fromNode.x;
    const toX = toNode.y;
    const toY = toNode.x;

    // Calculate connection points on node borders
    const dx = toX - fromX;
    const dy = toY - fromY;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance === 0) {
        // Self-transition (loop)
        const loopRadius = 25;
        return `M ${fromX + 40} ${fromY}
                    Q ${fromX + 40 + loopRadius} ${fromY - loopRadius}
                      ${fromX + 40} ${fromY - loopRadius * 2}
                    Q ${fromX + 40 - loopRadius} ${fromY - loopRadius}
                      ${fromX + 40} ${fromY}`;
    }

    const unitX = dx / distance;
    const unitY = dy / distance;

    // Calculate start and end points on node borders
    const nodeRadius = 40; // Half of node width
    const startX = fromX + unitX * nodeRadius;
    const startY = fromY + unitY * 15; // Half of node height
    const endX = toX - unitX * nodeRadius;
    const endY = toY - unitY * 15;

    // Create curved path with control points
    const controlOffset = Math.min(100, distance / 3);
    const controlX1 = startX + unitX * controlOffset;
    const controlY1 = startY + unitY * controlOffset;
    const controlX2 = endX - unitX * controlOffset;
    const controlY2 = endY - unitY * controlOffset;

    // Use cubic Bezier curve for smooth transitions
    return `M ${startX} ${startY} C ${controlX1} ${controlY1} ${controlX2} ${controlY2} ${endX} ${endY}`;
}

createEventButtons(events) {
    const buttonsContainer = document.getElementById('event-buttons');
    buttonsContainer.innerHTML = '';

    if (events && events.length > 0) {
        events.forEach(event => {
            const button = document.createElement('button');
            button.className = 'event-button';
            button.textContent = event.label;
            button.addEventListener('click', () => {
                document.getElementById('event-input').value = event.label;
                this.sendEvent();
            });
            buttonsContainer.appendChild(button);
        });
    }
}

updateVisualizationHighlights() {
    if (!this.stateNodes || !this.currentMachine) return;

    const activeStates = new Set(this.currentMachine.configuration || []);

    this.stateNodes.selectAll('.state-node')
        .classed('active', d => activeStates.has(d.data.label));

    this.stateNodes.selectAll('.state-label')
        .classed('active', d => activeStates.has(d.data.label));
}

addToHistory(eventName, step) {
    const historyContainer = document.getElementById('step-history');

    const stepElement = document.createElement('div');
    stepElement.className = 'history-step';

    const timestamp = new Date().toLocaleTimeString();
    stepElement.innerHTML = `
            <div class="history-event">Event: ${eventName}</div>
            <div class="history-states">Active States: ${(this.currentMachine.configuration || []).join(', ')}</div>
            <div style="font-size: 0.7em; color: #7f8c8d; margin-top: 0.2rem;">${timestamp}</div>
        `;

    historyContainer.insertBefore(stepElement, historyContainer.firstChild);

    // Limit history to 10 items
    while (historyContainer.children.length > 10) {
        historyContainer.removeChild(historyContainer.lastChild);
    }
}

updateConnectionStatus(status, message = null) {
    const statusElement = document.getElementById('connection-status');
    const statusText = statusElement.querySelector('.status-text');
    const statusIndicator = statusElement.querySelector('.status-indicator');

    if (!statusText || !statusIndicator) {
        console.error('Status elements not found');
        return;
    }

    // Remove existing status classes
    statusElement.classList.remove('connected', 'disconnected', 'connecting', 'error');

    // Update status classes and text based on connection state
    switch (status) {
        case 'connected':
            statusElement.classList.add('status-item', 'connected');
            statusText.textContent = message || 'Connected';
            break;
        case 'connecting':
            statusElement.classList.add('status-item', 'connecting');
            statusText.textContent = message || 'Connecting...';
            break;
        case 'error':
            statusElement.classList.add('status-item', 'error');
            statusText.textContent = message || 'Connection Error';
            break;
        case 'disconnected':
        default:
            statusElement.classList.add('status-item', 'disconnected');
            statusText.textContent = message || 'Disconnected';
            break;
    }

    // Dispatch custom event for other components to listen to
    this.dispatchEvent(new CustomEvent('connectionStatusChanged', {
        detail: { status, message }
    }));
}

updateMachineInfo(info) {
    document.getElementById('machine-info').textContent = info;
}

initializeNotifications() {
    // Create notification container if it doesn't exist
    if (!document.getElementById('notification-container')) {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'notification-container';
        document.body.appendChild(container);
    }

    // Listen to WebSocket connection events
    this.webSocketClient.addEventListener('connected', () => {
        this.showNotification('Connected to server', 'success', 3000);
    });

    this.webSocketClient.addEventListener('disconnected', () => {
        this.showNotification('Disconnected from server', 'warning', 5000);
    });

    this.webSocketClient.addEventListener('connectionFailed', (event) => {
        this.showNotification(
            `Connection failed: ${event.detail.message}`,
            'error',
            10000
        );
    });

    this.webSocketClient.addEventListener('error', (event) => {
        this.showNotification('Connection error occurred', 'error', 7000);
    });
}

showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

    // Add slide-in animation
    notification.style.transform = 'translateX(100%)';
    container.appendChild(notification);

    // Trigger animation
    requestAnimationFrame(() => {
        notification.style.transform = 'translateX(0)';
    });

    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => {
                    if (notification.parentElement) {
                        notification.remove();
                    }
                }, 300);
            }
        }, duration);
    }
}

initializeConnectionStatusHandler() {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        statusElement.addEventListener('click', () => {
            // Only allow reconnection when disconnected or in error state
            const state = this.webSocketClient.getConnectionState();
            if (state === 'disconnected' || state === 'error') {
                this.showNotification('Attempting to reconnect...', 'info', 2000);
                this.webSocketClient.reconnectNow();
            }
        });

        // Make it look clickable when appropriate
        statusElement.style.cursor = 'pointer';
    }
}

// Enhanced visualization control methods

setupSVGDefinitions(svg) {
    const defs = svg.append('defs');

    // Arrow marker for transitions
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#3498db');

    // Gradient for state highlighting
    const gradient = defs.append('linearGradient')
        .attr('id', 'activeStateGradient')
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '100%');

    gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#e74c3c')
        .attr('stop-opacity', 0.8);

    gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#c0392b')
        .attr('stop-opacity', 0.9);
}

calculateContentBounds(treeData) {
    const nodes = treeData.descendants();
    if (nodes.length === 0) return;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    nodes.forEach(node => {
        const x = node.y - 50; // Account for node width
        const y = node.x - 20; // Account for node height
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x + 100);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y + 40);
    });

    this.bounds.minX = minX;
    this.bounds.maxX = maxX;
    this.bounds.minY = minY;
    this.bounds.maxY = maxY;
}

initializeMinimap() {
    const minimapContainer = document.querySelector('.minimap');
    if (!minimapContainer) return;

    minimapContainer.innerHTML = '';

    const minimapWidth = 150;
    const minimapHeight = 100;

    const minimapSvg = d3.select(minimapContainer)
        .append('svg')
        .attr('width', minimapWidth)
        .attr('height', minimapHeight)
        .style('border', '1px solid #ddd')
        .style('background', '#f8f9fa');

    // Create minimap content
    this.updateMinimap();
}

updateMinimap() {
    const minimapContainer = document.querySelector('.minimap svg');
    if (!minimapContainer || !this.currentStatechart) return;

    // Clear existing content
    d3.select(minimapContainer).selectAll('*').remove();

    const minimapWidth = 150;
    const minimapHeight = 100;

    // Calculate scale for minimap
    const contentWidth = this.bounds.maxX - this.bounds.minX;
    const contentHeight = this.bounds.maxY - this.bounds.minY;
    const scale = Math.min(minimapWidth / contentWidth, minimapHeight / contentHeight) * 0.8;

    // Create minimap group
    const minimapGroup = d3.select(minimapContainer)
        .append('g')
        .attr('transform', `translate(${minimapWidth / 2 - contentWidth * scale / 2}, ${minimapHeight / 2 - contentHeight * scale / 2})`);

    // Draw simplified states
    if (this.stateNodes) {
        this.stateNodes.each(function (d) {
            minimapGroup.append('rect')
                .attr('x', (d.y - this.bounds.minX) * scale - 2)
                .attr('y', (d.x - this.bounds.minY) * scale - 2)
                .attr('width', 4)
                .attr('height', 4)
                .attr('fill', '#3498db');
        }.bind(this));
    }

    // Draw viewport indicator
    if (this.transform) {
        const viewportX = (-this.transform.x / this.transform.k - this.bounds.minX) * scale;
        const viewportY = (-this.transform.y / this.transform.k - this.bounds.minY) * scale;
        const viewportWidth = (this.bounds.width / this.transform.k) * scale;
        const viewportHeight = (this.bounds.height / this.transform.k) * scale;

        minimapGroup.append('rect')
            .attr('x', viewportX)
            .attr('y', viewportY)
            .attr('width', viewportWidth)
            .attr('height', viewportHeight)
            .attr('fill', 'none')
            .attr('stroke', '#e74c3c')
            .attr('stroke-width', 1);
    }
}

// Zoom and pan controls

zoomIn() {
    if (!this.svg || !this.zoom) return;

    this.svg.transition().duration(300)
        .call(this.zoom.scaleBy, 1.5);
}

zoomOut() {
    if (!this.svg || !this.zoom) return;

    this.svg.transition().duration(300)
        .call(this.zoom.scaleBy, 1 / 1.5);
}

resetZoom() {
    if (!this.svg || !this.zoom) return;

    this.svg.transition().duration(500)
        .call(this.zoom.transform, d3.zoomIdentity);
}

fitToScreen() {
    if (!this.svg || !this.zoom || !this.currentStatechart) return;

    const contentWidth = this.bounds.maxX - this.bounds.minX;
    const contentHeight = this.bounds.maxY - this.bounds.minY;

    if (contentWidth === 0 || contentHeight === 0) return;

    const padding = 50;
    const scale = Math.min(
        (this.bounds.width - padding * 2) / contentWidth,
        (this.bounds.height - padding * 2) / contentHeight
    );

    const centerX = this.bounds.width / 2;
    const centerY = this.bounds.height / 2;
    const contentCenterX = (this.bounds.minX + this.bounds.maxX) / 2;
    const contentCenterY = (this.bounds.minY + this.bounds.maxY) / 2;

    const transform = d3.zoomIdentity
        .translate(centerX - contentCenterX * scale, centerY - contentCenterY * scale)
        .scale(scale);

    this.svg.transition().duration(750)
        .call(this.zoom.transform, transform);
}

setVisualizationTool(tool) {
    // Update active tool button
    document.querySelectorAll('.tool-button').forEach(btn => {
        btn.classList.remove('active');
    });

    const activeBtn = document.querySelector(`[data-tool="${tool}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }

    // Configure interaction mode
    if (!this.svg) return;

    switch (tool) {
        case 'select':
            this.svg.style('cursor', 'default');
            // Enable node selection
            break;
        case 'pan':
            this.svg.style('cursor', 'grab');
            // Pan mode is handled by zoom behavior
            break;
        case 'zoom':
            this.svg.style('cursor', 'zoom-in');
            // Zoom mode is handled by zoom behavior
            break;
        case 'add-state':
            this.svg.style('cursor', 'crosshair');
            // Handle state addition
            break;
        case 'add-transition':
            this.svg.style('cursor', 'crosshair');
            // Handle transition addition
            break;
    }
}

exportVisualization() {
    if (!this.svg) return;

    try {
        // Get SVG element
        const svgElement = this.svg.node();

        // Create a copy for export
        const svgClone = svgElement.cloneNode(true);

        // Convert to string
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgClone);

        // Create download blob
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        // Create download link
        const link = document.createElement('a');
        link.href = url;
        link.download = `statechart-${Date.now()}.svg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Cleanup
        URL.revokeObjectURL(url);

        console.log('Visualization exported successfully');
    } catch (error) {
        console.error('Failed to export visualization:', error);
        alert('Failed to export visualization: ' + error.message);
    }
}
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const visualizer = new StatechartVisualizer();

    // Make visualizer available globally for WebSocket integration
    window.visualizer = visualizer;

    // Listen for WebSocket status updates
    window.addEventListener('websocketStatusUpdate', (event) => {
        const { status, attempts, clientId } = event.detail;
        let message = null;

        if (status === 'connecting' && attempts > 0) {
            message = `Reconnecting... (attempt ${attempts})`;
        } else if (status === 'connected' && clientId) {
            message = `Connected (${clientId.slice(-8)})`;
        } else if (status === 'error') {
            message = `Connection error (${attempts} attempts)`;
        }

        visualizer.updateConnectionStatus(status, message);
    });

    console.log('✅ Statechart Visualizer initialized with WebSocket integration');
});