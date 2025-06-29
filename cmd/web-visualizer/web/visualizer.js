// Import the new tab system components
import stateManager from './state-manager.js';
import TabSystem from './tab-system.js';
import KeyboardShortcuts from './keyboard-shortcuts.js';
import webSocketClient from './websocket-client.js';

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
        this.loadExamples();
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
            root_state: {
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
        
        try {
            const response = await fetch(`/api/v1/machines/${currentMachine.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentMachine),
            });
            
            if (response.ok) {
                console.log('Machine saved successfully');
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
            const response = await fetch('/api/v1/examples');
            this.examples = await response.json();
            
            const selectElement = document.getElementById('example-select');
            selectElement.innerHTML = '<option value="">Select an example...</option>';
            
            Object.keys(this.examples).forEach(key => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = key.charAt(0).toUpperCase() + key.slice(1) + ' Statechart';
                selectElement.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load examples:', error);
        }
    }

    async loadSelectedExample() {
        const selectedExample = document.getElementById('example-select').value;
        if (!selectedExample || !this.examples[selectedExample]) {
            return;
        }

        try {
            const statechart = this.examples[selectedExample];
            const machineId = `example-${selectedExample}-${Date.now()}`;
            
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
        
        // Create SVG with zoom/pan support
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('cursor', 'default');

        // Setup zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.transform = event.transform;
                this.mainGroup.attr('transform', this.transform);
                this.updateMinimap();
            });

        svg.call(this.zoom);

        // Add definitions for markers and patterns
        this.setupSVGDefinitions(svg);

        // Create main group for all content
        this.mainGroup = svg.append('g')
            .attr('class', 'main-group');

        // Create hierarchical layout
        const root = this.createHierarchy(statechart.root_state);
        const treeLayout = d3.tree().size([width - 200, height - 200]);
        const treeData = treeLayout(root);
        
        // Calculate content bounds
        this.calculateContentBounds(treeData);
        
        // Draw states and transitions
        this.drawStates(this.mainGroup, treeData);
        this.drawTransitions(this.mainGroup, statechart.transitions, treeData);
        
        // Initialize minimap
        this.initializeMinimap();
        
        // Create event buttons
        this.createEventButtons(statechart.events);
        
        // Fit to screen initially
        this.fitToScreen();
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
            .attr('class', 'state-group')
            .attr('transform', d => `translate(${d.y}, ${d.x})`)
            .style('cursor', 'pointer');

        // Draw state rectangles with enhanced styling
        nodes.append('rect')
            .attr('class', d => {
                let classes = 'state-node';
                if (d.data.type === 3) classes += ' parallel'; // STATE_TYPE_PARALLEL
                if (d.data.is_initial) classes += ' initial';
                if (d.data.is_final) classes += ' final';
                return classes;
            })
            .attr('x', -40)
            .attr('y', -15)
            .attr('width', 80)
            .attr('height', 30)
            .attr('rx', 5)
            .attr('fill', d => {
                if (d.data.is_final) return '#e74c3c';
                if (d.data.is_initial) return '#2ecc71';
                if (d.data.type === 3) return '#f39c12'; // Parallel states
                return '#3498db';
            })
            .attr('stroke', '#2c3e50')
            .attr('stroke-width', 1)
            .on('mouseover', function(event, d) {
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
                    ${d.data.is_initial ? 'Initial State<br/>' : ''}
                    ${d.data.is_final ? 'Final State<br/>' : ''}
                    Children: ${d.children ? d.children.length : 0}
                `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', function(event, d) {
                d3.select(this)
                    .attr('stroke-width', 1)
                    .attr('stroke', '#2c3e50');
                
                d3.selectAll('.tooltip').remove();
            })
            .on('click', function(event, d) {
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
        nodes.filter(d => d.data.is_initial)
            .append('circle')
            .attr('cx', -50)
            .attr('cy', 0)
            .attr('r', 3)
            .attr('fill', '#2ecc71');

        nodes.filter(d => d.data.is_final)
            .append('circle')
            .attr('cx', 50)
            .attr('cy', 0)
            .attr('r', 3)
            .attr('fill', '#e74c3c');

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
                        .on('mouseover', function() {
                            d3.select(this)
                                .attr('stroke', '#e74c3c')
                                .attr('stroke-width', 3);
                        })
                        .on('mouseout', function() {
                            d3.select(this)
                                .attr('stroke', '#3498db')
                                .attr('stroke-width', 2);
                        })
                        .on('click', function(event) {
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

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
        statusElement.className = status;
    }

    updateMachineInfo(info) {
        document.getElementById('machine-info').textContent = info;
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
            .attr('transform', `translate(${minimapWidth/2 - contentWidth*scale/2}, ${minimapHeight/2 - contentHeight*scale/2})`);

        // Draw simplified states
        if (this.stateNodes) {
            this.stateNodes.each(function(d) {
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
    new StatechartVisualizer();
});