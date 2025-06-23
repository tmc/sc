class StatechartVisualizer {
    constructor() {
        this.currentMachine = null;
        this.examples = {};
        this.websocket = null;
        this.svg = null;
        this.simulation = null;
        
        this.initializeUI();
        this.connectWebSocket();
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

        // Update status
        this.updateConnectionStatus('disconnected');
        this.updateMachineInfo('No machine loaded');
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            this.updateConnectionStatus('connected');
            console.log('WebSocket connected');
        };
        
        this.websocket.onclose = () => {
            this.updateConnectionStatus('disconnected');
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('WebSocket message:', data);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    async loadExamples() {
        try {
            const response = await fetch('/api/examples');
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
            
            const response = await fetch('/api/machines', {
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
                this.updateMachineDisplay();
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
            const response = await fetch(`/api/machines/${this.currentMachine.id}/reset`, {
                method: 'POST',
            });

            if (response.ok) {
                const machine = await response.json();
                this.currentMachine = machine;
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
            const response = await fetch(`/api/machines/${this.currentMachine.id}/events`, {
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
                
                this.updateMachineDisplay();
                this.addToHistory(eventName, result.last_step);
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
        const container = document.getElementById('statechart-container');
        container.innerHTML = '';
        
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Add arrow marker definition
        svg.append('defs').append('marker')
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

        this.svg = svg;
        
        // Create hierarchical layout
        const root = this.createHierarchy(statechart.root_state);
        const treeLayout = d3.tree().size([width - 100, height - 100]);
        const treeData = treeLayout(root);
        
        // Draw states
        const stateGroup = svg.append('g')
            .attr('transform', 'translate(50, 50)');
            
        this.drawStates(stateGroup, treeData);
        this.drawTransitions(stateGroup, statechart.transitions, treeData);
        
        // Create event buttons
        this.createEventButtons(statechart.events);
    }

    createHierarchy(state) {
        const hierarchy = d3.hierarchy(state, (d) => d.children);
        return hierarchy;
    }

    drawStates(container, treeData) {
        const nodes = container.selectAll('.state-group')
            .data(treeData.descendants())
            .enter()
            .append('g')
            .attr('class', 'state-group')
            .attr('transform', d => `translate(${d.y}, ${d.x})`);

        // Draw state rectangles
        nodes.append('rect')
            .attr('class', d => {
                let classes = 'state-node';
                if (d.data.type === 3) classes += ' parallel'; // STATE_TYPE_PARALLEL
                return classes;
            })
            .attr('x', -40)
            .attr('y', -15)
            .attr('width', 80)
            .attr('height', 30)
            .attr('rx', 5);

        // Add state labels
        nodes.append('text')
            .attr('class', 'state-label')
            .text(d => d.data.label);

        // Store nodes for later highlighting
        this.stateNodes = nodes;
    }

    drawTransitions(container, transitions, treeData) {
        if (!transitions) return;

        const nodeMap = new Map();
        treeData.descendants().forEach(d => {
            nodeMap.set(d.data.label, d);
        });

        transitions.forEach(transition => {
            if (transition.from && transition.to && transition.from.length > 0 && transition.to.length > 0) {
                const fromNode = nodeMap.get(transition.from[0]);
                const toNode = nodeMap.get(transition.to[0]);
                
                if (fromNode && toNode) {
                    // Draw transition line
                    container.append('path')
                        .attr('class', 'transition-line')
                        .attr('d', this.createTransitionPath(fromNode, toNode));
                    
                    // Add transition label
                    const midX = (fromNode.y + toNode.y) / 2;
                    const midY = (fromNode.x + toNode.x) / 2;
                    
                    container.append('text')
                        .attr('class', 'transition-label')
                        .attr('x', midX)
                        .attr('y', midY - 5)
                        .text(transition.event || transition.label);
                }
            }
        });
    }

    createTransitionPath(fromNode, toNode) {
        const x1 = fromNode.y + 40; // Right edge of from node
        const y1 = fromNode.x;
        const x2 = toNode.y - 40;   // Left edge of to node
        const y2 = toNode.x;
        
        return `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${y1} ${x2} ${y2}`;
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
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new StatechartVisualizer();
});