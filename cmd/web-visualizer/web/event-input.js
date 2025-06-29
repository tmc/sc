/**
 * Event Input System for Statechart Testing
 * 
 * Provides interactive UI for sending events to statechart machines
 * and observing state transitions in real-time.
 */

class EventInputSystem {
    constructor() {
        this.currentMachine = null;
        this.eventHistory = [];
        this.suggestions = new Set();
        this.isConnected = false;
        
        this.initializeUI();
        this.setupEventListeners();
    }

    initializeUI() {
        // Create event input panel
        this.createEventPanel();
        this.createEventHistory();
        this.createEventSuggestions();
    }

    createEventPanel() {
        const eventPanel = document.createElement('div');
        eventPanel.className = 'event-panel';
        eventPanel.innerHTML = `
            <div class="event-panel-header">
                <h3>Event Testing</h3>
                <div class="connection-status">
                    <span class="status-indicator disconnected"></span>
                    <span class="status-text">Disconnected</span>
                </div>
            </div>
            
            <div class="event-input-section">
                <div class="input-group">
                    <label for="event-name">Event Name:</label>
                    <input type="text" id="event-name" placeholder="Enter event name..." 
                           autocomplete="off" spellcheck="false">
                    <button id="send-event" disabled>Send Event</button>
                </div>
                
                <div class="event-data-section">
                    <label for="event-data">Event Data (JSON):</label>
                    <textarea id="event-data" placeholder='{"key": "value"}' rows="3"></textarea>
                </div>
                
                <div class="event-actions">
                    <button id="reset-machine" disabled>Reset Machine</button>
                    <button id="clear-history">Clear History</button>
                </div>
            </div>
            
            <div class="event-suggestions">
                <div class="suggestions-header">
                    <span>Suggested Events:</span>
                </div>
                <div class="suggestions-list"></div>
            </div>
        `;

        // Find or create container for event panel
        let container = document.querySelector('.event-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'event-container';
            
            // Insert into sidebar or create new section
            const sidebar = document.querySelector('.sidebar') || document.querySelector('.panel-left');
            if (sidebar) {
                sidebar.appendChild(container);
            } else {
                document.body.appendChild(container);
            }
        }
        
        container.appendChild(eventPanel);
        this.eventPanel = eventPanel;
    }

    createEventHistory() {
        const historyPanel = document.createElement('div');
        historyPanel.className = 'event-history-panel';
        historyPanel.innerHTML = `
            <div class="history-header">
                <h4>Event History</h4>
                <span class="history-count">0 events</span>
            </div>
            <div class="history-list"></div>
        `;
        
        this.eventPanel.appendChild(historyPanel);
        this.historyPanel = historyPanel;
    }

    createEventSuggestions() {
        this.suggestionsContainer = this.eventPanel.querySelector('.suggestions-list');
        this.updateSuggestions([]);
    }

    setupEventListeners() {
        // Event input handling
        const eventInput = this.eventPanel.querySelector('#event-name');
        const sendButton = this.eventPanel.querySelector('#send-event');
        const resetButton = this.eventPanel.querySelector('#reset-machine');
        const clearButton = this.eventPanel.querySelector('#clear-history');
        const eventDataInput = this.eventPanel.querySelector('#event-data');

        // Send event on button click
        sendButton.addEventListener('click', () => this.sendEvent());
        
        // Send event on Enter key
        eventInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !sendButton.disabled) {
                this.sendEvent();
            }
        });

        // Enable/disable send button based on input
        eventInput.addEventListener('input', () => {
            const hasEvent = eventInput.value.trim().length > 0;
            sendButton.disabled = !hasEvent || !this.isConnected;
        });

        // Reset machine
        resetButton.addEventListener('click', () => this.resetMachine());

        // Clear history
        clearButton.addEventListener('click', () => this.clearHistory());

        // Auto-complete functionality
        eventInput.addEventListener('input', (e) => {
            this.handleAutoComplete(e.target.value);
        });

        // Validate JSON in event data
        eventDataInput.addEventListener('input', () => {
            this.validateEventData();
        });
    }

    setMachine(machine) {
        this.currentMachine = machine;
        this.isConnected = !!machine;
        
        this.updateConnectionStatus();
        this.updateButtons();
        this.extractEventSuggestions();
    }

    updateConnectionStatus() {
        const indicator = this.eventPanel.querySelector('.status-indicator');
        const statusText = this.eventPanel.querySelector('.status-text');
        
        if (this.isConnected) {
            indicator.className = 'status-indicator connected';
            statusText.textContent = `Connected: ${this.currentMachine?.id || 'Unknown'}`;
        } else {
            indicator.className = 'status-indicator disconnected';
            statusText.textContent = 'Disconnected';
        }
    }

    updateButtons() {
        const sendButton = this.eventPanel.querySelector('#send-event');
        const resetButton = this.eventPanel.querySelector('#reset-machine');
        const eventInput = this.eventPanel.querySelector('#event-name');
        
        const hasEvent = eventInput.value.trim().length > 0;
        sendButton.disabled = !hasEvent || !this.isConnected;
        resetButton.disabled = !this.isConnected;
    }

    extractEventSuggestions() {
        if (!this.currentMachine?.statechart) return;

        const events = new Set();
        
        // Extract events from transitions
        if (this.currentMachine.statechart.transitions) {
            this.currentMachine.statechart.transitions.forEach(transition => {
                if (transition.event && transition.event !== 'τ') {
                    events.add(transition.event);
                }
            });
        }

        // Extract events from machine definition
        if (this.currentMachine.statechart.events) {
            this.currentMachine.statechart.events.forEach(event => {
                events.add(event.label || event.name);
            });
        }

        this.suggestions = events;
        this.updateSuggestions(Array.from(events));
    }

    updateSuggestions(events) {
        if (!this.suggestionsContainer) return;

        this.suggestionsContainer.innerHTML = '';
        
        events.forEach(event => {
            const suggestion = document.createElement('button');
            suggestion.className = 'event-suggestion';
            suggestion.textContent = event;
            suggestion.onclick = () => this.selectSuggestion(event);
            this.suggestionsContainer.appendChild(suggestion);
        });

        // Show/hide suggestions section
        const suggestionsSection = this.eventPanel.querySelector('.event-suggestions');
        suggestionsSection.style.display = events.length > 0 ? 'block' : 'none';
    }

    selectSuggestion(event) {
        const eventInput = this.eventPanel.querySelector('#event-name');
        eventInput.value = event;
        eventInput.focus();
        this.updateButtons();
    }

    handleAutoComplete(value) {
        if (!value || value.length < 1) {
            this.updateSuggestions(Array.from(this.suggestions));
            return;
        }

        const filtered = Array.from(this.suggestions).filter(event =>
            event.toLowerCase().includes(value.toLowerCase())
        );
        
        this.updateSuggestions(filtered);
    }

    validateEventData() {
        const eventDataInput = this.eventPanel.querySelector('#event-data');
        const data = eventDataInput.value.trim();
        
        if (!data) {
            eventDataInput.classList.remove('error');
            return true;
        }

        try {
            JSON.parse(data);
            eventDataInput.classList.remove('error');
            return true;
        } catch (error) {
            eventDataInput.classList.add('error');
            return false;
        }
    }

    async sendEvent() {
        if (!this.currentMachine) return;

        const eventInput = this.eventPanel.querySelector('#event-name');
        const eventDataInput = this.eventPanel.querySelector('#event-data');
        
        const eventName = eventInput.value.trim();
        if (!eventName) return;

        // Validate event data
        if (!this.validateEventData()) {
            alert('Invalid JSON in event data');
            return;
        }

        let eventData = {};
        if (eventDataInput.value.trim()) {
            try {
                eventData = JSON.parse(eventDataInput.value);
            } catch (error) {
                alert('Invalid JSON in event data');
                return;
            }
        }

        try {
            // Send event to machine via API
            const response = await fetch(`/api/v1/machines/${this.currentMachine.id}/events`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    event: eventName,
                    data: eventData
                }),
            });

            if (response.ok) {
                const result = await response.json();
                this.addEventToHistory(eventName, eventData, result, true);
                
                // Clear input after successful send
                eventInput.value = '';
                eventDataInput.value = '';
                this.updateButtons();
                
                // Trigger UI update if needed
                if (window.stateManager && window.stateManager.updateMachineState) {
                    window.stateManager.updateMachineState(result);
                }
                
            } else {
                const error = await response.text();
                this.addEventToHistory(eventName, eventData, { error }, false);
                alert(`Failed to send event: ${error}`);
            }
            
        } catch (error) {
            this.addEventToHistory(eventName, eventData, { error: error.message }, false);
            alert(`Network error: ${error.message}`);
        }
    }

    async resetMachine() {
        if (!this.currentMachine) return;

        try {
            const response = await fetch(`/api/v1/machines/${this.currentMachine.id}/reset`, {
                method: 'POST',
            });

            if (response.ok) {
                const result = await response.json();
                this.addEventToHistory('RESET', {}, result, true);
                
                // Trigger UI update
                if (window.stateManager && window.stateManager.updateMachineState) {
                    window.stateManager.updateMachineState(result);
                }
                
            } else {
                const error = await response.text();
                alert(`Failed to reset machine: ${error}`);
            }
            
        } catch (error) {
            alert(`Network error: ${error.message}`);
        }
    }

    addEventToHistory(eventName, eventData, result, success) {
        const historyItem = {
            timestamp: new Date(),
            event: eventName,
            data: eventData,
            result: result,
            success: success
        };

        this.eventHistory.unshift(historyItem);
        
        // Limit history size
        if (this.eventHistory.length > 100) {
            this.eventHistory = this.eventHistory.slice(0, 100);
        }

        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyList = this.historyPanel.querySelector('.history-list');
        const historyCount = this.historyPanel.querySelector('.history-count');
        
        historyCount.textContent = `${this.eventHistory.length} events`;
        
        historyList.innerHTML = '';
        
        this.eventHistory.slice(0, 20).forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = `history-item ${item.success ? 'success' : 'error'}`;
            
            const time = item.timestamp.toLocaleTimeString();
            const dataStr = Object.keys(item.data).length > 0 ? 
                ` (${JSON.stringify(item.data)})` : '';
            
            historyItem.innerHTML = `
                <div class="history-event">
                    <span class="event-name">${item.event}</span>
                    <span class="event-data">${dataStr}</span>
                </div>
                <div class="history-meta">
                    <span class="history-time">${time}</span>
                    <span class="history-status">${item.success ? '✓' : '✗'}</span>
                </div>
            `;
            
            if (item.result && item.result.configuration) {
                const configDiv = document.createElement('div');
                configDiv.className = 'history-config';
                configDiv.textContent = `→ ${item.result.configuration.join(', ')}`;
                historyItem.appendChild(configDiv);
            }
            
            historyList.appendChild(historyItem);
        });
    }

    clearHistory() {
        this.eventHistory = [];
        this.updateHistoryDisplay();
    }

    // Public API for integration
    getEventHistory() {
        return [...this.eventHistory];
    }

    addSuggestion(event) {
        this.suggestions.add(event);
        this.updateSuggestions(Array.from(this.suggestions));
    }

    setConnectionStatus(connected, machineId = null) {
        this.isConnected = connected;
        if (connected && machineId) {
            this.currentMachine = { id: machineId };
        } else {
            this.currentMachine = null;
        }
        this.updateConnectionStatus();
        this.updateButtons();
    }
}

// Export for global access
if (typeof window !== 'undefined') {
    window.EventInputSystem = EventInputSystem;
}