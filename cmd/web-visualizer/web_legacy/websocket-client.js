/**
 * WebSocket Client for Real-Time Synchronization
 * 
 * Manages WebSocket connections, handles real-time updates,
 * implements conflict resolution, and provides collaboration features.
 */
import stateManager from './state-manager.js';

class WebSocketClient extends EventTarget {
    constructor() {
        super();

        this.ws = null;
        this.clientId = null;
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, error
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds
        this.reconnectTimer = null;
        this.heartbeatInterval = null;
        this.heartbeatIntervalMs = 30000; // 30 seconds

        // Message queue for when offline
        this.messageQueue = [];
        this.maxQueueSize = 100;

        // Subscriptions
        this.subscribedMachines = new Set();

        // Conflict resolution
        this.pendingOperations = new Map();
        this.operationId = 0;

        this.initialize();
    }

    initialize() {
        this.connect();
        this.bindStateManagerEvents();
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.connectionState = 'connecting';
        this.updateConnectionStatus();

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            this.ws = new WebSocket(wsUrl);
            this.setupWebSocketEvents();
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.handleConnectionError(error);
        }
    }

    setupWebSocketEvents() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.connectionState = 'connected';
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;

            this.updateConnectionStatus();
            this.startHeartbeat();
            this.processMessageQueue();
            this.resubscribeToMachines();

            this.dispatchEvent(new CustomEvent('connected', { detail: event }));
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error, event.data);
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.connectionState = 'disconnected';
            this.updateConnectionStatus();
            this.stopHeartbeat();

            this.dispatchEvent(new CustomEvent('disconnected', { detail: event }));

            // Attempt to reconnect if not a clean close
            if (event.code !== 1000) {
                this.scheduleReconnect();
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError(error);
        };
    }

    handleMessage(message) {
        const { type, machine_id, client_id, data, timestamp } = message;

        switch (type) {
            case 'welcome':
                this.handleWelcome(data, client_id);
                break;

            case 'machine_state':
                this.handleMachineStateUpdate(machine_id, data);
                break;

            case 'event_processed':
                this.handleEventProcessed(machine_id, data);
                break;

            case 'machine_updated':
                this.handleMachineUpdated(machine_id, data);
                break;

            case 'operation_ack':
                this.handleOperationAck(data);
                break;

            case 'operation_conflict':
                this.handleOperationConflict(data);
                break;

            case 'pong':
                // Heartbeat response
                break;

            default:
                console.warn('Unknown WebSocket message type:', type, message);
        }

        // Emit generic message event
        this.dispatchEvent(new CustomEvent('message', { detail: message }));
    }

    handleWelcome(data, clientId) {
        this.clientId = clientId;
        console.log(`WebSocket client ID: ${clientId}`);

        if (data && data.connected_clients) {
            console.log(`Connected clients: ${data.connected_clients}`);
        }

        this.dispatchEvent(new CustomEvent('welcome', {
            detail: { clientId, data }
        }));

        // Update connection status with valid client ID
        this.updateConnectionStatus();
    }

    handleMachineStateUpdate(machineId, data) {
        if (!machineId || !data) return;

        const machine = stateManager.getCurrentMachine();
        if (machine && machine.id === machineId) {
            // Update current machine state
            stateManager.updateMachine(machineId, {
                configuration: data.configuration,
                context: data.context,
                state: data.state
            });
        }

        this.dispatchEvent(new CustomEvent('machineStateUpdate', {
            detail: { machineId, data }
        }));
    }

    handleEventProcessed(machineId, data) {
        if (!machineId || !data) return;

        const machine = stateManager.getCurrentMachine();
        if (machine && machine.id === machineId) {
            // Update machine state and add to history
            stateManager.updateMachine(machineId, {
                configuration: data.configuration,
                context: data.context
            });

            stateManager.addEventToHistory(data.event, data.last_step);
        }

        this.dispatchEvent(new CustomEvent('eventProcessed', {
            detail: { machineId, data }
        }));
    }

    handleMachineUpdated(machineId, data) {
        if (!machineId || !data) return;

        const { action } = data;

        switch (action) {
            case 'created':
                // Machine was created by another client
                console.log(`Machine ${machineId} created remotely`);
                break;

            case 'updated':
                // Machine was updated by another client
                stateManager.updateMachine(machineId, {
                    configuration: data.configuration,
                    context: data.context,
                    state: data.state
                });
                break;

            case 'deleted':
                // Machine was deleted by another client
                stateManager.removeMachine(machineId);
                break;

            case 'reset':
                // Machine was reset by another client
                stateManager.updateMachine(machineId, {
                    configuration: data.configuration,
                    context: data.context
                });
                break;
        }

        this.dispatchEvent(new CustomEvent('machineUpdated', {
            detail: { machineId, action, data }
        }));
    }

    handleOperationAck(data) {
        const { operation_id } = data;
        if (this.pendingOperations.has(operation_id)) {
            const operation = this.pendingOperations.get(operation_id);
            this.pendingOperations.delete(operation_id);

            if (operation.resolve) {
                operation.resolve(data);
            }
        }
    }

    handleOperationConflict(data) {
        const { operation_id, conflict_reason } = data;
        if (this.pendingOperations.has(operation_id)) {
            const operation = this.pendingOperations.get(operation_id);
            this.pendingOperations.delete(operation_id);

            if (operation.reject) {
                operation.reject(new Error(`Operation conflict: ${conflict_reason}`));
            }
        }

        // Show conflict resolution UI
        this.showConflictResolution(data);
    }

    showConflictResolution(conflictData) {
        // Future: Implement conflict resolution UI
        console.warn('Conflict detected:', conflictData);

        this.dispatchEvent(new CustomEvent('conflict', {
            detail: conflictData
        }));
    }

    // Public API Methods

    subscribe(machineId) {
        if (!machineId) return;

        this.subscribedMachines.add(machineId);

        if (this.isConnected()) {
            this.send({
                type: 'subscribe',
                machine_id: machineId,
                data: machineId
            });
        }
    }

    unsubscribe(machineId) {
        if (!machineId) return;

        this.subscribedMachines.delete(machineId);

        if (this.isConnected()) {
            this.send({
                type: 'unsubscribe',
                machine_id: machineId,
                data: machineId
            });
        }
    }

    sendEvent(machineId, event, data = null) {
        return this.sendOperation({
            type: 'process_event',
            machine_id: machineId,
            data: { event, data }
        });
    }

    updateMachine(machineId, updates) {
        return this.sendOperation({
            type: 'update_machine',
            machine_id: machineId,
            data: updates
        });
    }

    resetMachine(machineId) {
        return this.sendOperation({
            type: 'reset_machine',
            machine_id: machineId
        });
    }

    sendOperation(operation) {
        const operationId = ++this.operationId;
        const message = {
            ...operation,
            operation_id: operationId,
            timestamp: new Date().toISOString()
        };

        return new Promise((resolve, reject) => {
            this.pendingOperations.set(operationId, { resolve, reject });

            // Set timeout for operation
            setTimeout(() => {
                if (this.pendingOperations.has(operationId)) {
                    this.pendingOperations.delete(operationId);
                    reject(new Error('Operation timeout'));
                }
            }, 10000); // 10 second timeout

            this.send(message);
        });
    }

    send(message) {
        if (this.isConnected()) {
            try {
                this.ws.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('Failed to send WebSocket message:', error);
                this.queueMessage(message);
                return false;
            }
        } else {
            this.queueMessage(message);
            return false;
        }
    }

    queueMessage(message) {
        if (this.messageQueue.length >= this.maxQueueSize) {
            this.messageQueue.shift(); // Remove oldest message
        }

        this.messageQueue.push({
            message,
            timestamp: Date.now()
        });
    }

    processMessageQueue() {
        const now = Date.now();
        const maxAge = 5 * 60 * 1000; // 5 minutes

        // Filter out old messages
        this.messageQueue = this.messageQueue.filter(item =>
            now - item.timestamp < maxAge
        );

        // Send queued messages
        while (this.messageQueue.length > 0 && this.isConnected()) {
            const { message } = this.messageQueue.shift();
            try {
                this.ws.send(JSON.stringify(message));
            } catch (error) {
                console.error('Failed to send queued message:', error);
                break;
            }
        }
    }

    resubscribeToMachines() {
        this.subscribedMachines.forEach(machineId => {
            this.send({
                type: 'subscribe',
                machine_id: machineId,
                data: machineId
            });
        });
    }

    // Connection Management

    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    getConnectionState() {
        return this.connectionState;
    }

    getClientId() {
        return this.clientId;
    }

    disconnect() {
        // Cancel any pending reconnection
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        this.stopHeartbeat();

        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
        }

        this.connectionState = 'disconnected';
        this.updateConnectionStatus();
    }

    // Manual reconnect method (useful for UI controls)
    reconnectNow() {
        console.log('Manual reconnection requested');

        // Cancel any pending reconnection
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        // Reset attempts for manual reconnection
        this.reconnectAttempts = 0;

        // Close existing connection if any
        if (this.ws) {
            this.ws.close(1000, 'Manual reconnect');
        }

        // Connect immediately
        this.connect();
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.connectionState = 'error';
            this.updateConnectionStatus();

            // Dispatch a connection failed event
            this.dispatchEvent(new CustomEvent('connectionFailed', {
                detail: {
                    attempts: this.reconnectAttempts,
                    message: 'Maximum reconnection attempts reached'
                }
            }));
            return;
        }

        this.reconnectAttempts++;
        // Exponential backoff with jitter to prevent thundering herd
        const baseDelay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        const jitter = Math.random() * 0.3 * baseDelay; // Add up to 30% jitter
        const delay = Math.min(baseDelay + jitter, this.maxReconnectDelay);

        console.log(`Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        // Update status to show reconnecting
        this.connectionState = 'connecting';
        this.updateConnectionStatus();

        // Schedule the reconnection
        this.reconnectTimer = setTimeout(() => {
            if (this.connectionState !== 'connected') {
                console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}...`);
                this.connect();
            }
        }, delay);
    }

    handleConnectionError(error) {
        this.connectionState = 'error';
        this.updateConnectionStatus();
        this.stopHeartbeat();

        this.dispatchEvent(new CustomEvent('error', { detail: error }));
    }

    startHeartbeat() {
        this.stopHeartbeat();

        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected()) {
                this.send({
                    type: 'ping',
                    timestamp: new Date().toISOString()
                });
            }
        }, this.heartbeatIntervalMs);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    updateConnectionStatus() {
        // Update state manager
        stateManager.setConnectionState(
            this.connectionState === 'connected',
            this.connectionState === 'error' ? 'Connection error' : null
        );

        // Update UI status display - find visualizer instance
        const statusUpdateEvent = new CustomEvent('websocketStatusUpdate', {
            detail: {
                status: this.connectionState,
                attempts: this.reconnectAttempts,
                clientId: this.clientId
            }
        });

        // Dispatch to global window for visualizer to catch
        window.dispatchEvent(statusUpdateEvent);

        // Try to update status directly if visualizer is available
        if (window.visualizer && typeof window.visualizer.updateConnectionStatus === 'function') {
            let message = null;
            if (this.connectionState === 'connecting') {
                message = `Reconnecting... (attempt ${this.reconnectAttempts})`;
            } else if (this.connectionState === 'connected') {
                message = this.clientId ? `Connected (${this.clientId})` : `Connected...`;
            } else if (this.connectionState === 'error') {
                message = `Connection error (${this.reconnectAttempts}/${this.maxReconnectAttempts})`;
            }

            window.visualizer.updateConnectionStatus(this.connectionState, message);
        }
    }

    bindStateManagerEvents() {
        // Listen for current machine changes to update subscriptions
        stateManager.addEventListener('currentMachineChanged', (e) => {
            const { machineId } = e.detail;

            // Unsubscribe from previous machine
            const currentMachine = stateManager.getCurrentMachine();
            if (currentMachine && currentMachine.id !== machineId) {
                this.unsubscribe(currentMachine.id);
            }

            // Subscribe to new machine
            if (machineId) {
                this.subscribe(machineId);
            }
        });
    }

    // Debug and diagnostics

    getStats() {
        return {
            connectionState: this.connectionState,
            clientId: this.clientId,
            reconnectAttempts: this.reconnectAttempts,
            queuedMessages: this.messageQueue.length,
            subscribedMachines: Array.from(this.subscribedMachines),
            pendingOperations: this.pendingOperations.size
        };
    }

    destroy() {
        this.stopHeartbeat();

        // Cancel any pending reconnection
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        this.disconnect();
        this.subscribedMachines.clear();
        this.pendingOperations.clear();
        this.messageQueue.length = 0;
    }
}

// Export as singleton
const webSocketClient = new WebSocketClient();
export default webSocketClient;