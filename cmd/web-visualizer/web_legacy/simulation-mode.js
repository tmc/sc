/**
 * SimulationMode - Controls simulation mode for statechart visualization
 *
 * Features:
 * - Design/Simulation mode toggle
 * - Step forward/back through history
 * - Reset to initial state
 * - Visual highlighting of active states
 * - Enabled transitions display
 */

class SimulationMode extends EventTarget {
    constructor(options = {}) {
        super();

        // Dependencies
        this.stateManager = options.stateManager;
        this.apiBaseUrl = options.apiBaseUrl || '/api/v1';

        // State
        this.mode = 'edit'; // 'edit' | 'simulation'
        this.machineId = null;
        this.currentStepIndex = -1;
        this.totalSteps = 0;
        this.enabledTransitions = [];
        this.configuration = [];

        // UI Elements
        this.modeEditButton = document.getElementById('mode-edit');
        this.modeSimulationButton = document.getElementById('mode-simulation');
        this.simulationToolbar = document.getElementById('simulation-toolbar');
        this.stepBackButton = document.getElementById('step-back-button');
        this.stepForwardButton = document.getElementById('step-forward-button');
        this.resetButton = document.getElementById('reset-simulation-button');
        this.stepCounter = document.getElementById('step-counter');
        this.undoButton = document.getElementById('undo-button');
        this.redoButton = document.getElementById('redo-button');

        this.initialize();
    }

    initialize() {
        this.setupModeToggle();
        this.setupSimulationControls();
    }

    setupModeToggle() {
        if (this.modeEditButton) {
            this.modeEditButton.addEventListener('click', () => this.setMode('edit'));
        }
        if (this.modeSimulationButton) {
            this.modeSimulationButton.addEventListener('click', () => this.setMode('simulation'));
        }
    }

    setupSimulationControls() {
        if (this.stepBackButton) {
            this.stepBackButton.addEventListener('click', () => this.stepBack());
        }
        if (this.stepForwardButton) {
            this.stepForwardButton.addEventListener('click', () => this.showEventSelector());
        }
        if (this.resetButton) {
            this.resetButton.addEventListener('click', () => this.reset());
        }
    }

    setMode(mode) {
        if (this.mode === mode) return;

        const previousMode = this.mode;
        this.mode = mode;

        // Update UI
        this.updateModeUI();

        // Update body data attribute for CSS
        document.body.setAttribute('data-mode', mode);

        // Toggle simulation toolbar visibility
        if (this.simulationToolbar) {
            this.simulationToolbar.style.display = mode === 'simulation' ? 'flex' : 'none';
        }

        // If entering simulation mode, fetch enabled transitions
        if (mode === 'simulation' && this.machineId) {
            this.fetchEnabledTransitions();
        }

        this.dispatchEvent(new CustomEvent('modeChanged', {
            detail: { mode, previousMode }
        }));
    }

    updateModeUI() {
        if (this.modeEditButton) {
            this.modeEditButton.classList.toggle('active', this.mode === 'edit');
        }
        if (this.modeSimulationButton) {
            this.modeSimulationButton.classList.toggle('active', this.mode === 'simulation');
        }

        // Show/hide edit tools
        const editTools = document.querySelector('.edit-tools');
        if (editTools) {
            editTools.style.opacity = this.mode === 'simulation' ? '0.5' : '1';
            editTools.style.pointerEvents = this.mode === 'simulation' ? 'none' : 'auto';
        }
    }

    getMode() {
        return this.mode;
    }

    isSimulationMode() {
        return this.mode === 'simulation';
    }

    setMachine(machineId) {
        this.machineId = machineId;
        this.currentStepIndex = -1;
        this.totalSteps = 0;

        if (this.isSimulationMode()) {
            this.fetchEnabledTransitions();
        }
    }

    async fetchEnabledTransitions() {
        if (!this.machineId) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/machines/${this.machineId}/enabled-transitions`);
            if (!response.ok) throw new Error('Failed to fetch enabled transitions');

            const data = await response.json();
            this.enabledTransitions = data.transitions || [];
            this.configuration = data.configuration || [];

            this.dispatchEvent(new CustomEvent('enabledTransitionsChanged', {
                detail: {
                    transitions: this.enabledTransitions,
                    configuration: this.configuration
                }
            }));
        } catch (error) {
            console.error('Error fetching enabled transitions:', error);
        }
    }

    async stepBack() {
        if (this.currentStepIndex <= 0 || !this.machineId) return;

        const targetIndex = this.currentStepIndex - 1;

        try {
            const response = await fetch(`${this.apiBaseUrl}/machines/${this.machineId}/restore/${targetIndex}`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to restore to step');

            const data = await response.json();
            this.currentStepIndex = data.step_index;
            this.configuration = data.configuration || [];

            this.updateStepCounter();
            this.fetchEnabledTransitions();

            this.dispatchEvent(new CustomEvent('stepChanged', {
                detail: {
                    stepIndex: this.currentStepIndex,
                    configuration: this.configuration
                }
            }));
        } catch (error) {
            console.error('Error stepping back:', error);
        }
    }

    showEventSelector() {
        // Show available events that can be fired
        if (this.enabledTransitions.length === 0) {
            console.log('No enabled transitions available');
            return;
        }

        // Get unique events from enabled transitions
        const events = [...new Set(this.enabledTransitions.map(t => t.event).filter(Boolean))];

        if (events.length === 1) {
            // Only one event available, fire it directly
            this.sendEvent(events[0]);
        } else if (events.length > 1) {
            // Multiple events, show a selector
            this.dispatchEvent(new CustomEvent('showEventSelector', {
                detail: { events, transitions: this.enabledTransitions }
            }));
        }
    }

    async sendEvent(eventName) {
        if (!this.machineId || !eventName) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/machines/${this.machineId}/events`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ event: eventName })
            });

            if (!response.ok) throw new Error('Failed to send event');

            const data = await response.json();

            if (data.transitioned) {
                this.currentStepIndex++;
                this.totalSteps = this.currentStepIndex + 1;
                this.configuration = data.configuration || [];

                this.updateStepCounter();
                this.fetchEnabledTransitions();

                this.dispatchEvent(new CustomEvent('eventProcessed', {
                    detail: {
                        event: eventName,
                        transitioned: true,
                        configuration: this.configuration,
                        lastStep: data.last_step
                    }
                }));
            } else {
                this.dispatchEvent(new CustomEvent('eventProcessed', {
                    detail: {
                        event: eventName,
                        transitioned: false
                    }
                }));
            }
        } catch (error) {
            console.error('Error sending event:', error);
        }
    }

    async reset() {
        if (!this.machineId) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/machines/${this.machineId}/reset`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to reset machine');

            const data = await response.json();
            this.currentStepIndex = 0;
            this.totalSteps = 1;
            this.configuration = data.configuration || [];

            this.updateStepCounter();
            this.fetchEnabledTransitions();

            this.dispatchEvent(new CustomEvent('machineReset', {
                detail: { configuration: this.configuration }
            }));
        } catch (error) {
            console.error('Error resetting machine:', error);
        }
    }

    updateStepCounter() {
        if (this.stepCounter) {
            this.stepCounter.textContent = `Step: ${this.currentStepIndex + 1}`;
        }

        // Update button states
        if (this.stepBackButton) {
            this.stepBackButton.disabled = this.currentStepIndex <= 0;
        }
    }

    // Get active states for visualization highlighting
    getActiveStates() {
        return this.configuration;
    }

    // Get enabled transitions for visualization highlighting
    getEnabledTransitions() {
        return this.enabledTransitions;
    }

    // Update history from step history data
    updateFromStepHistory(stepHistory) {
        this.totalSteps = stepHistory?.length || 0;
        this.currentStepIndex = this.totalSteps - 1;
        this.updateStepCounter();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SimulationMode };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.SimulationMode = SimulationMode;
}
