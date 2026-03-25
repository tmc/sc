/**
 * TraceHistory - Displays execution trace history for simulation mode
 *
 * Features:
 * - Scrollable list of execution steps
 * - Each step shows event, states entered/exited
 * - Click to jump to a specific step
 * - Current step highlighting
 */

class TraceHistory extends EventTarget {
    constructor(options = {}) {
        super();

        // Dependencies
        this.container = options.container || document.getElementById('step-history');
        this.stateManager = options.stateManager;
        this.simulationMode = options.simulationMode;
        this.apiBaseUrl = options.apiBaseUrl || '/api/v1';

        // State
        this.machineId = null;
        this.steps = [];
        this.currentIndex = -1;

        this.initialize();
    }

    initialize() {
        if (this.simulationMode) {
            this.simulationMode.addEventListener('stepChanged', (e) => {
                this.currentIndex = e.detail.stepIndex;
                this.highlightCurrentStep();
            });

            this.simulationMode.addEventListener('eventProcessed', (e) => {
                if (e.detail.transitioned && e.detail.lastStep) {
                    this.addStep(e.detail.lastStep);
                }
            });

            this.simulationMode.addEventListener('machineReset', () => {
                this.clear();
            });
        }
    }

    setMachine(machineId) {
        this.machineId = machineId;
        this.steps = [];
        this.currentIndex = -1;
        this.render();
    }

    // Load step history from API
    async loadHistory() {
        if (!this.machineId) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/machines/${this.machineId}`);
            if (!response.ok) throw new Error('Failed to fetch machine');

            const data = await response.json();
            this.steps = data.step_history || [];
            this.currentIndex = this.steps.length - 1;
            this.render();
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    addStep(step) {
        this.steps.push(step);
        this.currentIndex = this.steps.length - 1;
        this.appendStep(step, this.currentIndex);
        this.highlightCurrentStep();
        this.scrollToBottom();
    }

    clear() {
        this.steps = [];
        this.currentIndex = -1;
        this.render();
    }

    render() {
        if (!this.container) return;

        if (this.steps.length === 0) {
            this.container.innerHTML = `
                <div class="trace-empty">
                    <p>No execution history yet</p>
                    <p class="trace-hint">Send events to see the execution trace</p>
                </div>
            `;
            return;
        }

        this.container.innerHTML = `
            <div class="trace-history">
                ${this.steps.map((step, index) => this.renderStep(step, index)).join('')}
            </div>
        `;

        this.setupClickHandlers();
        this.highlightCurrentStep();
    }

    appendStep(step, index) {
        let historyContainer = this.container.querySelector('.trace-history');

        if (!historyContainer) {
            this.container.innerHTML = '<div class="trace-history"></div>';
            historyContainer = this.container.querySelector('.trace-history');
        }

        historyContainer.insertAdjacentHTML('beforeend', this.renderStep(step, index));
        this.setupClickHandlers();
    }

    renderStep(step, index) {
        const events = step.events || [];
        const eventLabel = events.length > 0 ? events[0].label || events[0] : 'initial';

        // Get states entered and exited
        const statesEntered = this.getStatesFromConfiguration(step.resulting_configuration);
        const statesExited = this.getStatesFromConfiguration(step.starting_configuration);

        // Calculate what changed
        const entered = statesEntered.filter(s => !statesExited.includes(s));
        const exited = statesExited.filter(s => !statesEntered.includes(s));

        return `
            <div class="trace-step ${index === this.currentIndex ? 'current' : ''}" data-index="${index}">
                <div class="trace-step-header">
                    <span class="trace-step-index">#${index + 1}</span>
                    <span class="trace-step-event">${this.escapeHtml(eventLabel)}</span>
                </div>
                <div class="trace-step-states">
                    ${exited.map(s => `<span class="trace-state-badge exited">- ${this.escapeHtml(s)}</span>`).join('')}
                    ${entered.map(s => `<span class="trace-state-badge entered">+ ${this.escapeHtml(s)}</span>`).join('')}
                </div>
            </div>
        `;
    }

    getStatesFromConfiguration(configuration) {
        if (!configuration) return [];
        if (configuration.states) {
            return configuration.states.map(s => s.label || s);
        }
        return [];
    }

    setupClickHandlers() {
        const steps = this.container.querySelectorAll('.trace-step');
        steps.forEach(stepEl => {
            stepEl.addEventListener('click', () => {
                const index = parseInt(stepEl.dataset.index);
                this.jumpToStep(index);
            });
        });
    }

    async jumpToStep(index) {
        if (index === this.currentIndex || !this.machineId) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/machines/${this.machineId}/restore/${index}`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to restore to step');

            const data = await response.json();
            this.currentIndex = data.step_index;

            // Truncate steps array to current index
            this.steps = this.steps.slice(0, this.currentIndex + 1);

            this.highlightCurrentStep();

            this.dispatchEvent(new CustomEvent('stepJumped', {
                detail: {
                    index: this.currentIndex,
                    configuration: data.configuration
                }
            }));
        } catch (error) {
            console.error('Error jumping to step:', error);
        }
    }

    highlightCurrentStep() {
        const steps = this.container.querySelectorAll('.trace-step');
        steps.forEach((stepEl, index) => {
            stepEl.classList.toggle('current', index === this.currentIndex);
        });
    }

    scrollToBottom() {
        const historyContainer = this.container.querySelector('.trace-history');
        if (historyContainer) {
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }
    }

    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TraceHistory };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.TraceHistory = TraceHistory;
}
