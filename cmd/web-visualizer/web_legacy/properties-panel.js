/**
 * PropertiesPanel - Context-aware property editing sidebar
 *
 * Features:
 * - State property editing (label, type, initial, final, actions)
 * - Transition property editing (event, guard, actions, targets)
 * - Real-time validation feedback
 * - Tag management with autocomplete
 * - Entry/exit action editors
 */

class PropertiesPanel extends EventTarget {
    constructor(options = {}) {
        super();

        // Dependencies
        this.container = options.container;
        this.stateManager = options.stateManager;
        this.orchestrator = options.orchestrator;
        this.commandManager = options.commandManager;

        // State
        this.selectedItem = null; // { type: 'state' | 'transition', data: object }
        this.validationErrors = [];
        this.debounceTimer = null;
        this.debounceDelay = 150;

        // State type labels
        this.stateTypes = {
            1: { label: 'Basic', description: 'Atomic state with no children' },
            2: { label: 'Compound', description: 'OR-state with child states (XOR semantics)' },
            3: { label: 'Parallel', description: 'AND-state with concurrent regions' }
        };

        this.initialize();
    }

    initialize() {
        if (!this.container) return;
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="properties-panel">
                <div class="properties-header">
                    <h3 class="properties-title">Properties</h3>
                    <button class="properties-close" title="Close panel">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                        </svg>
                    </button>
                </div>
                <div class="properties-content">
                    <div class="properties-empty">
                        <p>Select a state or transition to view its properties</p>
                    </div>
                </div>
                <div class="properties-validation"></div>
            </div>
        `;

        this.contentElement = this.container.querySelector('.properties-content');
        this.validationElement = this.container.querySelector('.properties-validation');
    }

    setupEventListeners() {
        // Close button
        const closeBtn = this.container.querySelector('.properties-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.close());
        }

        // Listen for orchestrator selection changes
        if (this.orchestrator) {
            this.orchestrator.addEventListener('selectionChanged', (e) => {
                const selection = e.detail.selection;
                if (selection.length === 1) {
                    this.showProperties(selection[0]);
                } else if (selection.length === 0) {
                    this.clearSelection();
                } else {
                    this.showMultipleSelection(selection);
                }
            });
        }
    }

    showProperties(label) {
        const machine = this.stateManager?.getCurrentMachine();
        if (!machine?.statechart) return;

        // Try to find as state first
        const state = this.findState(machine.statechart.root_state, label);
        if (state) {
            this.selectedItem = { type: 'state', data: state, label };
            this.renderStateProperties(state);
            return;
        }

        // Try to find as transition
        const transition = machine.statechart.transitions?.find(t =>
            t.label === label || `${t.from[0]}_${t.to[0]}_${t.event}` === label
        );
        if (transition) {
            this.selectedItem = { type: 'transition', data: transition, label };
            this.renderTransitionProperties(transition);
            return;
        }

        this.clearSelection();
    }

    findState(state, label) {
        if (!state) return null;
        if (state.label === label) return state;
        if (state.children) {
            for (const child of state.children) {
                const found = this.findState(child, label);
                if (found) return found;
            }
        }
        return null;
    }

    renderStateProperties(state) {
        const typeInfo = this.stateTypes[state.type] || this.stateTypes[1];
        const isRoot = state.label === '__root__';

        this.contentElement.innerHTML = `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <rect x="3" y="3" width="18" height="18" rx="3" fill="none" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </span>
                    <span>State</span>
                </div>

                <div class="property-group">
                    <label class="property-label" for="state-label">Label</label>
                    <input type="text" id="state-label" class="property-input"
                           value="${this.escapeHtml(state.label)}"
                           ${isRoot ? 'disabled' : ''}>
                    <span class="property-hint">Unique identifier for this state</span>
                </div>

                <div class="property-group">
                    <label class="property-label" for="state-type">Type</label>
                    <select id="state-type" class="property-select" ${isRoot ? 'disabled' : ''}>
                        ${Object.entries(this.stateTypes).map(([value, info]) => `
                            <option value="${value}" ${state.type == value ? 'selected' : ''}>
                                ${info.label}
                            </option>
                        `).join('')}
                    </select>
                    <span class="property-hint">${typeInfo.description}</span>
                </div>

                ${!isRoot ? `
                <div class="property-group property-row">
                    <label class="property-checkbox">
                        <input type="checkbox" id="state-initial" ${state.is_initial ? 'checked' : ''}>
                        <span>Initial State</span>
                    </label>
                    <label class="property-checkbox">
                        <input type="checkbox" id="state-final" ${state.is_final ? 'checked' : ''}>
                        <span>Final State</span>
                    </label>
                </div>
                ` : ''}
            </div>

            <div class="property-section">
                <div class="property-section-header collapsible" data-section="actions">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M8 5v14l11-7z" fill="currentColor"/>
                        </svg>
                    </span>
                    <span>Actions</span>
                    <span class="property-section-toggle">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z"/>
                        </svg>
                    </span>
                </div>

                <div class="property-section-content" data-section="actions">
                    <div class="property-group">
                        <label class="property-label">Entry Actions</label>
                        <div class="action-list" id="entry-actions">
                            ${this.renderActionList(state.entry_actions || [], 'entry')}
                        </div>
                        <button class="property-button-small" data-action="add-entry-action">
                            + Add Entry Action
                        </button>
                    </div>

                    <div class="property-group">
                        <label class="property-label">Exit Actions</label>
                        <div class="action-list" id="exit-actions">
                            ${this.renderActionList(state.exit_actions || [], 'exit')}
                        </div>
                        <button class="property-button-small" data-action="add-exit-action">
                            + Add Exit Action
                        </button>
                    </div>
                </div>
            </div>

            ${state.type === 3 ? `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M4 4h7v7H4V4zm9 0h7v7h-7V4zm-9 9h7v7H4v-7zm9 0h7v7h-7v-7z" fill="none" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </span>
                    <span>Parallel Regions</span>
                </div>
                <div class="property-group">
                    <p class="property-hint">
                        This state has ${state.children?.length || 0} parallel regions.
                        Each region executes concurrently.
                    </p>
                </div>
            </div>
            ` : ''}

            <div class="property-section">
                <div class="property-section-header collapsible" data-section="metadata">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" fill="currentColor"/>
                        </svg>
                    </span>
                    <span>Metadata</span>
                    <span class="property-section-toggle">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z"/>
                        </svg>
                    </span>
                </div>

                <div class="property-section-content collapsed" data-section="metadata">
                    <div class="property-group">
                        <label class="property-label" for="state-description">Description</label>
                        <textarea id="state-description" class="property-textarea"
                                  rows="3" placeholder="Optional description...">${this.escapeHtml(state.description || '')}</textarea>
                    </div>
                </div>
            </div>

            <div class="property-actions">
                <button class="property-button danger" data-action="delete" ${isRoot ? 'disabled' : ''}>
                    Delete State
                </button>
            </div>
        `;

        this.setupStateEventHandlers(state);
    }

    renderTransitionProperties(transition) {
        this.contentElement.innerHTML = `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M4 12h16m-4-4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </span>
                    <span>Transition</span>
                </div>

                <div class="property-group">
                    <label class="property-label">From</label>
                    <div class="property-tags readonly">
                        ${transition.from.map(s => `<span class="property-tag">${this.escapeHtml(s)}</span>`).join('')}
                    </div>
                </div>

                <div class="property-group">
                    <label class="property-label">To</label>
                    <div class="property-tags readonly">
                        ${transition.to.map(s => `<span class="property-tag">${this.escapeHtml(s)}</span>`).join('')}
                    </div>
                </div>

                <div class="property-group">
                    <label class="property-label" for="transition-event">Event</label>
                    <input type="text" id="transition-event" class="property-input"
                           value="${this.escapeHtml(transition.event || '')}"
                           placeholder="EVENT_NAME">
                    <span class="property-hint">The event that triggers this transition</span>
                </div>

                <div class="property-group">
                    <label class="property-label" for="transition-label">Label</label>
                    <input type="text" id="transition-label" class="property-input"
                           value="${this.escapeHtml(transition.label || '')}"
                           placeholder="Optional label">
                </div>
            </div>

            <div class="property-section">
                <div class="property-section-header collapsible" data-section="guard">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z" fill="none" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </span>
                    <span>Guard Condition</span>
                    <span class="property-section-toggle">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z"/>
                        </svg>
                    </span>
                </div>

                <div class="property-section-content" data-section="guard">
                    <div class="property-group">
                        <label class="property-label" for="guard-type">Type</label>
                        <select id="guard-type" class="property-select">
                            <option value="1" ${(!transition.guard?.type || transition.guard?.type == 1) ? 'selected' : ''}>RAW</option>
                            <option value="2" ${transition.guard?.type == 2 ? 'selected' : ''}>CEL</option>
                            <option value="3" ${transition.guard?.type == 3 ? 'selected' : ''}>STARLARK</option>
                        </select>
                    </div>
                    <div class="property-group">
                        <label class="property-label" for="guard-expression">Expression</label>
                        <input type="text" id="guard-expression" class="property-input code"
                               value="${this.escapeHtml(transition.guard?.expression || '')}"
                               placeholder="context.value > 0">
                        <span class="property-hint">Boolean expression that must be true for transition</span>
                    </div>
                </div>
            </div>

            <div class="property-section">
                <div class="property-section-header collapsible" data-section="actions">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M8 5v14l11-7z" fill="currentColor"/>
                        </svg>
                    </span>
                    <span>Actions</span>
                    <span class="property-section-toggle">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z"/>
                        </svg>
                    </span>
                </div>

                <div class="property-section-content" data-section="actions">
                    <div class="property-group">
                        <div class="action-list" id="transition-actions">
                            ${this.renderActionList(transition.actions || [], 'transition')}
                        </div>
                        <button class="property-button-small" data-action="add-transition-action">
                            + Add Action
                        </button>
                    </div>
                </div>
            </div>

            <div class="property-actions">
                <button class="property-button danger" data-action="delete">
                    Delete Transition
                </button>
            </div>
        `;

        this.setupTransitionEventHandlers(transition);
        this.setupCollapsibleSections();
    }

    renderActionList(actions, prefix) {
        if (!actions || actions.length === 0) {
            return '<p class="property-hint">No actions defined</p>';
        }

        return actions.map((action, index) => `
            <div class="action-item" data-index="${index}">
                <input type="text" class="property-input code"
                       value="${this.escapeHtml(action.label || action.expression || '')}"
                       data-field="action-${prefix}-${index}">
                <button class="action-remove" data-action="remove-action" data-prefix="${prefix}" data-index="${index}">
                    <svg viewBox="0 0 24 24" width="14" height="14">
                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                    </svg>
                </button>
            </div>
        `).join('');
    }

    showMultipleSelection(labels) {
        this.selectedItem = { type: 'multiple', labels };

        this.contentElement.innerHTML = `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-icon">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M4 6h16M4 12h16M4 18h16" fill="none" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </span>
                    <span>Multiple Selection</span>
                </div>

                <div class="property-group">
                    <p class="property-value">${labels.length} items selected</p>
                    <div class="property-tags">
                        ${labels.map(l => `<span class="property-tag">${this.escapeHtml(l)}</span>`).join('')}
                    </div>
                </div>
            </div>

            <div class="property-actions">
                <button class="property-button" data-action="align-horizontal">
                    Align Horizontal
                </button>
                <button class="property-button" data-action="align-vertical">
                    Align Vertical
                </button>
                <button class="property-button danger" data-action="delete">
                    Delete Selected
                </button>
            </div>
        `;

        this.setupMultipleSelectionHandlers(labels);
    }

    clearSelection() {
        this.selectedItem = null;
        this.contentElement.innerHTML = `
            <div class="properties-empty">
                <p>Select a state or transition to view its properties</p>
            </div>
        `;
        this.validationElement.innerHTML = '';
    }

    setupStateEventHandlers(state) {
        // Label change
        const labelInput = this.container.querySelector('#state-label');
        if (labelInput) {
            labelInput.addEventListener('input', () => {
                this.debouncedUpdate(() => {
                    this.updateStateProperty(state.label, 'label', labelInput.value);
                });
            });
        }

        // Type change
        const typeSelect = this.container.querySelector('#state-type');
        if (typeSelect) {
            typeSelect.addEventListener('change', () => {
                this.updateStateProperty(state.label, 'type', parseInt(typeSelect.value));
            });
        }

        // Initial checkbox
        const initialCheckbox = this.container.querySelector('#state-initial');
        if (initialCheckbox) {
            initialCheckbox.addEventListener('change', () => {
                this.updateStateProperty(state.label, 'is_initial', initialCheckbox.checked);
            });
        }

        // Final checkbox
        const finalCheckbox = this.container.querySelector('#state-final');
        if (finalCheckbox) {
            finalCheckbox.addEventListener('change', () => {
                this.updateStateProperty(state.label, 'is_final', finalCheckbox.checked);
            });
        }

        // Description
        const descriptionInput = this.container.querySelector('#state-description');
        if (descriptionInput) {
            descriptionInput.addEventListener('input', () => {
                this.debouncedUpdate(() => {
                    this.updateStateProperty(state.label, 'description', descriptionInput.value);
                });
            });
        }

        // Delete button
        const deleteBtn = this.container.querySelector('[data-action="delete"]');
        if (deleteBtn && state.label !== '__root__') {
            deleteBtn.addEventListener('click', () => {
                if (confirm(`Delete state "${state.label}"?`)) {
                    this.orchestrator?.deleteSelection();
                }
            });
        }

        this.setupCollapsibleSections();
    }

    setupTransitionEventHandlers(transition) {
        // Event change
        const eventInput = this.container.querySelector('#transition-event');
        if (eventInput) {
            eventInput.addEventListener('input', () => {
                this.debouncedUpdate(() => {
                    this.updateTransitionProperty(transition.label, 'event', eventInput.value);
                });
            });
        }

        // Label change
        const labelInput = this.container.querySelector('#transition-label');
        if (labelInput) {
            labelInput.addEventListener('input', () => {
                this.debouncedUpdate(() => {
                    this.updateTransitionProperty(transition.label, 'label', labelInput.value);
                });
            });
        }

        // Guard expression
        // Guard expression
        const guardInput = this.container.querySelector('#guard-expression');
        const guardType = this.container.querySelector('#guard-type');

        if (guardInput) {
            guardInput.addEventListener('input', () => {
                this.debouncedUpdate(() => {
                    this.updateTransitionProperty(transition.label, 'guard', {
                        expression: guardInput.value,
                        type: parseInt(guardType ? guardType.value : 1)
                    });
                });
            });
        }

        if (guardType) {
            guardType.addEventListener('change', () => {
                this.debouncedUpdate(() => {
                    this.updateTransitionProperty(transition.label, 'guard', {
                        expression: guardInput ? guardInput.value : '',
                        type: parseInt(guardType.value)
                    });
                });
            });
        }

        // Delete button
        const deleteBtn = this.container.querySelector('[data-action="delete"]');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                if (confirm('Delete this transition?')) {
                    this.orchestrator?.deleteSelection();
                }
            });
        }
    }

    setupMultipleSelectionHandlers(labels) {
        const deleteBtn = this.container.querySelector('[data-action="delete"]');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                if (confirm(`Delete ${labels.length} selected items?`)) {
                    this.orchestrator?.deleteSelection();
                }
            });
        }

        // Alignment buttons could trigger batch commands
        const alignHBtn = this.container.querySelector('[data-action="align-horizontal"]');
        if (alignHBtn) {
            alignHBtn.addEventListener('click', () => {
                this.alignSelection('horizontal', labels);
            });
        }

        const alignVBtn = this.container.querySelector('[data-action="align-vertical"]');
        if (alignVBtn) {
            alignVBtn.addEventListener('click', () => {
                this.alignSelection('vertical', labels);
            });
        }
    }

    setupCollapsibleSections() {
        const headers = this.container.querySelectorAll('.property-section-header.collapsible');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const section = header.dataset.section;
                const content = this.container.querySelector(`.property-section-content[data-section="${section}"]`);
                if (content) {
                    content.classList.toggle('collapsed');
                    header.classList.toggle('collapsed');
                }
            });
        });
    }

    debouncedUpdate(fn) {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(fn, this.debounceDelay);
    }

    updateStateProperty(label, property, value) {
        if (!this.commandManager || !this.stateManager) return;

        this.commandManager.execute(
            new UpdateStateCommand(label, { [property]: value }, this.stateManager)
        );

        this.dispatchEvent(new CustomEvent('propertyChanged', {
            detail: { type: 'state', label, property, value }
        }));
    }

    updateTransitionProperty(label, property, value) {
        if (!this.commandManager || !this.stateManager) return;

        this.commandManager.execute(
            new UpdateTransitionCommand(label, { [property]: value }, this.stateManager)
        );

        this.dispatchEvent(new CustomEvent('propertyChanged', {
            detail: { type: 'transition', label, property, value }
        }));
    }

    alignSelection(direction, labels) {
        // Get positions of all selected states
        const machine = this.stateManager?.getCurrentMachine();
        if (!machine?.statechart) return;

        const positions = [];
        labels.forEach(label => {
            const state = this.findState(machine.statechart.root_state, label);
            if (state && state.x !== undefined) {
                positions.push({ label, x: state.x, y: state.y });
            }
        });

        if (positions.length < 2) return;

        // Calculate alignment target
        let targetValue;
        if (direction === 'horizontal') {
            // Align to average Y
            targetValue = positions.reduce((sum, p) => sum + p.y, 0) / positions.length;
        } else {
            // Align to average X
            targetValue = positions.reduce((sum, p) => sum + p.x, 0) / positions.length;
        }

        // Execute as batch
        this.commandManager.beginBatch(`Align ${direction}`);
        positions.forEach(pos => {
            const newPos = direction === 'horizontal'
                ? { x: pos.x, y: targetValue }
                : { x: targetValue, y: pos.y };

            this.commandManager.execute(
                new MoveStateCommand(pos.label, { x: pos.x, y: pos.y }, newPos, this.stateManager)
            );
        });
        this.commandManager.endBatch();
    }

    showValidation(errors) {
        this.validationErrors = errors;

        if (!errors || errors.length === 0) {
            this.validationElement.innerHTML = '';
            return;
        }

        this.validationElement.innerHTML = `
            <div class="validation-errors">
                ${errors.map(err => `
                    <div class="validation-error ${err.severity || 'error'}">
                        <span class="validation-icon">
                            <svg viewBox="0 0 24 24" width="14" height="14">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>
                            </svg>
                        </span>
                        <span class="validation-message">${this.escapeHtml(err.message)}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    close() {
        this.container.classList.remove('open');
        this.dispatchEvent(new CustomEvent('close'));
    }

    open() {
        this.container.classList.add('open');
        this.dispatchEvent(new CustomEvent('open'));
    }

    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Refresh the panel with current selection
    refresh() {
        if (this.selectedItem) {
            if (this.selectedItem.type === 'state') {
                this.showProperties(this.selectedItem.label);
            } else if (this.selectedItem.type === 'transition') {
                this.showProperties(this.selectedItem.label);
            } else if (this.selectedItem.type === 'multiple') {
                this.showMultipleSelection(this.selectedItem.labels);
            }
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PropertiesPanel };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.PropertiesPanel = PropertiesPanel;
}
