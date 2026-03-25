/**
 * CommandManager - Implements the Command pattern for undo/redo operations
 *
 * This module provides a robust undo/redo system for the statechart editor,
 * supporting command batching, history limits, and state serialization.
 */

// Command base class - all commands must implement execute() and undo()
class Command {
    constructor(type, description) {
        this.type = type;
        this.description = description;
        this.timestamp = Date.now();
    }

    execute() {
        throw new Error('Command.execute() must be implemented');
    }

    undo() {
        throw new Error('Command.undo() must be implemented');
    }

    // Optional: merge with another command of the same type
    canMerge(other) {
        return false;
    }

    merge(other) {
        return this;
    }
}

// Add State Command
class AddStateCommand extends Command {
    constructor(stateData, parentLabel, stateManager) {
        super('add-state', `Add state "${stateData.label}"`);
        this.stateData = { ...stateData };
        this.parentLabel = parentLabel;
        this.stateManager = stateManager;
    }

    execute() {
        this.stateManager.addState(this.stateData, this.parentLabel, { skipHistory: true });
    }

    undo() {
        this.stateManager.removeState(this.stateData.label, { skipHistory: true });
    }
}

// Remove State Command
class RemoveStateCommand extends Command {
    constructor(stateLabel, stateManager) {
        super('remove-state', `Remove state "${stateLabel}"`);
        this.stateLabel = stateLabel;
        this.stateManager = stateManager;
        this.removedState = null;
        this.parentLabel = null;
        this.removedTransitions = [];
    }

    execute() {
        // Capture state data before removal for undo
        const state = this.stateManager.findState(this.stateLabel);
        if (state) {
            this.removedState = JSON.parse(JSON.stringify(state));
            this.parentLabel = this.stateManager.findParentLabel(this.stateLabel);
            // Capture transitions involving this state
            this.removedTransitions = this.stateManager.getTransitionsForState(this.stateLabel)
                .map(t => JSON.parse(JSON.stringify(t)));
        }
        this.stateManager.removeState(this.stateLabel, { skipHistory: true });
    }

    undo() {
        if (this.removedState) {
            this.stateManager.addState(this.removedState, this.parentLabel, { skipHistory: true });
            // Restore transitions
            this.removedTransitions.forEach(t => {
                this.stateManager.addTransition(t, { skipHistory: true });
            });
        }
    }
}

// Update State Command
class UpdateStateCommand extends Command {
    constructor(stateLabel, changes, stateManager) {
        super('update-state', `Update state "${stateLabel}"`);
        this.stateLabel = stateLabel;
        this.changes = { ...changes };
        this.stateManager = stateManager;
        this.previousValues = null;
    }

    execute() {
        const state = this.stateManager.findState(this.stateLabel);
        if (state) {
            // Capture previous values for undo
            this.previousValues = {};
            Object.keys(this.changes).forEach(key => {
                this.previousValues[key] = state[key];
            });
        }
        this.stateManager.updateState(this.stateLabel, this.changes, { skipHistory: true });
    }

    undo() {
        if (this.previousValues) {
            this.stateManager.updateState(this.stateLabel, this.previousValues, { skipHistory: true });
        }
    }

    canMerge(other) {
        // Merge consecutive updates to the same state within 500ms
        return other instanceof UpdateStateCommand &&
               other.stateLabel === this.stateLabel &&
               (other.timestamp - this.timestamp) < 500;
    }

    merge(other) {
        // Keep original previous values, use latest changes
        this.changes = { ...this.changes, ...other.changes };
        this.timestamp = other.timestamp;
        return this;
    }
}

// Move State Command
class MoveStateCommand extends Command {
    constructor(stateLabel, fromPosition, toPosition, stateManager) {
        super('move-state', `Move state "${stateLabel}"`);
        this.stateLabel = stateLabel;
        this.fromPosition = { ...fromPosition };
        this.toPosition = { ...toPosition };
        this.stateManager = stateManager;
    }

    execute() {
        this.stateManager.updateStatePosition(this.stateLabel, this.toPosition, { skipHistory: true });
    }

    undo() {
        this.stateManager.updateStatePosition(this.stateLabel, this.fromPosition, { skipHistory: true });
    }

    canMerge(other) {
        // Merge consecutive moves of the same state within 200ms (for dragging)
        return other instanceof MoveStateCommand &&
               other.stateLabel === this.stateLabel &&
               (other.timestamp - this.timestamp) < 200;
    }

    merge(other) {
        // Keep original from position, use latest to position
        this.toPosition = { ...other.toPosition };
        this.timestamp = other.timestamp;
        return this;
    }
}

// Reparent State Command
class ReparentStateCommand extends Command {
    constructor(stateLabel, fromParent, toParent, stateManager) {
        super('reparent-state', `Reparent state "${stateLabel}"`);
        this.stateLabel = stateLabel;
        this.fromParent = fromParent;
        this.toParent = toParent;
        this.stateManager = stateManager;
    }

    execute() {
        this.stateManager.reparentState(this.stateLabel, this.toParent, { skipHistory: true });
    }

    undo() {
        this.stateManager.reparentState(this.stateLabel, this.fromParent, { skipHistory: true });
    }
}

// Add Transition Command
class AddTransitionCommand extends Command {
    constructor(transitionData, stateManager) {
        super('add-transition', `Add transition "${transitionData.label || transitionData.event}"`);
        this.transitionData = { ...transitionData };
        this.stateManager = stateManager;
    }

    execute() {
        this.stateManager.addTransition(this.transitionData, { skipHistory: true });
    }

    undo() {
        this.stateManager.removeTransition(this.transitionData.label || this.transitionData.id, { skipHistory: true });
    }
}

// Remove Transition Command
class RemoveTransitionCommand extends Command {
    constructor(transitionId, stateManager) {
        super('remove-transition', `Remove transition "${transitionId}"`);
        this.transitionId = transitionId;
        this.stateManager = stateManager;
        this.removedTransition = null;
    }

    execute() {
        // Capture transition data before removal
        this.removedTransition = this.stateManager.findTransition(this.transitionId);
        if (this.removedTransition) {
            this.removedTransition = JSON.parse(JSON.stringify(this.removedTransition));
        }
        this.stateManager.removeTransition(this.transitionId, { skipHistory: true });
    }

    undo() {
        if (this.removedTransition) {
            this.stateManager.addTransition(this.removedTransition, { skipHistory: true });
        }
    }
}

// Update Transition Command
class UpdateTransitionCommand extends Command {
    constructor(transitionId, changes, stateManager) {
        super('update-transition', `Update transition "${transitionId}"`);
        this.transitionId = transitionId;
        this.changes = { ...changes };
        this.stateManager = stateManager;
        this.previousValues = null;
    }

    execute() {
        const transition = this.stateManager.findTransition(this.transitionId);
        if (transition) {
            this.previousValues = {};
            Object.keys(this.changes).forEach(key => {
                this.previousValues[key] = transition[key];
            });
        }
        this.stateManager.updateTransition(this.transitionId, this.changes, { skipHistory: true });
    }

    undo() {
        if (this.previousValues) {
            this.stateManager.updateTransition(this.transitionId, this.previousValues, { skipHistory: true });
        }
    }
}

// Batch Command - groups multiple commands into a single undoable action
class BatchCommand extends Command {
    constructor(commands, description) {
        super('batch', description || `Batch of ${commands.length} operations`);
        this.commands = commands;
    }

    execute() {
        this.commands.forEach(cmd => cmd.execute());
    }

    undo() {
        // Undo in reverse order
        for (let i = this.commands.length - 1; i >= 0; i--) {
            this.commands[i].undo();
        }
    }
}

// Command Manager
class CommandManager extends EventTarget {
    constructor(options = {}) {
        super();
        this.undoStack = [];
        this.redoStack = [];
        this.maxHistorySize = options.maxHistorySize || 100;
        this.pendingBatch = null;
        this.batchDepth = 0;
    }

    // Execute a command and add to history
    execute(command) {
        command.execute();

        if (this.batchDepth > 0) {
            // Add to pending batch
            this.pendingBatch.push(command);
        } else {
            this.addToHistory(command);
        }

        this.dispatchEvent(new CustomEvent('commandExecuted', { detail: command }));
    }

    addToHistory(command) {
        // Try to merge with the last command
        if (this.undoStack.length > 0) {
            const lastCommand = this.undoStack[this.undoStack.length - 1];
            if (lastCommand.canMerge && lastCommand.canMerge(command)) {
                lastCommand.merge(command);
                this.dispatchEvent(new CustomEvent('historyChanged'));
                return;
            }
        }

        this.undoStack.push(command);
        this.redoStack = []; // Clear redo stack on new command

        // Enforce history limit
        while (this.undoStack.length > this.maxHistorySize) {
            this.undoStack.shift();
        }

        this.dispatchEvent(new CustomEvent('historyChanged'));
    }

    // Start a batch operation
    beginBatch(description) {
        if (this.batchDepth === 0) {
            this.pendingBatch = [];
            this.pendingBatchDescription = description;
        }
        this.batchDepth++;
    }

    // End a batch operation
    endBatch() {
        this.batchDepth--;
        if (this.batchDepth === 0 && this.pendingBatch && this.pendingBatch.length > 0) {
            const batchCommand = new BatchCommand(this.pendingBatch, this.pendingBatchDescription);
            this.addToHistory(batchCommand);
            this.pendingBatch = null;
            this.pendingBatchDescription = null;
        }
    }

    // Cancel the current batch
    cancelBatch() {
        if (this.batchDepth > 0) {
            // Undo all pending commands
            if (this.pendingBatch) {
                for (let i = this.pendingBatch.length - 1; i >= 0; i--) {
                    this.pendingBatch[i].undo();
                }
            }
            this.pendingBatch = null;
            this.pendingBatchDescription = null;
            this.batchDepth = 0;
        }
    }

    // Undo the last command
    undo() {
        if (!this.canUndo()) return null;

        const command = this.undoStack.pop();
        command.undo();
        this.redoStack.push(command);

        this.dispatchEvent(new CustomEvent('undo', { detail: command }));
        this.dispatchEvent(new CustomEvent('historyChanged'));
        return command;
    }

    // Redo the last undone command
    redo() {
        if (!this.canRedo()) return null;

        const command = this.redoStack.pop();
        command.execute();
        this.undoStack.push(command);

        this.dispatchEvent(new CustomEvent('redo', { detail: command }));
        this.dispatchEvent(new CustomEvent('historyChanged'));
        return command;
    }

    canUndo() {
        return this.undoStack.length > 0;
    }

    canRedo() {
        return this.redoStack.length > 0;
    }

    // Get history for display
    getHistory() {
        return {
            undo: this.undoStack.map(cmd => ({
                type: cmd.type,
                description: cmd.description,
                timestamp: cmd.timestamp
            })),
            redo: this.redoStack.map(cmd => ({
                type: cmd.type,
                description: cmd.description,
                timestamp: cmd.timestamp
            }))
        };
    }

    // Clear all history
    clear() {
        this.undoStack = [];
        this.redoStack = [];
        this.pendingBatch = null;
        this.batchDepth = 0;
        this.dispatchEvent(new CustomEvent('historyChanged'));
    }

    // Serialize history for persistence
    serialize() {
        return JSON.stringify({
            undoStack: this.undoStack.map(cmd => ({
                type: cmd.type,
                data: this.serializeCommand(cmd)
            })),
            redoStack: this.redoStack.map(cmd => ({
                type: cmd.type,
                data: this.serializeCommand(cmd)
            }))
        });
    }

    serializeCommand(cmd) {
        // Command-specific serialization
        const data = {
            type: cmd.type,
            description: cmd.description,
            timestamp: cmd.timestamp
        };

        switch (cmd.type) {
            case 'add-state':
                data.stateData = cmd.stateData;
                data.parentLabel = cmd.parentLabel;
                break;
            case 'remove-state':
                data.stateLabel = cmd.stateLabel;
                data.removedState = cmd.removedState;
                data.parentLabel = cmd.parentLabel;
                data.removedTransitions = cmd.removedTransitions;
                break;
            case 'update-state':
                data.stateLabel = cmd.stateLabel;
                data.changes = cmd.changes;
                data.previousValues = cmd.previousValues;
                break;
            case 'move-state':
                data.stateLabel = cmd.stateLabel;
                data.fromPosition = cmd.fromPosition;
                data.toPosition = cmd.toPosition;
                break;
            case 'reparent-state':
                data.stateLabel = cmd.stateLabel;
                data.fromParent = cmd.fromParent;
                data.toParent = cmd.toParent;
                break;
            case 'add-transition':
                data.transitionData = cmd.transitionData;
                break;
            case 'remove-transition':
                data.transitionId = cmd.transitionId;
                data.removedTransition = cmd.removedTransition;
                break;
            case 'update-transition':
                data.transitionId = cmd.transitionId;
                data.changes = cmd.changes;
                data.previousValues = cmd.previousValues;
                break;
            case 'batch':
                data.commands = cmd.commands.map(c => this.serializeCommand(c));
                break;
        }

        return data;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CommandManager,
        Command,
        AddStateCommand,
        RemoveStateCommand,
        UpdateStateCommand,
        MoveStateCommand,
        ReparentStateCommand,
        AddTransitionCommand,
        RemoveTransitionCommand,
        UpdateTransitionCommand,
        BatchCommand
    };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.CommandManager = CommandManager;
    window.Command = Command;
    window.AddStateCommand = AddStateCommand;
    window.RemoveStateCommand = RemoveStateCommand;
    window.UpdateStateCommand = UpdateStateCommand;
    window.MoveStateCommand = MoveStateCommand;
    window.ReparentStateCommand = ReparentStateCommand;
    window.AddTransitionCommand = AddTransitionCommand;
    window.RemoveTransitionCommand = RemoveTransitionCommand;
    window.UpdateTransitionCommand = UpdateTransitionCommand;
    window.BatchCommand = BatchCommand;
}
