import type { SimulationStep } from '../store';

export class HistoryTracker {
    private history: SimulationStep[] = [];

    addStep(step: SimulationStep) {
        this.history.push(step);
    }

    getHistory() {
        return this.history;
    }

    clear() {
        this.history = [];
    }
}
