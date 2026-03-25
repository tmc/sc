import { create } from 'zustand';

export interface SimulationStep {
    stepIndex: number;
    activeStateIds: string[];
    event: string | null;
    timestamp: number;
}

interface SimulationState {
    activeStateIds: string[];
    history: SimulationStep[];
    stepIndex: number;
    isRunning: boolean;

    // Actions
    startSimulation: () => void;
    stopSimulation: () => void;
    setActiveStates: (ids: string[]) => void;
    recordStep: (activeStateIds: string[], event: string | null) => void;
    stepBack: () => void;
    stepForward: () => void;
    reset: () => void;
}

export const useSimulationStore = create<SimulationState>((set, get) => ({
    activeStateIds: [],
    history: [],
    stepIndex: -1,
    isRunning: false,

    startSimulation: () => set({ isRunning: true, stepIndex: -1, history: [], activeStateIds: [] }),
    stopSimulation: () => set({ isRunning: false, activeStateIds: [] }),

    setActiveStates: (activeStateIds) => set({ activeStateIds }),

    recordStep: (activeStateIds, event) => {
        const { history, stepIndex } = get();
        // Truncate future history if we were in the past
        const newHistory = history.slice(0, stepIndex + 1);

        const step: SimulationStep = {
            stepIndex: newHistory.length,
            activeStateIds,
            event,
            timestamp: Date.now(),
        };

        set({
            history: [...newHistory, step],
            stepIndex: step.stepIndex,
            activeStateIds
        });
    },

    stepBack: () => {
        const { stepIndex, history } = get();
        if (stepIndex > 0) {
            const prevStep = history[stepIndex - 1];
            set({
                stepIndex: stepIndex - 1,
                activeStateIds: prevStep.activeStateIds
            });
        } else if (stepIndex === 0) {
            // Initial state
            set({ stepIndex: -1, activeStateIds: [] });
        }
    },

    stepForward: () => {
        const { stepIndex, history } = get();
        if (stepIndex < history.length - 1) {
            const nextStep = history[stepIndex + 1];
            set({
                stepIndex: stepIndex + 1,
                activeStateIds: nextStep.activeStateIds
            });
        }
    },

    reset: () => set({ activeStateIds: [], history: [], stepIndex: -1 })
}));
