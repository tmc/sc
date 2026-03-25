import { GuardEvaluator } from './GuardEvaluator';

export interface Transition {
    source: string;
    target: string;
    event?: string;
    guard?: string;
    action?: string;
}

export interface StateDef {
    id: string;
    type: 'atomic' | 'compound' | 'parallel' | 'history' | 'final';
    parent?: string;
    initial?: string;
    children?: string[];
    onEntry?: string;
    onExit?: string;
}

// Simplified Engine for the visualizer
export class StatechartEngine {
    private states: Map<string, StateDef>;
    private transitions: Transition[];

    // Runtime state
    private activeStateIds: Set<string>;
    private context: Record<string, unknown>;

    constructor(states: StateDef[], transitions: Transition[], initialContext: Record<string, unknown> = {}) {
        this.states = new Map(states.map(s => [s.id, s]));
        this.transitions = transitions;
        this.activeStateIds = new Set();
        this.context = { ...initialContext };
    }

    start(): string[] {
        // Find initial state(s)
        const rootNodes = Array.from(this.states.values()).filter(s => !s.parent);
        // const initialStates: string[] = [];

        // Simple initialization: activate initial child if compound, or self if atomic
        // This is a naive implementation; full SCXML algorithm is complex.
        const activate = (stateId: string) => {
            const state = this.states.get(stateId);
            if (!state) return;

            this.activeStateIds.add(stateId);

            if (state.type === 'compound' && state.initial) {
                activate(state.initial);
            } else if (state.type === 'parallel' && state.children) {
                state.children.forEach(child => activate(child));
            }
        };

        rootNodes.forEach(node => {
            // Assume single root for now or parallel roots
            if (node.initial) {
                activate(node.initial);
            } else {
                activate(node.id);
            }
        });

        return Array.from(this.activeStateIds);
    }

    send(event: string): { activeStates: string[]; transitionTaken: Transition | null } {
        // Find first transition that matches event and guard
        // Prioritize active states (deepest first traditionally)

        // Sort active states by depth (not implemented here, assuming simple iteration)
        const active = Array.from(this.activeStateIds);
        let taken: Transition | null = null;
        let statesExited: string[] = [];
        let statesEntered: string[] = [];

        for (const stateId of active) {
            const relevantTransitions = this.transitions.filter(t => t.source === stateId && t.event === event);

            for (const t of relevantTransitions) {
                if (!t.guard || GuardEvaluator.evaluate(t.guard, this.context)) {
                    taken = t;
                    break;
                }
            }

            if (taken) break;
        }

        if (taken) {
            // Execute Transition
            this.activeStateIds.delete(taken.source);
            statesExited.push(taken.source);

            // Enter target
            // Logic should handle LCA (Least Common Ancestor) etc.
            // Simplified: simple switching
            this.activeStateIds.add(taken.target);
            statesEntered.push(taken.target);

            // Handle 'initial' of target if compound
            const targetState = this.states.get(taken.target);
            if (targetState && targetState.type === 'compound' && targetState.initial) {
                this.activeStateIds.add(targetState.initial);
                statesEntered.push(targetState.initial);
            }

            // Apply actions (simulated)
            if (taken.action) {
                console.log(`Executing action: ${taken.action}`);
                // Could update context via eval
            }
        }

        return {
            activeStates: Array.from(this.activeStateIds),
            transitionTaken: taken
        };
    }

    getContext() {
        return this.context;
    }
}
