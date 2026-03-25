import { create } from 'zustand';

export type NodeType = 'atomic' | 'compound' | 'parallel' | 'final' | 'history';

export interface Point {
    x: number;
    y: number;
}

export interface Size {
    width: number;
    height: number;
}

export interface FlowNode {
    id: string;
    label: string;
    type: NodeType;
    position: Point;
    size: Size;
    children?: string[]; // IDs of children
    parentId?: string;
}

export interface FlowEdge {
    id: string;
    source: string;
    target: string;
    label?: string;
    waypoints: Point[];
}

interface AppState {
    nodes: FlowNode[];
    edges: FlowEdge[];
    selection: string[]; // IDs of selected nodes/edges
    activeStateIDs: string[]; // IDs of currently active states (for simulation)

    // Actions
    addNode: (node: FlowNode) => void;
    updateNode: (id: string, updates: Partial<FlowNode>) => void;
    selectNode: (id: string, toggle: boolean) => void;
    clearSelection: () => void;
    setActiveStates: (ids: string[]) => void;

    // Mock Generation
    generateMachine: (prompt: string) => Promise<void>;
}

export const useAppStore = create<AppState>((set) => ({
    nodes: [],
    edges: [],
    selection: [],
    activeStateIDs: [],

    addNode: (node: FlowNode) => set((state: AppState) => ({ nodes: [...state.nodes, node] })),

    updateNode: (id: string, updates: Partial<FlowNode>) => set((state: AppState) => ({
        nodes: state.nodes.map((n) => (n.id === id ? { ...n, ...updates } : n)),
    })),

    selectNode: (id: string, toggle: boolean) => set((state: AppState) => {
        if (toggle) {
            return {
                selection: state.selection.includes(id)
                    ? state.selection.filter((s: string) => s !== id)
                    : [...state.selection, id]
            };
        }
        return { selection: [id] };
    }),

    clearSelection: () => set({ selection: [] }),

    setActiveStates: (ids: string[]) => set({ activeStateIDs: ids }),

    generateMachine: async (prompt: string) => {
        // Mock simulation of AI generation delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Simple mock logic for Traffic Light
        if (prompt.toLowerCase().includes("traffic")) {
            set({
                nodes: [
                    { id: 'red', label: 'Red', type: 'atomic', position: { x: 100, y: 100 }, size: { width: 120, height: 60 } },
                    { id: 'yellow', label: 'Yellow', type: 'atomic', position: { x: 300, y: 100 }, size: { width: 120, height: 60 } },
                    { id: 'green', label: 'Green', type: 'atomic', position: { x: 500, y: 100 }, size: { width: 120, height: 60 } },
                ],
                edges: [
                    { id: 'e1', source: 'red', target: 'green', waypoints: [] },
                    { id: 'e2', source: 'green', target: 'yellow', waypoints: [] },
                    { id: 'e3', source: 'yellow', target: 'red', waypoints: [] },
                ]
            });
        }
    }
}));
