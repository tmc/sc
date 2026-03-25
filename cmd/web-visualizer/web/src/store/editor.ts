import { create } from 'zustand';
import {
    type Node,
    type Edge,
    type OnNodesChange,
    type OnEdgesChange,
    applyNodeChanges,
    applyEdgeChanges,
    addEdge,
    type Connection,
} from '@xyflow/react';

export interface EditorSnapshot {
    nodes: Node[];
    edges: Edge[];
    timestamp: number;
}


export type EditorTool = 'select' | 'pan' | 'node' | 'text';

interface EditorState {
    nodes: Node[];
    edges: Edge[];
    selection: string[];
    mode: 'editing' | 'simulation';
    activeTool: EditorTool;

    // React Flow integration
    onNodesChange: OnNodesChange;
    onEdgesChange: OnEdgesChange;
    onConnect: (connection: Connection) => void;

    // Actions
    setNodes: (nodes: Node[]) => void;
    setEdges: (edges: Edge[]) => void;
    setSelection: (ids: string[]) => void;
    setMode: (mode: 'editing' | 'simulation') => void;
    setActiveTool: (tool: EditorTool) => void;
    updateNodeData: (id: string, data: any) => void;

    // Undo/Redo (Basic implementation)
    undoStack: EditorSnapshot[];
    redoStack: EditorSnapshot[];
    pushSnapshot: () => void;
    undo: () => void;
    redo: () => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
    nodes: [],
    edges: [],
    selection: [],
    mode: 'editing',
    activeTool: 'select',
    undoStack: [],
    redoStack: [],

    onNodesChange: (changes) => {
        // If adding nodes, we might want to intercept clicks, but for now standard behavior
        set({
            nodes: applyNodeChanges(changes, get().nodes),
        });
    },

    onEdgesChange: (changes) => {
        set({
            edges: applyEdgeChanges(changes, get().edges),
        });
    },

    onConnect: (connection) => {
        get().pushSnapshot();
        set({
            edges: addEdge(connection, get().edges),
        });
    },

    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),
    setSelection: (selection) => set({ selection }),
    setMode: (mode) => set({ mode }),
    setActiveTool: (activeTool) => set({ activeTool }),
    updateNodeData: (id, data) => {
        set((state) => ({
            nodes: state.nodes.map((node) =>
                node.id === id ? { ...node, data: { ...node.data, ...data } } : node
            ),
        }));
    },

    pushSnapshot: () => {
        const { nodes, edges } = get();
        // Limit stack size to 50
        const newStack = [...get().undoStack, { nodes, edges, timestamp: Date.now() }].slice(-50);
        set({ undoStack: newStack, redoStack: [] });
    },

    undo: () => {
        const { undoStack, nodes, edges } = get();
        if (undoStack.length === 0) return;

        const snapshot = undoStack[undoStack.length - 1];
        const newUndoStack = undoStack.slice(0, -1);

        set({
            undoStack: newUndoStack,
            redoStack: [...get().redoStack, { nodes, edges, timestamp: Date.now() }],
            nodes: snapshot.nodes,
            edges: snapshot.edges
        });
    },

    redo: () => {
        const { redoStack, nodes, edges } = get();
        if (redoStack.length === 0) return;

        const snapshot = redoStack[redoStack.length - 1];
        const newRedoStack = redoStack.slice(0, -1);

        set({
            redoStack: newRedoStack,
            undoStack: [...get().undoStack, { nodes, edges, timestamp: Date.now() }],
            nodes: snapshot.nodes,
            edges: snapshot.edges
        });
    }
}));
