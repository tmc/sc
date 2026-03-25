import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';

export interface StatechartWrapper {
    id: string;
    name: string;
    jsonContent: string;
    updatedAt: number;
}

interface MachinesState {
    machines: StatechartWrapper[];
    selectedMachineId: string | null;
    openMachineIds: string[];

    // Actions
    addMachine: (machine: Partial<StatechartWrapper>) => void;
    loadSamples: (samples: { name: string; jsonContent: string }[]) => void;
    deleteMachine: (id: string) => void;
    updateMachine: (id: string, updates: Partial<StatechartWrapper>) => void;
    selectMachine: (id: string) => void;
    openMachine: (id: string) => void;
    closeMachine: (id: string) => void;
}

export const useMachinesStore = create<MachinesState>()(
    persist(
        (set) => ({
            machines: [],
            selectedMachineId: null,
            openMachineIds: [],

            addMachine: (machine) => set((state) => {
                const newMachine: StatechartWrapper = {
                    id: machine.id || uuidv4(),
                    name: machine.name || 'Untitled Machine',
                    jsonContent: machine.jsonContent || '{}',
                    updatedAt: Date.now(),
                };
                return {
                    machines: [...state.machines, newMachine],
                    selectedMachineId: newMachine.id,
                    openMachineIds: [...new Set([...state.openMachineIds, newMachine.id])],
                };
            }),

            loadSamples: (samples) => set((state) => {
                // Only load if empty to prevent duplicates on reload if persisted
                if (state.machines.length > 0) return {};

                const newMachines = samples.map(s => ({
                    id: uuidv4(),
                    name: s.name,
                    jsonContent: s.jsonContent,
                    updatedAt: Date.now(),
                }));

                return {
                    machines: newMachines,
                    selectedMachineId: null,
                    openMachineIds: []
                };
            }),

            deleteMachine: (id) => set((state) => ({
                machines: state.machines.filter((m) => m.id !== id),
                openMachineIds: state.openMachineIds.filter((mid) => mid !== id),
                selectedMachineId: state.selectedMachineId === id ? null : state.selectedMachineId,
            })),

            updateMachine: (id, updates) => set((state) => ({
                machines: state.machines.map((m) =>
                    m.id === id ? { ...m, ...updates, updatedAt: Date.now() } : m
                ),
            })),

            selectMachine: (id) => set({ selectedMachineId: id }),

            openMachine: (id) => set((state) => ({
                openMachineIds: [...new Set([...state.openMachineIds, id])],
                selectedMachineId: id,
            })),

            closeMachine: (id) => set((state) => {
                const newOpenIds = state.openMachineIds.filter((mid) => mid !== id);
                return {
                    openMachineIds: newOpenIds,
                    selectedMachineId:
                        state.selectedMachineId === id
                            ? newOpenIds[newOpenIds.length - 1] || null
                            : state.selectedMachineId,
                };
            }),
        }),
        { name: 'sc-machines' }
    )
);
