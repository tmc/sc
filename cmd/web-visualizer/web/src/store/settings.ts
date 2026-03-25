import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsState {
    theme: 'light' | 'dark' | 'system';
    showGrid: boolean;
    showMinimap: boolean;
    snapToGrid: boolean;

    // Actions
    setTheme: (theme: 'light' | 'dark' | 'system') => void;
    toggleGrid: () => void;
    toggleMinimap: () => void;
    toggleSnapToGrid: () => void;
}

export const useSettingsStore = create<SettingsState>()(
    persist(
        (set) => ({
            theme: 'system',
            showGrid: true,
            showMinimap: true,
            snapToGrid: true,

            setTheme: (theme) => set({ theme }),
            toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),
            toggleMinimap: () => set((state) => ({ showMinimap: !state.showMinimap })),
            toggleSnapToGrid: () => set((state) => ({ snapToGrid: !state.snapToGrid })),
        }),
        { name: 'sc-settings' }
    )
);
