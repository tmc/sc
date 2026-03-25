import { useEffect, useState } from 'preact/hooks';
import { Sidebar } from './Sidebar';
import { TabBar } from './TabBar';
import { FlowCanvas } from './FlowCanvas';
import { DesignToolbar } from './DesignToolbar';
import { SimulationToolbar } from './SimulationToolbar';
import { useMachinesStore, useEditorStore } from '../store';
import { Menu } from 'lucide-react';

export function EditorPane() {
    const { openMachineIds, selectedMachineId, machines, addMachine } = useMachinesStore();
    const { undo, redo, setNodes, setEdges } = useEditorStore();
    const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

    // Hydrate Editor Store when selected machine changes
    useEffect(() => {
        if (!selectedMachineId) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const machine = machines.find(m => m.id === selectedMachineId);
        if (!machine) return;

        try {
            const data = JSON.parse(machine.jsonContent);
            if (data.nodes && Array.isArray(data.nodes)) {
                setNodes(data.nodes);
                setEdges(data.edges || []);
            }
        } catch (e) {
            console.error("Failed to parse machine content", e);
        }

    }, [selectedMachineId, machines, setNodes, setEdges]);

    // Keyboard Shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            const meta = e.metaKey || e.ctrlKey;

            if (meta && e.key === 'z') {
                e.preventDefault();
                if (e.shiftKey) redo();
                else undo();
            }

            if (meta && e.key === 'n') {
                e.preventDefault();
                addMachine({ name: 'New Machine' });
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [addMachine, undo, redo]);

    return (
        <div className="relative h-screen w-screen bg-midnight-950 text-slate-100 font-sans overflow-hidden selection:bg-neon-blue/30">
            {/* Background Gradient Mesh */}
            <div className="absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-midnight-800 via-midnight-950 to-midnight-950 pointer-events-none" />

            {/* Layer 0: Canvas (Full Screen) */}
            <div className="absolute inset-0 z-0">
                {openMachineIds.length > 0 ? (
                    <FlowCanvas />
                ) : (
                    <div className="flex items-center justify-center h-full text-slate-400">
                        <div className="text-center p-8 rounded-2xl bg-white/5 border border-white/5 backdrop-blur-sm animate-fade-in mx-4">
                            <div className="mb-4 text-neon-blue opacity-50">
                                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="mx-auto">
                                    <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                                    <line x1="3" x2="21" y1="9" y2="9" />
                                    <line x1="9" x2="9" y1="21" y2="9" />
                                </svg>
                            </div>
                            <h3 className="text-xl font-medium text-white mb-2">No machine open</h3>
                            <button
                                onClick={() => addMachine({ name: 'New Machine' })}
                                className="mt-6 px-4 py-2 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-blue rounded-lg border border-neon-blue/20 transition-all font-medium text-sm"
                            >
                                Create New Machine
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Layer 10: Floating UI */}
            <div className="absolute inset-0 z-10 pointer-events-none p-4 flex flex-col md:flex-row gap-4">

                {/* Desktop Sidebar */}
                <div className="hidden md:flex h-full pointer-events-auto items-start">
                    <Sidebar />
                </div>

                <div className="flex-1 flex flex-col items-center gap-4 pointer-events-none">
                    {/* TabBar (Floating Top) */}
                    <TabBar />

                    {/* Toolbars (Center Bottom-ish or standard) */}
                    <div className="pointer-events-auto flex flex-col md:flex-row gap-4 animate-slide-up origin-top mt-auto mb-8">
                        <DesignToolbar />
                        {/* SimulationToolbar placeholder if needed, or remove if not refactored yet */}
                    </div>
                </div>

                {/* Mobile Menu Toggle (Top Right or similar) */}
                <button
                    className="md:hidden absolute top-4 right-4 pointer-events-auto p-3 text-slate-400 hover:text-white glass-button rounded-xl"
                    onClick={() => setIsMobileSidebarOpen(true)}
                >
                    <Menu size={24} />
                </button>
            </div>

            {/* Mobile Sidebar Overlay */}
            {isMobileSidebarOpen && (
                <div className="fixed inset-0 z-50 md:hidden flex pointer-events-auto">
                    <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={() => setIsMobileSidebarOpen(false)} />
                    <div className="w-4/5 h-full bg-midnight-900 border-r border-white/10 shadow-2xl relative animate-slide-right">
                        <Sidebar mobile />
                    </div>
                </div>
            )}
        </div>
    );
}
