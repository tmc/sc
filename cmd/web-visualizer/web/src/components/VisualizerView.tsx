import { useState } from 'react';
import { FlowView } from './FlowView';
import { useAppStore } from '../store/AppViewModel';
import { Play, Maximize, Share, Plus, Settings } from 'lucide-react';

export function VisualizerView() {
    const { generateMachine } = useAppStore();
    const [isGenerating, setIsGenerating] = useState(false);
    const [prompt, setPrompt] = useState('');
    const [showGenSheet, setShowGenSheet] = useState(false);

    const handleGenerate = async () => {
        setIsGenerating(true);
        await generateMachine(prompt);
        setIsGenerating(false);
        setShowGenSheet(false);
    };

    return (
        <div className="flex flex-col h-screen w-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans">
            {/* Toolbar */}
            <header className="h-14 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md flex items-center px-4 justify-between sticky top-0 z-10">
                <div className="flex items-center gap-4">
                    <h1 className="font-bold text-lg tracking-tight">States<span className="text-blue-500">.web</span></h1>

                    <div className="h-6 w-px bg-slate-200 dark:bg-slate-700 mx-2" />

                    <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
                        <button className="p-1.5 rounded-md hover:bg-white dark:hover:bg-slate-700 shadow-sm transition-all" title="Design Mode">
                            <Settings size={18} />
                        </button>
                        <button className="p-1.5 rounded-md hover:bg-white dark:hover:bg-slate-700 text-slate-400 hover:text-blue-500 transition-all" title="Simulate Mode">
                            <Play size={18} />
                        </button>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowGenSheet(true)}
                        className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-full text-sm font-medium shadow-lg shadow-blue-500/20 transition-all"
                    >
                        <Plus size={16} />
                        Generate
                    </button>

                    <button className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors" title="Extensions">
                        <Maximize size={18} />
                    </button>
                    <button className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors">
                        <Share size={18} />
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 overflow-hidden relative">
                <FlowView />

                {/* Generation Sheet (Overlay) */}
                {showGenSheet && (
                    <div className="absolute inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50">
                        <div className="bg-white dark:bg-slate-900 w-[500px] rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-800 p-6 flex flex-col gap-4 animate-in fade-in zoom-in-95 duration-200">
                            <div>
                                <h2 className="text-xl font-bold">Generate Statechart</h2>
                                <p className="text-slate-500 dark:text-slate-400 text-sm">Describe the logic you want to build.</p>
                            </div>

                            <textarea
                                value={prompt}
                                onChange={(e: any) => setPrompt(e.currentTarget.value)}
                                placeholder="e.g. A traffic light system with a pedestrian crossing..."
                                className="w-full h-32 p-4 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 resize-none focus:ring-2 focus:ring-blue-500 outline-none"
                            />

                            <div className="flex gap-2">
                                {['Traffic Light', 'Login Flow', 'Music Player'].map(p => (
                                    <button
                                        key={p}
                                        onClick={() => setPrompt(p)}
                                        className="px-3 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-xs font-medium rounded-full transition-colors border border-transparent hover:border-blue-200 dark:hover:border-blue-800"
                                    >
                                        {p}
                                    </button>
                                ))}
                            </div>

                            <div className="flex justify-end gap-2 mt-2">
                                <button
                                    onClick={() => setShowGenSheet(false)}
                                    className="px-4 py-2 text-slate-500 hover:text-slate-700 font-medium"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleGenerate}
                                    disabled={!prompt || isGenerating}
                                    className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-xl font-semibold shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2"
                                >
                                    {isGenerating ? 'Designing...' : 'Generate'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
