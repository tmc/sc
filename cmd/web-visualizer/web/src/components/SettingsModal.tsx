import { X } from 'lucide-react';
import { useSettingsStore } from '../store';

interface SettingsModalProps {
    onClose: () => void;
}

export function SettingsModal({ onClose }: SettingsModalProps) {
    const { theme, showGrid, showMinimap, setTheme, toggleGrid, toggleMinimap } = useSettingsStore();

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50" onClick={onClose}>
            <div className="bg-white dark:bg-slate-900 w-[500px] rounded-2xl shadow-2xl p-6" onClick={e => e.stopPropagation()}>
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-bold dark:text-white">Settings</h2>
                    <button onClick={onClose} className="p-1 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800">
                        <X size={20} />
                    </button>
                </div>

                <div className="space-y-6">
                    {/* Appearance */}
                    <div>
                        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">Appearance</h3>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <span className="text-slate-700 dark:text-slate-300">Theme</span>
                                <select
                                    value={theme}
                                    onChange={(e: any) => setTheme(e.target.value)}
                                    className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white"
                                >
                                    <option value="light">Light</option>
                                    <option value="dark">Dark</option>
                                    <option value="system">System</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* View */}
                    <div>
                        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">View</h3>
                        <div className="space-y-3">
                            <label className="flex items-center justify-between cursor-pointer">
                                <span className="text-slate-700 dark:text-slate-300">Show Grid</span>
                                <input type="checkbox" checked={showGrid} onChange={toggleGrid} className="accent-blue-500" />
                            </label>
                            <label className="flex items-center justify-between cursor-pointer">
                                <span className="text-slate-700 dark:text-slate-300">Show Minimap</span>
                                <input type="checkbox" checked={showMinimap} onChange={toggleMinimap} className="accent-blue-500" />
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
