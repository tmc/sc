import { useRef, useState } from 'preact/hooks';
import { useMachinesStore } from '../store';
import { Plus, Search, Trash2, FileText, Upload, Download, ChevronLeft, ChevronRight } from 'lucide-react';

interface SidebarProps {
    mobile?: boolean;
}

export function Sidebar({ mobile }: SidebarProps) {
    const { machines, selectMachine, selectedMachineId, addMachine, deleteMachine, openMachine } = useMachinesStore();
    const [searchTerm, setSearchTerm] = useState('');
    const [isCollapsed, setIsCollapsed] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const filteredMachines = machines.filter(m =>
        m.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleCreateNew = () => {
        addMachine({ name: 'New Machine' });
    };

    const handleImport = (e: any) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const content = event.target?.result as string;
                JSON.parse(content);
                addMachine({
                    name: file.name.replace('.json', ''),
                    jsonContent: content
                });
            } catch (err) {
                console.error('Failed to import', err);
                alert('Invalid JSON file');
            }
        };
        reader.readAsText(file);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleExport = (e: any, machine: typeof machines[0]) => {
        e.stopPropagation();
        const blob = new Blob([machine.jsonContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${machine.name}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Mobile specific adjustments
    if (mobile) {
        return (
            <div className="flex flex-col h-full p-4 overflow-hidden">
                <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 mb-6">Library</h2>

                <div className="flex gap-2 mb-6">
                    <div className="relative group flex-1">
                        <Search className="absolute left-3 top-3 text-slate-500 group-focus-within:text-neon-blue transition-colors" size={18} />
                        <input
                            type="text"
                            placeholder="Search..."
                            value={searchTerm}
                            onChange={(e: any) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-midnight-800/50 border border-white/10 text-slate-200 focus:border-neon-blue/50 focus:ring-1 focus:ring-neon-blue/50 text-base"
                        />
                    </div>
                    <button
                        onClick={handleCreateNew}
                        className="p-3 bg-neon-blue/10 rounded-xl text-neon-blue border border-neon-blue/20 hover:bg-neon-blue/20 active:scale-95 transition-all"
                    >
                        <Plus size={20} />
                    </button>
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className="p-3 bg-white/5 rounded-xl text-slate-400 border border-white/10 hover:bg-white/10 active:scale-95 transition-all"
                    >
                        <Upload size={20} />
                    </button>
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept=".json"
                        onChange={handleImport}
                    />
                </div>

                <div className="flex-1 overflow-y-auto space-y-2">
                    {filteredMachines.map(machine => (
                        <div
                            key={machine.id}
                            onClick={() => { selectMachine(machine.id); openMachine(machine.id); }}
                            className={`flex items-center p-3 rounded-xl border transition-all active:scale-98
                                ${selectedMachineId === machine.id
                                    ? 'bg-neon-blue/5 border-neon-blue/20 shadow-neon-blue/5'
                                    : 'bg-white/5 border-white/5 hover:bg-white/10'}`}
                        >
                            <div className={`p-2 rounded-lg mr-3 ${selectedMachineId === machine.id ? 'bg-neon-blue/20 text-neon-blue' : 'bg-midnight-950/50 text-slate-500'}`}>
                                <FileText size={20} />
                            </div>
                            <span className={`text-base font-medium flex-1 ${selectedMachineId === machine.id ? 'text-white' : 'text-slate-300'}`}>
                                {machine.name}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Desktop Floating Sidebar
    return (
        <div
            className={`${isCollapsed ? 'w-18' : 'w-72'} 
            transition-all duration-300 ease-in-out 
            glass-panel rounded-2xl flex flex-col h-[calc(100vh-32px)] 
            relative group/sidebar`}
        >
            {/* Collapse Toggle */}
            <button
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="absolute -right-3 top-6 bg-midnight-800 border border-white/10 rounded-full p-1.5 text-slate-400 
                           hover:text-white shadow-lg z-50 transition-all hover:scale-110 opacity-0 group-hover/sidebar:opacity-100"
                title={isCollapsed ? "Expand" : "Collapse"}
            >
                {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
            </button>

            <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept=".json"
                onChange={handleImport}
            />

            {/* Header */}
            <div className="p-4 flex flex-col justify-between shrink-0 gap-4">
                <div className="flex items-center justify-between">
                    {!isCollapsed && <h2 className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 text-lg tracking-tight pl-1">Library</h2>}
                    <div className={`flex gap-1 ${isCollapsed ? 'flex-col w-full items-center' : ''}`}>
                        {/* Action Buttons */}
                        <button
                            onClick={handleCreateNew}
                            className={`p-2 rounded-lg transition-all duration-200 
                                       ${isCollapsed
                                    ? 'bg-neon-blue/10 text-neon-blue hover:bg-neon-blue/20'
                                    : 'hover:bg-white/10 text-slate-400 hover:text-white'}`}
                            title="New Machine"
                        >
                            <Plus size={18} />
                        </button>
                    </div>
                </div>

                {/* Compact Search / Expanded Search */}
                {!isCollapsed ? (
                    <div className="relative group">
                        <Search className="absolute left-3 top-2.5 text-slate-500 group-focus-within:text-neon-blue transition-colors" size={16} />
                        <input
                            type="text"
                            placeholder="Search..."
                            value={searchTerm}
                            onChange={(e: any) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-3 py-2 rounded-xl bg-midnight-950/30 border border-white/5 
                                       focus:border-neon-blue/30 focus:shadow-neon/20 focus:ring-1 focus:ring-neon-blue/30 
                                       active:outline-none focus:outline-none text-sm text-slate-200 placeholder-slate-600 transition-all font-medium"
                        />
                    </div>
                ) : (
                    <button
                        className="w-full aspect-square flex items-center justify-center rounded-xl bg-midnight-950/30 hover:bg-white/5 text-slate-500 hover:text-white transition-colors"
                        onClick={() => setIsCollapsed(false)}
                        title="Search"
                    >
                        <Search size={18} />
                    </button>
                )}
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1 scrollbar-hide">
                {filteredMachines.map(machine => (
                    <div
                        key={machine.id}
                        onClick={() => selectMachine(machine.id)}
                        onDblClick={() => openMachine(machine.id)}
                        className={`flex items-center justify-between p-2 pl-2 rounded-xl cursor-pointer group transition-all duration-200 relative overflow-hidden
                            ${selectedMachineId === machine.id
                                ? 'bg-gradient-to-br from-white/10 to-transparent border border-white/5 shadow-glass-sm'
                                : 'hover:bg-white/5 border border-transparent hover:border-white/5 text-slate-400 hover:text-slate-200'
                            }`}
                        title={isCollapsed ? machine.name : undefined}
                    >
                        {/* Selected Indicator - Left Bar */}
                        {selectedMachineId === machine.id && (
                            <div className="absolute left-0 top-2 bottom-2 w-1 bg-neon-blue rounded-r-full shadow-[0_0_10px_rgba(0,240,255,0.5)]" />
                        )}

                        <div className={`flex items-center gap-3 overflow-hidden ${isCollapsed ? 'justify-center w-full' : 'pl-2'}`}>
                            <div className={`p-1.5 rounded-lg shrink-0 ${selectedMachineId === machine.id ? 'bg-neon-blue/20 text-neon-blue' : 'bg-midnight-950/50 text-slate-500 group-hover:text-slate-300'}`}>
                                <FileText size={16} />
                            </div>
                            {!isCollapsed && <span className={`truncate text-sm font-medium ${selectedMachineId === machine.id ? 'text-white' : ''}`}>{machine.name}</span>}
                        </div>

                        {!isCollapsed && (
                            <div className="flex opacity-0 group-hover:opacity-100 transition-opacity absolute right-2 bg-midnight-800/80 backdrop-blur-sm rounded-lg border border-white/10 p-0.5 shadow-lg">
                                <button
                                    onClick={(e) => handleExport(e, machine)}
                                    className="p-1.5 hover:text-neon-blue transition-colors rounded hover:bg-white/10"
                                    title="Export JSON"
                                >
                                    <Download size={12} />
                                </button>
                                <button
                                    onClick={(e) => { e.stopPropagation(); deleteMachine(machine.id); }}
                                    className="p-1.5 hover:text-neon-pink transition-colors rounded hover:bg-white/10"
                                    title="Delete"
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Footer / Import / Settings Area */}
            <div className="p-2 border-t border-white/5 bg-black/20 mt-auto rounded-b-2xl">
                {!isCollapsed ? (
                    <div className="flex gap-2">
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="flex-1 flex items-center justify-center gap-2 p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-all text-xs font-medium border border-transparent hover:border-white/10"
                        >
                            <Upload size={14} />
                            Import
                        </button>
                    </div>
                ) : (
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-all flex justify-center"
                        title="Import"
                    >
                        <Upload size={16} />
                    </button>
                )}
            </div>
        </div>
    );
}
