import { X } from 'lucide-react';
import { useMachinesStore } from '../store';

export function TabBar() {
    const { openMachineIds, machines, selectedMachineId, selectMachine, closeMachine } = useMachinesStore();

    if (openMachineIds.length === 0) return null;

    const openMachines = machines.filter(m => openMachineIds.includes(m.id));

    return (
        <div className="glass-pill flex items-center p-1 gap-1 max-w-[80vw] overflow-x-auto scrollbar-hide shrink-0 z-20 pointer-events-auto mx-auto mt-2">
            {openMachines.map(machine => {
                const isActive = selectedMachineId === machine.id;
                return (
                    <div
                        key={machine.id}
                        onClick={() => selectMachine(machine.id)}
                        className={`
                            group relative flex items-center gap-2 px-3 py-1.5 rounded-full cursor-pointer transition-all duration-200 select-none border whitespace-nowrap
                            ${isActive
                                ? 'bg-neon-blue/10 border-neon-blue/20 text-white shadow-neon/20'
                                : 'bg-transparent border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-200'
                            }
                        `}
                    >
                        <span className="text-xs font-medium">{machine.name}</span>

                        <button
                            onClick={(e) => { e.stopPropagation(); closeMachine(machine.id); }}
                            className={`p-0.5 rounded-full transition-all 
                                ${isActive
                                    ? 'hover:bg-neon-blue/20 text-neon-blue'
                                    : 'opacity-0 group-hover:opacity-100 hover:bg-white/10 text-slate-500 hover:text-white'}`}
                        >
                            <X size={12} />
                        </button>
                    </div>
                );
            })}
        </div>
    );
}
