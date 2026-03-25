import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import { useSimulationStore } from '../store';

export function SimulationToolbar() {
    const { isRunning, startSimulation, stopSimulation, stepForward, stepBack, reset } = useSimulationStore();

    return (
        <div className="glass-pill flex gap-1 p-1.5 items-center">
            <button onClick={stepBack} className="p-2.5 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-all duration-200 hover:scale-105 active:scale-95" title="Step Back">
                <SkipBack size={18} />
            </button>

            <button
                onClick={isRunning ? stopSimulation : startSimulation}
                className={`p-2.5 rounded-full transition-all duration-200 hover:scale-105 active:scale-95 shadow-lg ${isRunning
                    ? 'bg-neon-pink/20 text-neon-pink hover:bg-neon-pink/30 shadow-neon-pink/20'
                    : 'bg-neon-green/20 text-neon-green hover:bg-neon-green/30 shadow-neon-green/20'
                    }`}
                title={isRunning ? "Pause" : "Play"}
            >
                {isRunning ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
            </button>

            <button onClick={stepForward} className="p-2.5 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-all duration-200 hover:scale-105 active:scale-95" title="Step Forward">
                <SkipForward size={18} />
            </button>

            <div className="w-px h-5 bg-white/10 mx-1" />

            <button onClick={reset} className="p-2.5 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-all duration-200 hover:scale-105 active:scale-95" title="Reset">
                <RotateCcw size={18} />
            </button>
        </div>
    );
}
