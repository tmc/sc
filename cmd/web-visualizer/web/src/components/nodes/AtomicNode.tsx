import { Handle, Position, type NodeProps } from '@xyflow/react';

export function AtomicNode({ data, selected }: NodeProps) {
    return (
        <div className={`
            px-5 py-3 rounded-xl 
            transition-all duration-300 ease-out
            min-w-[120px] text-center
            backdrop-blur-md
            ${selected
                ? 'bg-midnight-800/80 border-neon-blue shadow-[0_0_15px_rgba(0,240,255,0.3)] scale-105'
                : 'bg-midnight-900/60 border-white/10 hover:border-white/20 hover:bg-midnight-800/60'
            }
            border
        `}>
            {/* Input Handle */}
            <Handle
                type="target"
                position={Position.Left}
                className={`!w-3 !h-3 !-left-1.5 transition-colors duration-300 ${selected ? '!bg-neon-blue' : '!bg-slate-500'}`}
            />

            <div className="flex flex-col items-center">
                <span className={`font-medium tracking-wide transition-colors duration-300 ${selected ? 'text-neon-blue' : 'text-slate-200'}`}>
                    {data.label as string}
                </span>
            </div>

            {/* Output Handle */}
            <Handle
                type="source"
                position={Position.Right}
                className={`!w-3 !h-3 !-right-1.5 transition-colors duration-300 ${selected ? '!bg-neon-blue' : '!bg-slate-500'}`}
            />

            {/* Active Glow Effect (Optional - if active) */}
            {data.active && (
                <div className="absolute inset-0 rounded-xl bg-neon-green/10 animate-pulse-slow pointer-events-none" />
            )}
        </div>
    );
}
