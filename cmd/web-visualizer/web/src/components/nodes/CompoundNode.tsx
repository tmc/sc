import { Handle, Position, type NodeProps } from '@xyflow/react';

export function CompoundNode({ data, selected }: NodeProps) {
    return (
        <div className={`
            rounded-2xl border transition-all duration-300
            bg-midnight-950/40 backdrop-blur-sm
            min-w-[200px] min-h-[150px]
            ${selected
                ? 'border-neon-purple shadow-[0_0_20px_rgba(176,38,255,0.2)]'
                : 'border-white/5 hover:border-white/10'
            }
        `}>
            <Handle
                type="target"
                position={Position.Left}
                className={`!w-3 !h-3 !-left-1.5 !bg-neon-purple`}
            />

            {/* Header */}
            <div className={`
                px-4 py-2 rounded-t-2xl border-b transition-colors duration-300
                ${selected
                    ? 'bg-neon-purple/20 border-neon-purple/30 text-neon-purple'
                    : 'bg-white/5 border-white/5 text-slate-400'
                }
            `}>
                <span className="text-xs font-bold uppercase tracking-wider">{data.label as string}</span>
            </div>

            {/* Body */}
            <div className="p-4">
                {/* Child nodes will be rendered here by React Flow due to parenting, 
                    but visually this is just the container */}
            </div>

            <Handle
                type="source"
                position={Position.Right}
                className={`!w-3 !h-3 !-right-1.5 !bg-neon-purple`}
            />
        </div>
    );
}
