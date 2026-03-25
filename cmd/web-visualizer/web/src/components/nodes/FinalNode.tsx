import { Handle, Position, type NodeProps } from '@xyflow/react';

export function FinalNode({ selected }: NodeProps) {
    return (
        <div className={`
      w-8 h-8 rounded-full bg-slate-50 dark:bg-slate-900 
      border-2 flex items-center justify-center transition-all
      ${selected
                ? 'border-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,0.5)]'
                : 'border-slate-800 dark:border-slate-200'
            }
    `}>
            <div className="w-5 h-5 rounded-full bg-slate-800 dark:bg-slate-200" />
            <Handle type="target" position={Position.Left} className="w-2 h-2 bg-slate-500" />
        </div>
    );
}
