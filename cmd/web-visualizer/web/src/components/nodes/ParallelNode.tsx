import { Handle, Position, type NodeProps } from '@xyflow/react';

export function ParallelNode({ data, selected }: NodeProps) {
    return (
        <div className={`
      rounded-xl bg-transparent
      border-2 border-dashed transition-all min-w-[200px] min-h-[150px]
      ${selected
                ? 'border-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,0.5)]'
                : 'border-slate-400 dark:border-slate-500'
            }
    `}>
            <div className="absolute -top-3 left-4 bg-slate-50 dark:bg-slate-900 px-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
                {data.label as string}
            </div>

            <Handle type="target" position={Position.Left} className="w-3 h-3 bg-blue-500" />
            <Handle type="source" position={Position.Right} className="w-3 h-3 bg-blue-500" />
        </div>
    );
}
