import { Handle, Position, type NodeProps } from '@xyflow/react';

export function HistoryNode({ data, selected }: NodeProps) {
    const label = data.label as string || 'H';

    return (
        <div className={`
      w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/30 
      border-2 flex items-center justify-center shadow-sm transition-all
      ${selected
                ? 'border-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,0.5)]'
                : 'border-amber-400 dark:border-amber-600'
            }
    `}>
            <span className="font-bold text-amber-700 dark:text-amber-500 text-sm">{label}</span>
            <Handle type="target" position={Position.Left} className="w-2 h-2 bg-amber-500" />
        </div>
    );
}
