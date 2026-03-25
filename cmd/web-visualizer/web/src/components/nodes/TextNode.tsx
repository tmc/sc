import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useEditorStore } from '../../store';

export const TextNode = memo(({ id, data, selected }: NodeProps) => {
    const updateNodeData = useEditorStore((state) => state.updateNodeData);

    return (
        <div className={`group relative min-w-[100px] min-h-[40px] p-2 transition-all duration-300
            ${selected ? 'ring-1 ring-neon-blue/50' : 'hover:ring-1 hover:ring-white/10'}
            rounded-lg
        `}>
            <div className="text-slate-300 font-medium text-lg bg-transparent border-none outline-none w-full h-full cursor-text"
                contentEditable={selected}
                suppressContentEditableWarning={true}
                onBlur={(e) => {
                    const newLabel = e.currentTarget.textContent;
                    // Only update store if changed to avoid unnecessary re-renders
                    if (newLabel !== data.label) {
                        updateNodeData(id, { label: newLabel });
                    }
                }}
            >
                {data.label as string || 'Text'}
            </div>

            {/* Invisible handles for connecting if desired, or omit for pure text */}
            <Handle type="target" position={Position.Top} className="opacity-0" />
            <Handle type="source" position={Position.Bottom} className="opacity-0" />
        </div>
    );
});
