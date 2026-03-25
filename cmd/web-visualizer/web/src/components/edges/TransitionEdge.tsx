import { BaseEdge, EdgeLabelRenderer, type EdgeProps, getSmoothStepPath } from '@xyflow/react';

export function TransitionEdge({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
    label,
    selected,
}: EdgeProps) {
    const [edgePath, labelX, labelY] = getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });

    return (
        <>
            <BaseEdge
                path={edgePath}
                markerEnd={markerEnd}
                style={{
                    ...style,
                    strokeWidth: selected ? 3 : 2,
                    stroke: selected ? '#00f0ff' : '#475569', // Neon Blue or Slate 600
                    filter: selected ? 'drop-shadow(0 0 4px rgba(0, 240, 255, 0.5))' : 'none',
                    transition: 'all 0.3s ease',
                }}
            />
            {label && (
                <EdgeLabelRenderer>
                    <div
                        style={{
                            position: 'absolute',
                            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                            pointerEvents: 'all',
                        }}
                        className={`
                            nodrag nopan px-2 py-1 rounded-md text-[10px] font-mono tracking-wide
                            transition-all duration-200
                            ${selected
                                ? 'bg-midnight-900 text-neon-blue border border-neon-blue shadow-neon'
                                : 'bg-midnight-900/80 text-slate-400 border border-white/10 hover:border-white/20'
                            }
                        `}
                    >
                        {label}
                    </div>
                </EdgeLabelRenderer>
            )}
        </>
    );
}
