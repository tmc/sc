import React from 'react';
import { ReactFlow, ReactFlowProvider, Background, Controls, MiniMap, type NodeTypes, type EdgeTypes, MarkerType, BackgroundVariant, useReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useEditorStore, useSettingsStore } from '../store';
import { AtomicNode, CompoundNode, ParallelNode, HistoryNode, FinalNode, TextNode } from './nodes';
import { TransitionEdge } from './edges';

const nodeTypes: NodeTypes = {
    atomic: AtomicNode,
    compound: CompoundNode,
    parallel: ParallelNode,
    history: HistoryNode,
    final: FinalNode,
    text: TextNode,
} as any;

const edgeTypes: EdgeTypes = {
    transition: TransitionEdge,
} as any;

function FlowCanvasInner() {
    const { nodes, edges, onNodesChange, onEdgesChange, onConnect, activeTool, setNodes, setActiveTool } = useEditorStore();
    const { showGrid, showMinimap, snapToGrid } = useSettingsStore();
    const { screenToFlowPosition } = useReactFlow();

    // Default to dark for Midnight Neon theme
    const isDark = true;

    const handlePaneClick = (event: React.MouseEvent<Element>) => {
        if (activeTool === 'node' || activeTool === 'text') {
            const position = screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });

            if (activeTool === 'node') {
                const newNode = {
                    id: `node-${Date.now()}`,
                    type: 'atomic',
                    position,
                    data: { label: 'New State' },
                };
                setNodes([...nodes, newNode]);
            } else if (activeTool === 'text') {
                const newNode = {
                    id: `text-${Date.now()}`,
                    type: 'text',
                    position,
                    data: { label: 'Double click to edit' },
                };
                setNodes([...nodes, newNode]);
            }

            setActiveTool('select'); // Switch back to select after adding
        }
    };

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            snapToGrid={snapToGrid}
            fitView
            className={`bg-transparent ${activeTool === 'node' || activeTool === 'text' ? 'cursor-crosshair' : ''}`} // handled by parent container gradient
            minZoom={0.2}
            maxZoom={4}
            panOnDrag={activeTool === 'pan' || (activeTool !== 'select' && activeTool !== 'node' && activeTool !== 'text')}
            selectionOnDrag={activeTool === 'select'}
            nodesDraggable={activeTool === 'select'}
            onPaneClick={handlePaneClick}
            defaultEdgeOptions={{
                type: 'transition',
                markerEnd: { type: MarkerType.ArrowClosed, color: isDark ? '#475569' : '#000' },
                style: { strokeWidth: 2, stroke: isDark ? '#475569' : '#64748b' },
            }}
        >
            {showGrid && (
                <Background
                    color="#334155" // Slate 700
                    gap={20}
                    size={1}
                    variant={BackgroundVariant.Dots}
                    className="opacity-50"
                />
            )}
            <Controls className="bg-white/5 border-white/10 text-slate-300 fill-slate-300 [&>button]:border-white/10 [&>button:hover]:bg-white/10" />

            {showMinimap && (
                <MiniMap
                    style={{
                        backgroundColor: 'rgba(15, 23, 42, 0.5)', // midnight-900/50
                        border: '1px solid rgba(255,255,255,0.05)',
                    }}
                    nodeColor={(n) => {
                        if (n.type === 'compound') return 'rgba(176, 38, 255, 0.2)';
                        if (n.selected) return '#00f0ff';
                        return '#1e293b';
                    }}
                    maskColor="rgba(2, 6, 23, 0.7)" // midnight-950/70
                />
            )}
        </ReactFlow>
    );
}

export function FlowCanvas() {
    return (
        <ReactFlowProvider>
            <FlowCanvasInner />
        </ReactFlowProvider>
    );
}
