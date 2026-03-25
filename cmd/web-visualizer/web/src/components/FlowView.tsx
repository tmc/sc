import { useRef, useEffect, useState } from 'react';
import { useAppStore, type FlowNode } from '../store/AppViewModel';

export function FlowView() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const { nodes, edges, selection, activeStateIDs, selectNode } = useAppStore();

    // Viewport State
    const [scale, setScale] = useState(1);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

    // Constants
    const DOT_SPACING = 20;
    const DOT_SIZE = 2;

    // --- Rendering Logic ---
    const draw = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
        // Clear
        ctx.fillStyle = '#f8fafc'; // bg-slate-50
        // Check dark mode
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            ctx.fillStyle = '#0f172a'; // bg-slate-900
        }
        ctx.fillRect(0, 0, width, height);

        const centerX = width / 2 + offset.x;
        const centerY = height / 2 + offset.y;

        // 1. Grid (Dot Grid)
        ctx.fillStyle = 'rgba(148, 163, 184, 0.2)'; // slate-400 opacity 0.2
        const spacing = DOT_SPACING * scale;
        const startX = (centerX % spacing) - spacing;
        const startY = (centerY % spacing) - spacing;

        for (let x = startX; x < width; x += spacing) {
            for (let y = startY; y < height; y += spacing) {
                ctx.beginPath();
                ctx.arc(x, y, DOT_SIZE * scale / 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Apply Transform for World Space
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.scale(scale, scale);

        // 2. Edges
        ctx.strokeStyle = '#94a3b8'; // slate-400
        ctx.lineWidth = 1.5;
        edges.forEach(edge => {
            const source = nodes.find(n => n.id === edge.source);
            const target = nodes.find(n => n.id === edge.target);
            if (!source || !target) return;

            const start = { x: source.position.x + source.size.width / 2, y: source.position.y + source.size.height / 2 };

            // Draw Edge
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            // Orthogonal Routing placeholder (Straight for MVP)
            ctx.lineTo(target.position.x + target.size.width / 2, target.position.y + target.size.height / 2);
            ctx.stroke();
        });

        // 3. Nodes
        nodes.forEach(node => {
            drawNode(ctx, node);
        });

        ctx.restore();
    };

    const drawNode = (ctx: CanvasRenderingContext2D, node: FlowNode) => {
        const isSelected = selection.includes(node.id);
        const isActive = activeStateIDs.includes(node.id);
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

        const x = node.position.x;
        const y = node.position.y;
        const w = node.size.width;
        const h = node.size.height;
        let r = 8; // corner radius

        if (node.type === 'compound' || node.type === 'parallel') r = 12;
        if (node.type === 'history' || node.type === 'final') r = w / 2;

        // Path
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);

        // Halo (Selection)
        if (isSelected) {
            ctx.save();
            ctx.shadowColor = 'rgba(59, 130, 246, 0.5)'; // blue-500
            ctx.shadowBlur = 10;
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 4;
            ctx.stroke();
            ctx.restore();
        }

        // Active Glow
        if (isActive) {
            ctx.save();
            ctx.shadowColor = 'rgba(34, 197, 94, 0.6)'; // green-500
            ctx.shadowBlur = 12;
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 4;
            ctx.stroke();
            ctx.restore();
        }

        // Fill
        ctx.fillStyle = isDark ? '#1e293b' : '#ffffff'; // slate-800 : white
        if (node.type === 'compound') ctx.fillStyle = isDark ? '#0f172a' : '#f8fafc';

        ctx.fill();

        // Stroke
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.strokeStyle = isSelected ? '#3b82f6' : (isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)');
        ctx.stroke();

        // Header (Compound)
        if (node.type === 'compound') {
            const headerH = 24;
            ctx.save();
            ctx.beginPath();
            ctx.roundRect(x, y, w, h, r);
            ctx.clip(); // Clip to node shape

            ctx.fillStyle = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            ctx.beginPath();
            ctx.rect(x, y, w, headerH);
            ctx.fill();
            ctx.restore();

            // Label in Header
            ctx.fillStyle = isDark ? '#e2e8f0' : '#475569';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.fillText(node.label, x + 8, y + 16);
        } else {
            // Centered Label
            ctx.fillStyle = isDark ? '#f8fafc' : '#0f172a';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.label, x + w / 2, y + h / 2);
        }
    };

    // --- Interaction Handlers ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;


        const render = () => {
            if (containerRef.current) {
                canvas.width = containerRef.current.clientWidth;
                canvas.height = containerRef.current.clientHeight;
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    draw(ctx, canvas.width, canvas.height);
                }
            }
            // animationFrameId = requestAnimationFrame(render); // Only needed if animating continuously
        };

        render();
        window.addEventListener('resize', render);

        // Re-render on state changes
        return () => window.removeEventListener('resize', render);
    }, [nodes, edges, selection, activeStateIDs, scale, offset]);


    const handleMouseDown = (e: any) => {
        setIsDragging(true);
        setLastMousePos({ x: e.clientX, y: e.clientY });
    };

    const handleMouseMove = (e: any) => {
        if (isDragging) {
            const dx = e.clientX - lastMousePos.x;
            const dy = e.clientY - lastMousePos.y;
            setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
            setLastMousePos({ x: e.clientX, y: e.clientY });
        }
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleWheel = (e: any) => {
        const zoomSensitivity = 0.001;
        const newScale = Math.max(0.1, Math.min(5, scale - e.deltaY * zoomSensitivity));
        setScale(newScale);
    };

    // Hit Test (Simple)
    const handleClick = (e: any) => {
        if (isDragging) return; // Don't select if dragged

        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;

        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Transform to world space
        const worldX = (clickX - rect.width / 2 - offset.x) / scale;
        const worldY = (clickY - rect.height / 2 - offset.y) / scale;

        // Find node
        // Reverse iterate to hit top-most first
        for (let i = nodes.length - 1; i >= 0; i--) {
            const n = nodes[i];
            if (worldX >= n.position.x && worldX <= n.position.x + n.size.width &&
                worldY >= n.position.y && worldY <= n.position.y + n.size.height) {
                selectNode(n.id, e.shiftKey || e.metaKey);
                return;
            }
        }

        selectNode('__clear__', false);
    };

    return (
        <div ref={containerRef} className="w-full h-full bg-slate-50 dark:bg-slate-900 overflow-hidden relative">
            <canvas
                ref={canvasRef}
                className="block cursor-grab active:cursor-grabbing"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                onClick={handleClick}
            />
        </div>
    );
}
