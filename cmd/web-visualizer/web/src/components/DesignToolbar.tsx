import { MousePointer2, Move, PlusSquare, Type } from 'lucide-react';
import { useEditorStore, type EditorTool } from '../store';

export function DesignToolbar() {
    const { activeTool, setActiveTool } = useEditorStore();

    const tools: { id: EditorTool; icon: typeof MousePointer2; title: string }[] = [
        { id: 'select', icon: MousePointer2, title: 'Select' },
        { id: 'pan', icon: Move, title: 'Pan' },
        { id: 'node', icon: PlusSquare, title: 'Add State' },
        { id: 'text', icon: Type, title: 'Add Text' },
    ];

    return (
        <div className="glass-pill flex gap-1 p-1.5 items-center z-50 pointer-events-auto">
            {tools.map((tool, index) => {
                const isActive = activeTool === tool.id;
                return (
                    <div key={tool.id} className="contents">
                        {index === 2 && <div className="w-px h-5 bg-white/10 mx-1" />}
                        <button
                            onClick={() => setActiveTool(tool.id)}
                            className={`p-2.5 rounded-full transition-all duration-200 hover:scale-105 active:scale-95 ${isActive
                                ? 'bg-neon-blue/20 text-neon-blue shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                                : 'hover:bg-white/10 text-slate-400 hover:text-white'
                                }`}
                            title={tool.title}
                        >
                            <tool.icon size={18} />
                        </button>
                    </div>
                );
            })}
        </div>
    );
}
