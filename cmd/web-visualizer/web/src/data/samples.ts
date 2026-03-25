export const SAMPLES = [
    {
        name: '01 Welcome',
        jsonContent: JSON.stringify({
            nodes: [
                { id: '1', type: 'atomic', position: { x: 100, y: 100 }, data: { label: 'Start' } },
                { id: '2', type: 'atomic', position: { x: 300, y: 100 }, data: { label: 'Working' } },
                { id: '3', type: 'final', position: { x: 500, y: 100 }, data: { label: 'Done' } },
            ],
            edges: [
                { id: 'e1-2', source: '1', target: '2', label: 'start' },
                { id: 'e2-3', source: '2', target: '3', label: 'finish' },
            ]
        })
    },
    {
        name: '02 Traffic Light',
        jsonContent: JSON.stringify({
            nodes: [
                { id: 'red', type: 'atomic', position: { x: 200, y: 50 }, data: { label: 'Red' }, className: 'bg-red-100 dark:bg-red-900 border-red-500' },
                { id: 'yellow', type: 'atomic', position: { x: 200, y: 150 }, data: { label: 'Yellow' }, className: 'bg-yellow-100 dark:bg-yellow-900 border-yellow-500' },
                { id: 'green', type: 'atomic', position: { x: 200, y: 250 }, data: { label: 'Green' }, className: 'bg-green-100 dark:bg-green-900 border-green-500' },
            ],
            edges: [
                { id: 'e1', source: 'red', target: 'green', label: 'TIMER' }, // Simplified
                { id: 'e2', source: 'green', target: 'yellow', label: 'TIMER' },
                { id: 'e3', source: 'yellow', target: 'red', label: 'TIMER' },
            ]
        })
    }
];
