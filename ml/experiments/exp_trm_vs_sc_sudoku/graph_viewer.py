#!/usr/bin/env python3
"""
Interactive webapp to view SC-TRM benchmark graphs with Chart.js.

Usage:
    python experiments/exp_trm_vs_sc_sudoku/graph_viewer.py

Then open http://localhost:5050
"""

import os
import sys
import json
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

app = Flask(__name__)

RESULTS_DIR = "/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_trm_vs_sc_sudoku/benchmark_results"

MODEL_COLORS = {
    'vanilla': '#808080',
    'faithful': '#000000',
    'ste': '#2196F3',
    'confidence': '#4CAF50',
    'attention_bias': '#FF9800',
    'hierarchical': '#9C27B0',
    'learned': '#E91E63',
    'nas': '#00BCD4',
    # New models
    'iterative': '#3F51B5',
    'iterative_mlp': '#673AB7',
    'faithful_v2': '#009688',
    'faithful_v2_attn': '#00796B',
    'hard_mask': '#F44336',
    'hard_mask_iter': '#D32F2F',
    'nas_pretrained': '#607D8B',
    'large': '#795548',
    'xlarge': '#5D4037',
    # SAE+Diff models
    'sae_diff': '#8BC34A',
    'sae_diff_sudoku': '#689F38',
    # Faithful v3 (corrected)
    'faithful_v3': '#1E88E5',
    'faithful_v3_attn': '#1565C0',
    # Faithful v4 (StableMax + trunc_normal)
    'faithful_v4': '#43A047',
    'faithful_v4_attn': '#2E7D32',
    'faithful_v4_rope': '#1B5E20',
    # ACT (Adaptive Computation Time)
    'faithful_act': '#FF5722',
    'faithful_act_attn': '#E64A19',
    # Hybrid
    'hybrid': '#D50000',
}

MODEL_LABELS = {
    'vanilla': 'Vanilla TRM',
    'faithful': 'Faithful TRM',
    'ste': 'STE Guards',
    'confidence': 'Confidence Guards',
    'attention_bias': 'Attention Bias',
    'hierarchical': 'Hierarchical SC',
    'hybrid': 'Hybrid (Attn+Trunc)',
    'learned': 'Learned SC',
    'nas': 'NAS-SC',
    # New models
    'iterative': 'Iterative (Attn)',
    'iterative_mlp': 'Iterative (MLP)',
    'faithful_v2': 'Faithful v2 (MLP)',
    'faithful_v2_attn': 'Faithful v2 (Attn)',
    'hard_mask': 'Hard Mask',
    'hard_mask_iter': 'Hard Mask + Iter',
    'nas_pretrained': 'NAS Pretrained',
    'large': 'Large (256d)',
    'xlarge': 'XLarge (512d)',
    # SAE+Diff models
    'sae_diff': 'SAE+Diff (cold)',
    'sae_diff_sudoku': 'SAE+Diff (warm)',
    # Faithful v3 (corrected)
    'faithful_v3': 'Faithful v3 (MLP)',
    'faithful_v3_attn': 'Faithful v3 (Attn)',
    # Faithful v4 (StableMax)
    'faithful_v4': 'Faithful v4 (MLP)',
    'faithful_v4_attn': 'Faithful v4 (Attn)',
    'faithful_v4_rope': 'Faithful v4 (RoPE)',
    # ACT (Adaptive Computation Time)
    'faithful_act': 'ACT (MLP)',
    'faithful_act_attn': 'ACT (Attn)',
}

# Model groups for toggle buttons
MODEL_GROUPS = {
    'Faithful': ['faithful', 'faithful_v2', 'faithful_v2_attn', 'faithful_v3', 'faithful_v3_attn', 'faithful_v4', 'faithful_v4_attn', 'faithful_v4_rope'],
    'ACT': ['faithful_act', 'faithful_act_attn'],
    'Iterative': ['iterative', 'iterative_mlp'],
    'Hard Mask': ['hard_mask', 'hard_mask_iter'],
    'NAS': ['nas', 'nas_pretrained'],
    'SAE+Diff': ['sae_diff', 'sae_diff_sudoku'],
    'Guards': ['ste', 'confidence'],
    'Structure': ['hierarchical', 'learned', 'attention_bias', 'hybrid'],
    'Scale': ['vanilla', 'large', 'xlarge'],
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SC-TRM Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #FF9800;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .controls {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        button {
            background: #FF9800;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        button:hover { background: #F57C00; }
        button.secondary { background: #2196F3; }
        button.secondary:hover { background: #1976D2; }
        button.danger { background: #f44336; }
        button.danger:hover { background: #d32f2f; }
        .status { color: #666; font-size: 12px; }
        .refresh-time { color: #999; font-size: 12px; margin-left: auto; }
        .graphs {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 1200px) {
            .graphs { grid-template-columns: 1fr; }
        }
        .graph-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .graph-container h3 {
            margin: 0 0 5px 0;
            color: #555;
        }
        .graph-hint {
            font-size: 11px;
            color: #999;
            margin-bottom: 10px;
        }
        .chart-wrapper {
            position: relative;
            height: 400px;
        }
        .summary {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table { width: 100%; border-collapse: collapse; }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th { background: #f9f9f9; font-weight: 600; }
        tr:hover { background: #fafafa; }
        .best { background: #FFF3E0; font-weight: bold; }
        .running {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .status-running {
            background: #E3F2FD;
            color: #1976D2;
        }
        .status-complete {
            background: #E8F5E9;
            color: #388E3C;
        }
        .legend-toggle {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            border: 1px solid #ddd;
            transition: all 0.2s;
        }
        .legend-item:hover { background: #f5f5f5; }
        .legend-item.hidden { opacity: 0.4; text-decoration: line-through; }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        .group-toggles {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
            padding: 10px;
            background: #fafafa;
            border-radius: 6px;
        }
        .group-btn {
            padding: 6px 12px;
            border: 2px solid #ddd;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .group-btn:hover { border-color: #999; }
        .group-btn.active {
            background: #FF9800;
            border-color: #FF9800;
            color: white;
        }
        .group-btn.partial {
            background: #FFF3E0;
            border-color: #FF9800;
        }
        .toggle-all-btns {
            display: flex;
            gap: 5px;
            margin-left: auto;
        }
        .toggle-all-btns button {
            padding: 4px 10px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <h1>SC-TRM Benchmark Results</h1>

    <div class="controls">
        <button onclick="refreshData()">Refresh Data</button>
        <button class="secondary" onclick="resetZoom()">Reset Zoom</button>
        <button class="danger" onclick="toggleAutoRefresh()">
            <span id="autoRefreshBtn">Pause Auto-Refresh</span>
        </button>
        <select id="runSelect" onchange="refreshData()" style="padding: 10px; border-radius: 4px; border: 1px solid #ccc;">
            <option value=".">Loading runs...</option>
        </select>
        <span class="status" id="status"></span>
        <span class="refresh-time" id="refreshTime">Last refresh: -</span>
    </div>

    <div class="summary">
        <h3>Results Summary</h3>
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Progress</th>
                    <th>Cell Accuracy</th>
                    <th>Training Time</th>
                    <th>Efficiency</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <div class="graphs">
        <div class="graph-container">
            <div class="chart-header">
                <h3>Cell Accuracy Learning Curves</h3>
                <button onclick="resetZoom()">Reset Zoom</button>
            </div>
            <p class="graph-hint">Scroll to zoom, drag to pan. Click legend items to filter.</p>
            <div class="chart-wrapper">
                <canvas id="learningChart"></canvas>
            </div>
        </div>

        <div class="graph-container">
            <div class="chart-header">
                <h3>Exact Accuracy Learning Curves</h3>
            </div>
            <p class="graph-hint">Percentage of fully correct puzzles over time.</p>
            <div class="chart-wrapper">
                <canvas id="exactLineChart"></canvas>
            </div>
        </div>

        <div class="graph-container">
            <h3>Time vs Accuracy</h3>
            <p class="graph-hint">Hover for details. Shows final accuracy vs training time.</p>
            <div class="chart-wrapper">
                <canvas id="scatterChart"></canvas>
            </div>
        </div>

        <div class="graph-container">
            <h3>Cell vs Exact Accuracy</h3>
            <p class="graph-hint">Correlation between individual cell accuracy and full puzzle solution rate.</p>
            <div class="chart-wrapper">
                <canvas id="exactChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const MODEL_COLORS = {{ model_colors | tojson }};
        const MODEL_LABELS = {{ model_labels | tojson }};
        const MODEL_GROUPS = {{ model_groups | tojson }};

        // Reverse mapping: model -> group
        const MODEL_TO_GROUP = {};
        Object.entries(MODEL_GROUPS).forEach(([group, models]) => {
            models.forEach(m => MODEL_TO_GROUP[m] = group);
        });

        let learningChart, scatterChart, exactChart;
        let autoRefresh = true;
        let refreshInterval;
        let hiddenDatasets = new Set();
        let hiddenGroups = new Set();
        let availableModels = new Set();  // Track which models have data

        function initCharts() {
            const learningCtx = document.getElementById('learningChart').getContext('2d');
            learningChart = new Chart(learningCtx, {
                type: 'line',
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                title: (items) => `Epoch ${items[0].label}`,
                                label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`
                            }
                        },
                        zoom: {
                            zoom: {
                                wheel: { enabled: true },
                                pinch: { enabled: true },
                                mode: 'xy',
                            },
                            pan: {
                                enabled: true,
                                mode: 'xy',
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Epoch' },
                            min: 0
                        },
                        y: {
                            title: { display: true, text: 'Cell Accuracy (%)' },
                            min: 0,
                            max: 100
                        }
                    }
                }
            });

            const scatterCtx = document.getElementById('scatterChart').getContext('2d');
            scatterChart = new Chart(scatterCtx, {
                type: 'scatter',
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'right' },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.y.toFixed(1)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Training Time (s)' },
                            min: 0
                        },
                        y: {
                            title: { display: true, text: 'Final Cell Accuracy (%)' },
                            min: 0, max: 100
                        }
                    }
                }
            });

            const exactLineCtx = document.getElementById('exactLineChart').getContext('2d');
            exactLineChart = new Chart(exactLineCtx, {
                type: 'line',
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'right' }, // No click handler for now or reuse?
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.y.toFixed(1)}%`
                            }
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Epoch' } },
                        y: { title: { display: true, text: 'Exact Accuracy (%)' }, min: 0, max: 100 }
                    }
                }
            });

            const exactCtx = document.getElementById('exactChart').getContext('2d');
            exactChart = new Chart(exactCtx, {
                type: 'scatter',
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'right' },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `${ctx.dataset.label}: Cell=${ctx.raw.x.toFixed(1)}%, Exact=${ctx.raw.y.toFixed(1)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Final Cell Accuracy (%)' },
                            min: 0, max: 100
                        },
                        y: {
                            title: { display: true, text: 'Final Exact Accuracy (%)' },
                            min: 0, max: 100
                        }
                    }
                }
            });
        }

        function updateLegend(datasets) {
            const legend = document.getElementById('learningLegend');
            legend.innerHTML = '';
            datasets.forEach((ds, i) => {
                const item = document.createElement('div');
                item.className = 'legend-item' + (hiddenDatasets.has(ds.label) ? ' hidden' : '');
                item.innerHTML = `<div class="legend-color" style="background:${ds.borderColor}"></div>${ds.label}`;
                item.onclick = () => toggleDataset(ds.label, i);
                legend.appendChild(item);
            });
        }

        function updateGroupButtons() {
            const container = document.getElementById('groupButtons');
            container.innerHTML = '';

            Object.entries(MODEL_GROUPS).forEach(([groupName, models]) => {
                // Check if any models in this group have data
                const groupModelsWithData = models.filter(m => availableModels.has(m));
                if (groupModelsWithData.length === 0) return;  // Skip empty groups

                // Check visibility state
                const visibleCount = groupModelsWithData.filter(m => {
                    const label = MODEL_LABELS[m] || m;
                    return !hiddenDatasets.has(label);
                }).length;

                const btn = document.createElement('button');
                btn.className = 'group-btn';
                if (visibleCount === groupModelsWithData.length) {
                    btn.classList.add('active');
                } else if (visibleCount > 0) {
                    btn.classList.add('partial');
                }
                btn.textContent = `${groupName} (${visibleCount}/${groupModelsWithData.length})`;
                btn.onclick = () => toggleGroup(groupName);
                container.appendChild(btn);
            });
        }

        function toggleGroup(groupName) {
            const models = MODEL_GROUPS[groupName] || [];
            const groupModelsWithData = models.filter(m => availableModels.has(m));

            // Check if all visible
            const allVisible = groupModelsWithData.every(m => {
                const label = MODEL_LABELS[m] || m;
                return !hiddenDatasets.has(label);
            });

            // Toggle: if all visible, hide all; otherwise show all
            groupModelsWithData.forEach(m => {
                const label = MODEL_LABELS[m] || m;
                if (allVisible) {
                    hiddenDatasets.add(label);
                } else {
                    hiddenDatasets.delete(label);
                }
            });

            // Update chart
            learningChart.data.datasets.forEach((ds, i) => {
                ds.hidden = hiddenDatasets.has(ds.label);
            });
            learningChart.update();
            updateLegend(learningChart.data.datasets);
            updateGroupButtons();
        }

        function showAllGroups() {
            hiddenDatasets.clear();
            learningChart.data.datasets.forEach(ds => ds.hidden = false);
            learningChart.update();
            updateLegend(learningChart.data.datasets);
            updateGroupButtons();
        }

        function hideAllGroups() {
            learningChart.data.datasets.forEach(ds => {
                hiddenDatasets.add(ds.label);
                ds.hidden = true;
            });
            learningChart.update();
            updateLegend(learningChart.data.datasets);
            updateGroupButtons();
        }

        function toggleDataset(label, index) {
            if (hiddenDatasets.has(label)) {
                hiddenDatasets.delete(label);
            } else {
                hiddenDatasets.add(label);
            }
            learningChart.data.datasets[index].hidden = hiddenDatasets.has(label);
            learningChart.update();
            updateLegend(learningChart.data.datasets);
            updateGroupButtons();
        }

        async function loadRuns() {
            try {
                const resp = await fetch('/api/runs');
                const runs = await resp.json();
                const select = document.getElementById('runSelect');
                select.innerHTML = '';
                runs.forEach(run => {
                    const opt = document.createElement('option');
                    opt.value = run.id;
                    opt.textContent = run.name;
                    select.appendChild(opt);
                });
            } catch (e) {
                console.error("Failed to load runs", e);
            }
        }

        async function refreshData() {
            const runId = document.getElementById('runSelect').value || '.';
            document.getElementById('status').textContent = 'Loading...';
            try {
                const resp = await fetch(`/api/data?run=${encodeURIComponent(runId)}`);
                const data = await resp.json();

                // Track available models
                availableModels.clear();
                data.learning_curves.forEach(lc => availableModels.add(lc.model));

                // Update learning curves
                const datasets = data.learning_curves.map(lc => ({
                    label: MODEL_LABELS[lc.model] || lc.model,
                    modelKey: lc.model,  // Store original key for group lookup
                    data: lc.epochs.map((e, i) => ({ x: e, y: lc.accuracy[i] })),
                    borderColor: MODEL_COLORS[lc.model] || '#333',
                    backgroundColor: MODEL_COLORS[lc.model] || '#333',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.1,
                    hidden: hiddenDatasets.has(MODEL_LABELS[lc.model] || lc.model)
                }));

                learningChart.data.datasets = datasets;
                learningChart.update();
                updateLegend(datasets);
                updateGroupButtons();

                // Update scatter plot
                const scatterDatasets = data.scatter_points.map(sp => ({
                    label: MODEL_LABELS[sp.model] || sp.model,
                    data: [{ x: sp.time, y: sp.accuracy }],
                    backgroundColor: MODEL_COLORS[sp.model] || '#333',
                    borderColor: MODEL_COLORS[sp.model] || '#333',
                    pointRadius: 10,
                    pointHoverRadius: 14,
                }));

                scatterChart.update();

                // Update exact line chart
                const exactLineDatasets = data.learning_curves.map(lc => ({
                    label: MODEL_LABELS[lc.model] || lc.model,
                    modelKey: lc.model,
                    data: lc.epochs.map((e, i) => ({ x: e, y: lc.exact_accuracy ? lc.exact_accuracy[i] : 0 })),
                    borderColor: MODEL_COLORS[lc.model] || '#333',
                    backgroundColor: MODEL_COLORS[lc.model] || '#333',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.1,
                    hidden: hiddenDatasets.has(MODEL_LABELS[lc.model] || lc.model)
                }));
                
                exactLineChart.data.datasets = exactLineDatasets;
                exactLineChart.update();

                // Update exact scatter chart
                const exactDatasets = data.results.map(r => ({
                    label: MODEL_LABELS[r.model] || r.model,
                    data: [{ x: r.accuracy, y: r.exact_accuracy }],
                    backgroundColor: MODEL_COLORS[r.model] || '#333',
                    borderColor: MODEL_COLORS[r.model] || '#333',
                    pointRadius: 10,
                    pointHoverRadius: 14,
                }));
                exactChart.data.datasets = exactDatasets;
                exactChart.update();

                // Update table
                updateTable(data.results);

                document.getElementById('status').textContent = '';
                document.getElementById('refreshTime').textContent =
                    'Last refresh: ' + new Date().toLocaleTimeString();
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }

        function updateTable(results) {
            const tbody = document.querySelector('#resultsTable tbody');
            tbody.innerHTML = '';

            results.sort((a, b) => b.accuracy - a.accuracy);
            const bestAcc = results[0]?.accuracy || 0;

            results.forEach(r => {
                const isBest = r.accuracy === bestAcc && r.status !== 'running';
                const tr = document.createElement('tr');
                tr.className = (isBest ? 'best ' : '') + (r.status === 'running' ? 'running' : '');

                const efficiency = r.time > 0 ? (r.accuracy / r.time).toFixed(2) + '%/s' : 'N/A';
                const progress = r.current_epoch + '/' + r.total_epochs;
                const statusBadge = r.status === 'running'
                    ? '<span class="status-badge status-running">Running</span>'
                    : '<span class="status-badge status-complete">Complete</span>';

                tr.innerHTML = `
                    <td><strong>${MODEL_LABELS[r.model] || r.model}</strong></td>
                    <td>${statusBadge}</td>
                    <td>${progress}</td>
                    <td>${r.accuracy.toFixed(1)}%</td>
                    <td>${r.time.toFixed(1)}s</td>
                    <td>${efficiency}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function resetZoom() {
            learningChart.resetZoom();
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            document.getElementById('autoRefreshBtn').textContent =
                autoRefresh ? 'Pause Auto-Refresh' : 'Resume Auto-Refresh';

            if (autoRefresh) {
                refreshInterval = setInterval(refreshData, 5000);
            } else {
                clearInterval(refreshInterval);
            }
        }

        // Initialize
        initCharts();
        loadRuns().then(() => refreshData());
        refreshInterval = setInterval(refreshData, 5000);
    </script>
</body>
</html>
"""
@app.route('/')
def index():
    """Serve the single-page app."""
    return render_template_string(HTML_TEMPLATE,
                                model_colors=MODEL_COLORS,
                                model_labels=MODEL_LABELS,
                                model_groups=MODEL_GROUPS)


@app.route('/api/runs')
def api_runs():
    """List available benchmark runs (subdirectories + root)."""
    runs = [{'id': '.', 'name': 'Current Run (Active)'}]
    
    # Walk subdirectories
    for root, dirs, files in os.walk(RESULTS_DIR):
        for d in dirs:
            rel_path = os.path.relpath(os.path.join(root, d), RESULTS_DIR)
            if 'figures' in rel_path: continue
            runs.append({'id': rel_path, 'name': rel_path})
            
    return jsonify(runs)

@app.route('/api/data')
def api_data():
    """Load data for a specific run directory."""
    run_path = request.args.get('run', '.')
    target_dir = os.path.join(RESULTS_DIR, run_path)
    
    # Security check: ensure target_dir is inside RESULTS_DIR
    if not os.path.abspath(target_dir).startswith(os.path.abspath(RESULTS_DIR)):
        return jsonify({'error': 'Invalid path'}), 403
        
    return jsonify(load_all_data(target_dir))

def load_all_data(directory):
    """Load all results data from a specific directory."""
    results = []
    learning_curves = []
    scatter_points = []

    if not os.path.exists(directory):
        return {'results': [], 'learning_curves': [], 'scatter_points': []}

    for filename in os.listdir(directory):
        if not (filename.startswith('results_') and filename.endswith('.json')):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
        except:
            continue

        model = data.get('model_name', 'unknown')
        # Handle cases where final_cell_accuracy might be missing in early partial saves
        acc = data.get('final_cell_accuracy', 0)
        if acc is None: acc = 0
        acc = acc * 100
        
        time_s = data.get('training_time_seconds', 0)
        status = data.get('status', 'complete')
        current_epoch = data.get('current_epoch', data.get('epochs', 0))
        total_epochs = data.get('epochs', 0)
        history = data.get('train_history', [])

        results.append({
            'model': model,
            'accuracy': acc,
            'exact_accuracy': data.get('final_exact_accuracy', 0) * 100,
            'time': time_s,
            'status': status,
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
        })

        if history:
            learning_curves.append({
                'model': model,
                'epochs': [h['epoch'] for h in history],
                'accuracy': [h['cell_accuracy'] * 100 for h in history],
                'exact_accuracy': [h.get('exact_accuracy', 0) * 100 for h in history],
            })

        scatter_points.append({
            'model': model,
            'time': time_s,
            'accuracy': acc,
        })

    return {
        'results': results,
        'learning_curves': learning_curves,
        'scatter_points': scatter_points,
    }

if __name__ == '__main__':
    print("Starting SC-TRM Interactive Graph Viewer...")
    print("Open http://localhost:5050 in your browser")
    from flask import request
    app.run(host='0.0.0.0', port=5050, debug=True)
