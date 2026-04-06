const METHOD_COLORS = {
    explicit:    { pos: "#4a90d9", neg: "#e8c06a", boundary: "#2c5f9a" },
    kernel:      { pos: "#e87c4a", neg: "#6ac4e8", boundary: "#9a4a1e" },
    reuploading: { pos: "#5ab45a", neg: "#d46ab4", boundary: "#2a7a2a" },
};

const PLOTLY_BASE = {
    responsive: true,
    displayModeBar: false,
};

function buildScatterTraces(points) {
    if (!points?.length) return [];
    const cls0x = [], cls0y = [], cls1x = [], cls1y = [];
    for (const p of points) {
        if (p.label === 0) { cls0x.push(p.x); cls0y.push(p.y); }
        else               { cls1x.push(p.x); cls1y.push(p.y); }
    }
    return [
        {
            x: cls0x, y: cls0y, mode: "markers", type: "scatter",
            marker: { color: "#1565c0", size: 5, symbol: "circle",
                      line: { color: "white", width: 0.5 } },
            name: "Class 0", showlegend: false,
        },
        {
            x: cls1x, y: cls1y, mode: "markers", type: "scatter",
            marker: { color: "#c62828", size: 5, symbol: "square",
                      line: { color: "white", width: 0.5 } },
            name: "Class 1", showlegend: false,
        },
    ];
}

/**
 * Render a single decision-boundary panel.
 *
 * @param {HTMLElement} container
 * @param {object} heatmap  { x: number[], y: number[], z: number[][] }
 *                          z values are probabilities in [0,1] for class 1
 * @param {object[]} points [{ x, y, label }] scatter points
 * @param {string} method   "explicit" | "kernel" | "reuploading"
 */
export function renderBoundaryPlot(container, heatmap, points, method) {
    if (!container) return;

    const colors = METHOD_COLORS[method] ?? METHOD_COLORS.reuploading;

    const traces = [];

    if (heatmap?.z?.length) {
        traces.push({
            x: heatmap.x,
            y: heatmap.y,
            z: heatmap.z,
            type: "heatmap",
            colorscale: [
                [0,   colors.neg],
                [0.5, "#f8f4ee"],
                [1,   colors.pos],
            ],
            zmin: 0, zmax: 1,
            showscale: false,
            hoverinfo: "skip",
        });
    } else {
        // Placeholder while no data is loaded
        traces.push({
            x: [-1, 1], y: [-1, 1],
            mode: "text",
            text: ["No data"],
            textfont: { color: "#aaa", size: 11 },
            type: "scatter",
            showlegend: false,
        });
    }

    traces.push(...buildScatterTraces(points));

    const layout = {
        margin: { t: 0, b: 0, l: 0, r: 0 },
        xaxis: { visible: false, fixedrange: true },
        yaxis: { visible: false, fixedrange: true, scaleanchor: "x" },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor:  "rgba(0,0,0,0)",
        dragmode: false,
    };

    if (container._hasPlot) {
        Plotly.react(container, traces, layout, PLOTLY_BASE);
    } else {
        Plotly.newPlot(container, traces, layout, PLOTLY_BASE);
        container._hasPlot = true;
    }
}

/**
 * Render the accuracy / loss curves.
 *
 * @param {object[]} steps   Array of step objects from timeline_steps
 * @param {number}   current Index of current step (highlighted)
 */
export function renderAccuracyChart(steps, current) {
    const lossChart = document.getElementById("loss-chart");
    if (!lossChart || !steps?.length) return;

    const epochs      = steps.map((s) => s.global_step ?? s.epoch ?? 0);
    const trainAcc    = steps.map((s) => s.train_acc  ?? null);
    const testAcc     = steps.map((s) => s.test_acc   ?? null);
    const trainLoss   = steps.map((s) => s.train_loss ?? null);
    const testLoss    = steps.map((s) => s.test_loss  ?? null);

    const curEpoch = steps[current]?.global_step ?? steps[current]?.epoch ?? null;

    const shapes = curEpoch !== null ? [{
        type: "line", xref: "x", yref: "paper",
        x0: curEpoch, x1: curEpoch, y0: 0, y1: 1,
        line: { color: "rgba(13,143,113,0.5)", width: 1.5, dash: "dot" },
    }] : [];

    const traces = [
        { x: epochs, y: trainAcc,  name: "Train acc",  mode: "lines", line: { color: "#0d8f71", width: 2 } },
        { x: epochs, y: testAcc,   name: "Test acc",   mode: "lines", line: { color: "#ef8354", width: 2 } },
        { x: epochs, y: trainLoss, name: "Train loss", mode: "lines", line: { color: "#0d8f71", width: 1.5, dash: "dot" } },
        { x: epochs, y: testLoss,  name: "Test loss",  mode: "lines", line: { color: "#ef8354", width: 1.5, dash: "dot" } },
    ].filter((t) => t.y.some((v) => v !== null));

    const layout = {
        margin: { t: 8, b: 36, l: 42, r: 16 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor:  "rgba(0,0,0,0)",
        xaxis: { title: "Step", gridcolor: "rgba(23,33,29,0.08)", zeroline: false },
        yaxis: { gridcolor: "rgba(23,33,29,0.08)", zeroline: false },
        legend: { orientation: "h", y: -0.22, font: { size: 11 } },
        shapes,
        font: { family: "IBM Plex Sans, sans-serif", size: 11 },
    };

    if (lossChart._hasPlot) {
        Plotly.react(lossChart, traces, layout, PLOTLY_BASE);
    } else {
        Plotly.newPlot(lossChart, traces, layout, PLOTLY_BASE);
        lossChart._hasPlot = true;
    }
}

export function renderEmptyState() {
    const chartEmpty = document.getElementById("chart-empty");
    if (chartEmpty) {
        chartEmpty.hidden = false;
        chartEmpty.style.display = "flex";
    }
    const lossChart = document.getElementById("loss-chart");
    if (lossChart) lossChart.style.display = "none";
}
