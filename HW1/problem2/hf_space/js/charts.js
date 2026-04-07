const METHOD_COLORS = {
    explicit:    { pos: "#4a90d9", neg: "#e8c06a" },
    kernel:      { pos: "#e87c4a", neg: "#6ac4e8" },
    reuploading: { pos: "#5ab45a", neg: "#d46ab4" },
};

const PLOTLY_BASE = {
    responsive: true,
    displayModeBar: false,
};

// Slightly-tilted isometric view, zoomed in
export const DEFAULT_CAMERA = {
    eye:    { x: 0.8, y: -0.8, z: 0.65 },
    up:     { x: 0,   y: 0,    z: 1    },
    center: { x: 0,   y: 0,    z: 0    },
};

// Moons home — same tilt but rotated 90° clockwise in XY
export const MOONS_DEFAULT_CAMERA = {
    eye:    { x: -0.8, y: -0.8, z: 0.65 },
    up:     { x: 0,    y: 0,    z: 1    },
    center: { x: 0,    y: 0,    z: 0    },
};

// Top-down view for Circle — up matches Circle home direction (0.8, -0.8) → up = (-√2/2, √2/2)
export const TOP_CAMERA = {
    eye:    { x: 0,      y: 0,     z: 1.4 },
    up:     { x: -0.707, y: 0.707, z: 0   },
    center: { x: 0,      y: 0,     z: 0   },
};

// Top-down view for Moons — up rotated 45° CW to match Moons home direction
export const MOONS_TOP_CAMERA = {
    eye:    { x: 0,     y: 0,     z: 1.4 },
    up:     { x: 0.707, y: 0.707, z: 0   },
    center: { x: 0,     y: 0,     z: 0   },
};

/**
 * Render a decision-boundary panel as an interactive 3D surface.
 *
 * @param {HTMLElement} container
 * @param {object|null} heatmap  { x, y, z } — z is P(class 1) in [0, 1]
 * @param {object[]|null} points [{ x, y, label }] scatter points
 * @param {string} method        "explicit" | "kernel" | "reuploading"
 * @param {object|null} camera   Plotly scene.camera object; null → DEFAULT_CAMERA
 */
export function renderBoundarySurface(container, heatmap, points, method, camera) {
    if (!container) return;

    const colors = METHOD_COLORS[method] ?? METHOD_COLORS.reuploading;
    const appliedCamera = camera ?? DEFAULT_CAMERA;
    const traces = [];

    if (heatmap?.z?.length) {
        traces.push({
            type: "surface",
            x: heatmap.x,
            y: heatmap.y,
            z: heatmap.z,
            colorscale: [
                [0,    colors.neg],
                [0.45, "#f8f4ee"],
                [0.55, "#f8f4ee"],
                [1,    colors.pos],
            ],
            cmin: 0, cmax: 1,
            showscale: false,
            opacity: 0.72,
            contours: {
                z: {
                    show: true,
                    start: 0.5, end: 0.5, size: 0.1,
                    color: "rgba(20,20,20,0.85)",
                    width: 3,
                    usecolormap: false,
                },
            },
            hovertemplate: "x: %{x:.2f}<br>y: %{y:.2f}<br>P(1): %{z:.2f}<extra></extra>",
        });
    } else {
        traces.push({
            type: "scatter3d", mode: "text",
            x: [0], y: [0], z: [0.5],
            text: ["No data"],
            textfont: { color: "#aaa", size: 11 },
            showlegend: false,
        });
    }

    // Scatter points floating just above the surface
    if (points?.length) {
        const cls0x = [], cls0y = [], cls1x = [], cls1y = [];
        for (const p of points) {
            if (p.label === 0) { cls0x.push(p.x); cls0y.push(p.y); }
            else               { cls1x.push(p.x); cls1y.push(p.y); }
        }
        const Z = 0.5;
        if (cls0x.length) traces.push({
            type: "scatter3d", mode: "markers",
            x: cls0x, y: cls0y, z: cls0x.map(() => Z),
            marker: { color: "#1565c0", size: 2.5 },
            showlegend: false, hoverinfo: "skip",
        });
        if (cls1x.length) traces.push({
            type: "scatter3d", mode: "markers",
            x: cls1x, y: cls1y, z: cls1x.map(() => Z),
            marker: { color: "#c62828", size: 2.5, symbol: "square" },
            showlegend: false, hoverinfo: "skip",
        });
    }

    const layout = {
        margin: { t: 0, b: 0, l: 0, r: 0 },
        scene: {
            xaxis: { visible: false, showgrid: false },
            yaxis: { visible: false, showgrid: false },
            zaxis: {
                title: "", range: [0, 1.1],
                tickvals: [0, 0.5, 1], ticktext: ["0", "·5", "1"],
                tickfont: { size: 8 },
                gridcolor: "rgba(23,33,29,0.08)",
            },
            camera: appliedCamera,
            dragmode: "turntable",
            aspectmode: "manual",
            aspectratio: { x: 1, y: 1, z: 0.55 },
            bgcolor: "rgba(0,0,0,0)",
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor:  "rgba(0,0,0,0)",
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
