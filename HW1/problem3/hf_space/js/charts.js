const PLOTLY_BASE = {
    responsive: true,
    displayModeBar: false,
};

const MLP_COLOR = "#5b50c8";
const QNN_COLOR = "#ef8354";

const CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];

/**
 * Render overlaid training curves for MLP vs QNN.
 */
export function renderTrainingCurves(steps, current) {
    const lossChart = document.getElementById("loss-chart");
    if (!lossChart || !steps?.length) return;

    const epochs = steps.map((s) => s.epoch ?? s.global_step ?? 0);

    const traces = [
        {
            x: epochs, y: steps.map((s) => s.mlp_train_acc ?? null),
            name: "MLP train acc", mode: "lines",
            line: { color: MLP_COLOR, width: 2 },
        },
        {
            x: epochs, y: steps.map((s) => s.mlp_test_acc ?? null),
            name: "MLP test acc", mode: "lines",
            line: { color: MLP_COLOR, width: 2, dash: "dot" },
        },
        {
            x: epochs, y: steps.map((s) => s.qnn_train_acc ?? null),
            name: "QNN train acc", mode: "lines",
            line: { color: QNN_COLOR, width: 2 },
        },
        {
            x: epochs, y: steps.map((s) => s.qnn_test_acc ?? null),
            name: "QNN test acc", mode: "lines",
            line: { color: QNN_COLOR, width: 2, dash: "dot" },
        },
    ].filter((t) => t.y.some((v) => v !== null));

    const curEpoch = steps[current]?.epoch ?? null;
    const shapes = curEpoch !== null ? [{
        type: "line", xref: "x", yref: "paper",
        x0: curEpoch, x1: curEpoch, y0: 0, y1: 1,
        line: { color: "rgba(13,143,113,0.5)", width: 1.5, dash: "dot" },
    }] : [];

    const layout = {
        margin: { t: 8, b: 36, l: 42, r: 16 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Epoch", gridcolor: "rgba(23,33,29,0.08)", zeroline: false },
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

/**
 * Render a confusion matrix heatmap.
 *
 * @param {HTMLElement} container
 * @param {number[][]} matrix  10x10 confusion matrix (row=true, col=predicted)
 * @param {string} method  "mlp" or "qnn"
 */
export function renderConfusionMatrix(container, matrix, method) {
    if (!container) return;

    if (!matrix?.length) {
        if (!container._hasPlot) {
            Plotly.newPlot(container, [{
                x: [0, 1], y: [0, 1],
                mode: "text", text: ["No data"],
                textfont: { color: "#aaa", size: 11 },
                type: "scatter", showlegend: false,
            }], {
                margin: { t: 0, b: 0, l: 0, r: 0 },
                xaxis: { visible: false }, yaxis: { visible: false },
                paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
            }, PLOTLY_BASE);
            container._hasPlot = true;
        }
        return;
    }

    // Normalize rows to percentages for better visualization
    const normalized = matrix.map((row) => {
        const total = row.reduce((a, b) => a + b, 0);
        return total > 0 ? row.map((v) => v / total) : row;
    });

    const color = method === "mlp" ? MLP_COLOR : QNN_COLOR;

    const traces = [{
        z: normalized,
        x: CLASS_NAMES,
        y: CLASS_NAMES,
        type: "heatmap",
        colorscale: [
            [0, "#f8f4ee"],
            [0.5, method === "mlp" ? "rgba(91,80,200,0.35)" : "rgba(239,131,84,0.35)"],
            [1, color],
        ],
        zmin: 0, zmax: 1,
        showscale: false,
        hovertemplate: "True: %{y}<br>Pred: %{x}<br>Rate: %{z:.2f}<extra></extra>",
        // Show raw counts as text
        text: matrix.map((row) => row.map((v) => String(v))),
        texttemplate: "%{text}",
        textfont: { size: 9 },
    }];

    const layout = {
        margin: { t: 4, b: 60, l: 60, r: 4 },
        xaxis: {
            title: "Predicted",
            tickangle: -45,
            gridcolor: "rgba(23,33,29,0.08)",
            side: "bottom",
        },
        yaxis: {
            title: "True",
            gridcolor: "rgba(23,33,29,0.08)",
            autorange: "reversed",
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "IBM Plex Sans, sans-serif", size: 10 },
    };

    if (container._hasPlot) {
        Plotly.react(container, traces, layout, PLOTLY_BASE);
    } else {
        Plotly.newPlot(container, traces, layout, PLOTLY_BASE);
        container._hasPlot = true;
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
