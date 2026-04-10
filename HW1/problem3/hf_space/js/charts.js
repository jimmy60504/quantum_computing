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

const BATCHES_PER_EPOCH = 781;

/**
 * Render overlaid training curves for MLP vs QNN.
 *
 * @param {object[]} epochSteps   - per-epoch timeline records
 * @param {number}   curEpochFrac - current position as epoch fraction (e.g. 3.5)
 * @param {object}   stepAccData  - { mlp:{steps,accs}, qnn:{steps,accs} } | null
 */
export function renderTrainingCurves(epochSteps, curEpochFrac, stepAccData) {
    const lossChart = document.getElementById("loss-chart");
    if (!lossChart || !epochSteps?.length) return;

    const epochs = epochSteps.map((s) => s.epoch ?? s.global_step ?? 0);
    const traces = [];

    // Train acc — epoch-level, dotted + semi-transparent
    for (const [method, color] of [["mlp", MLP_COLOR], ["qnn", QNN_COLOR]]) {
        const y = epochSteps.map((s) => s[`${method}_train_acc`] ?? null);
        if (y.some((v) => v !== null))
            traces.push({
                x: epochs, y,
                name: `${method.toUpperCase()} train`, mode: "lines",
                line: { color, width: 1.5, dash: "dot" }, opacity: 0.5,
            });
    }

    // Test acc — 400-pt checkpoint-level when available, else epoch-level
    for (const [method, color] of [["mlp", MLP_COLOR], ["qnn", QNN_COLOR]]) {
        const sd = stepAccData?.[method];
        if (sd?.steps?.length) {
            traces.push({
                x: sd.steps.map((s) => s / BATCHES_PER_EPOCH),
                y: sd.accs,
                name: `${method.toUpperCase()} test`, mode: "lines",
                line: { color, width: 2 },
            });
        } else {
            const y = epochSteps.map((s) => s[`${method}_test_acc`] ?? null);
            if (y.some((v) => v !== null))
                traces.push({
                    x: epochs, y,
                    name: `${method.toUpperCase()} test`, mode: "lines",
                    line: { color, width: 2 },
                });
        }
    }

    const shapes = curEpochFrac != null ? [{
        type: "line", xref: "x", yref: "paper",
        x0: curEpochFrac, x1: curEpochFrac, y0: 0, y1: 1,
        line: { color: "rgba(13,143,113,0.5)", width: 1.5, dash: "dot" },
    }] : [];

    const layout = {
        margin: { t: 8, b: 36, l: 42, r: 16 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor:  "rgba(0,0,0,0)",
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

// Distinct colors for 10 CIFAR-10 classes (same order as CLASS_NAMES)
const TSNE_CLASS_COLORS = [
    "#e6194b", // airplane   — red
    "#3cb44b", // automobile — green
    "#4363d8", // bird       — blue
    "#f58231", // cat        — orange
    "#911eb4", // deer       — purple
    "#42d4f4", // dog        — cyan
    "#f032e6", // frog       — magenta
    "#bfef45", // horse      — lime
    "#fabed4", // ship       — pink
    "#469990", // truck      — teal
];

/**
 * Render the UMAP / t-SNE feature-space scatter for one method at one training step.
 *
 * Data format (methodData):
 *   coords    : number[][][]  — [n_steps][N][2] 2-D positions (change each frame)
 *   preds     : number[][]   — [n_steps][N] predicted class index at each step
 *   steps     : number[]     — global training step at each frame
 *   reduction : string        — "umap" | "tsne" | "pca"
 *
 * Visualisation (blob → clusters animation):
 *   • Position   = per-step UMAP coords in shared embedding space
 *                  (step 1 ≈ one central blob; final step ≈ 10 class clusters)
 *   • Fill colour = TRUE class (fixed) so you can read which cluster is which
 *   • Accuracy shown in title
 *
 * @param {HTMLElement} container
 * @param {object} methodData
 * @param {Array<{class_idx: number, class_name: string}>} samples
 * @param {number} stepIndex
 * @param {string} method  "mlp" | "qnn"
 */
export function renderTsneChart(container, methodData, samples, stepIndex, method) {
    if (!container || !methodData?.coords?.length) return;

    const coords = methodData.coords[stepIndex];    // [N][2] — changes each frame
    const preds  = methodData.preds?.[stepIndex];   // [N]    — for accuracy label
    if (!coords) return;

    const step = methodData.steps?.[stepIndex] ?? stepIndex;
    const reductionLabel = (methodData.reduction ?? "umap").toUpperCase();
    const N = samples.length;

    // Group by TRUE class — track sample index in customdata for hover preview
    const traces = CLASS_NAMES.map((name, classIdx) => {
        const xs = [], ys = [], customdata = [];
        samples.forEach((s, i) => {
            if (s.class_idx === classIdx && coords[i]) {
                xs.push(coords[i][0]);
                ys.push(coords[i][1]);
                customdata.push(i);   // global sample index → used by hover handler
            }
        });
        return {
            x: xs, y: ys, customdata,
            mode: "markers",
            type: "scatter",
            name,
            marker: {
                color: TSNE_CLASS_COLORS[classIdx],
                size: 7,
                opacity: 0.85,
                line: { width: 0 },
            },
            hoverinfo: "none",   // we handle hover ourselves
        };
    });

    // Class centroid markers (fixed, computed from full test set at final checkpoint)
    const centroids = methodData.class_centroids;
    if (centroids?.length) {
        traces.push({
            x: centroids.map(c => c.x),
            y: centroids.map(c => c.y),
            mode: "markers+text",
            type: "scatter",
            name: "centroids",
            showlegend: false,
            text: centroids.map(c => c.class_name),
            textposition: "top center",
            textfont: { size: 9, color: "#333" },
            marker: {
                symbol: "star",
                size: 14,
                color: centroids.map(c => TSNE_CLASS_COLORS[c.class_idx]),
                line: { color: "#333", width: 1 },
            },
            hovertemplate: "%{text}<extra>centroid</extra>",
        });
    }

    // Accuracy from predicted classes
    let accStr = "";
    if (preds) {
        const correct = preds.filter((p, i) => p === samples[i].class_idx).length;
        accStr = ` | Acc ${(correct / N * 100).toFixed(1)}%`;
    }

    const layout = {
        margin: { t: 36, b: 8, l: 8, r: 8 },
        title: {
            text: `Step ${step}${accStr} &nbsp;(${reductionLabel})`,
            font: { size: 11, color: "#555" },
            x: 0.5,
            xanchor: "center",
            y: 0.99,
            yanchor: "top",
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { visible: false },
        yaxis: { visible: false, scaleanchor: "x" },
        legend: {
            orientation: "h",
            y: -0.02,
            x: 0.5,
            xanchor: "center",
            font: { size: 9 },
            itemsizing: "constant",
        },
        font: { family: "IBM Plex Sans, sans-serif", size: 10 },
    };

    if (container._hasPlot) {
        Plotly.react(container, traces, layout, PLOTLY_BASE);
    } else {
        Plotly.newPlot(container, traces, layout, PLOTLY_BASE);
        container._hasPlot = true;
        _attachTsneHover(container, samples);
    }
}

function _getOrCreatePreview() {
    let el = document.getElementById("tsne-hover-preview");
    if (!el) {
        el = document.createElement("div");
        el.id = "tsne-hover-preview";
        Object.assign(el.style, {
            position: "fixed", pointerEvents: "none", zIndex: "9999",
            display: "none", flexDirection: "column", alignItems: "center",
            background: "#fff", border: "1px solid #ccc", borderRadius: "6px",
            padding: "6px 8px", boxShadow: "0 2px 8px rgba(0,0,0,0.18)",
            fontSize: "11px", color: "#333", lineHeight: "1.4", gap: "3px",
        });
        document.body.appendChild(el);
    }
    return el;
}

function _attachTsneHover(container, samples) {
    const preview = _getOrCreatePreview();

    container.on("plotly_hover", function(data) {
        const pt = data.points[0];
        if (pt.customdata == null) return;   // centroid marker, skip
        const sampleIdx = pt.customdata;
        const s = samples[sampleIdx];
        if (!s?.image_base64) return;

        preview.innerHTML = `
            <img src="${s.image_base64}"
                 style="width:80px;height:80px;image-rendering:pixelated;display:block;">
            <span style="font-weight:600">${s.class_name}</span>
        `;
        preview.style.display = "flex";

        const e = data.event;
        const vw = window.innerWidth, vh = window.innerHeight;
        let x = e.clientX + 16, y = e.clientY - 50;
        if (x + 110 > vw) x = e.clientX - 110;
        if (y + 110 > vh) y = vh - 120;
        if (y < 0) y = 8;
        preview.style.left = x + "px";
        preview.style.top  = y + "px";
    });

    container.on("plotly_unhover", function() {
        preview.style.display = "none";
    });
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
