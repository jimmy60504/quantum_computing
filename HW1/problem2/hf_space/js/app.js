import {
    pageTitle, pageSubtitle, exportStatus,
    runSelect, runNote, resultsTableBody,
    stepSlider, currentStepLabel,
    timelineCaption, trainAccPill, testAccPill, chartEmpty,
    experimentMeta, datasetsImage,
    boundaryPlots, accPills,
    lossChart,
    state,
} from "./dom.js";
import { bindImageLightbox, bindAnalysisModal, maybeShowAnalysisHint } from "./overlays.js";
import { renderBoundaryPlot, renderAccuracyChart, renderEmptyState } from "./charts.js";
import {
    formatMetric, formatInteger, appendMetaRow, withCacheBust,
    setLoadingState, loadManifest, loadRunData, loadRuntimeSource,
} from "./data.js";

const METHODS   = ["explicit", "kernel", "reuploading"];
const DATASETS  = ["circle", "moons"];

// ── helpers ───────────────────────────────────────────────────────────────────

function formatAcc(v) {
    if (v === null || v === undefined) return "—";
    return (v * 100).toFixed(1) + "%";
}

function renderResultsTable(runs, selectedRunId) {
    if (!resultsTableBody) return;
    resultsTableBody.innerHTML = "";
    runs.forEach((run) => {
        const row = document.createElement("tr");
        row.dataset.runId = run.id;
        if (run.id === selectedRunId) row.classList.add("is-selected");

        const layerText = (run.num_layers_explicit !== undefined || run.num_layers_reuploading !== undefined)
            ? `E${formatInteger(run.num_layers_explicit)} / R${formatInteger(run.num_layers_reuploading)}`
            : formatInteger(run.num_layers);

        const values = [
            { text: formatAcc(run.best_test_acc ?? run.final_test_acc), className: "metric-cell metric-cell-strong" },
            { text: (run.methods ?? [run.method]).filter(Boolean).join(", ") || "—" },
            { text: (run.datasets ?? [run.dataset]).filter(Boolean).join(" + ") || "—" },
            { text: run.label || run.id, className: "run-cell" },
            { text: formatInteger(run.num_qubits) },
            { text: layerText },
        ];

        values.forEach(({ text, className }) => {
            const cell = document.createElement("td");
            cell.textContent = text;
            if (className) cell.className = className;
            row.appendChild(cell);
        });

        row.addEventListener("click", async () => {
            runSelect.value = run.id;
            await applyRun(run.id);
        });

        resultsTableBody.appendChild(row);
    });
}

function updateAccPills(step) {
    METHODS.forEach((method, mi) => {
        DATASETS.forEach((dataset, di) => {
            const pill = accPills[mi]?.[di];
            if (!pill) return;
            const acc = step?.accuracies?.[method]?.[dataset] ?? null;
            pill.textContent = formatAcc(acc);
        });
    });
}

// ── step rendering ─────────────────────────────────────────────────────────────

async function refreshStepState(data, index, stepToken) {
    const steps = data.timeline_steps || [];
    if (!steps.length) {
        renderEmptyState();
        return;
    }

    const step = steps[index];
    if (currentStepLabel) currentStepLabel.textContent = step.label || `Epoch ${step.epoch ?? index}`;
    if (timelineCaption)  timelineCaption.hidden = true;

    if (trainAccPill) {
        trainAccPill.textContent = `Train ${formatAcc(step.train_acc)}`;
        trainAccPill.hidden = false;
    }
    if (testAccPill) {
        testAccPill.textContent = `Test ${formatAcc(step.test_acc)}`;
        testAccPill.hidden = false;
    }

    renderAccuracyChart(steps, index);
    updateAccPills(step);

    if (stepToken !== state.activeStepToken) return;

    // Decision boundaries
    METHODS.forEach((method, mi) => {
        DATASETS.forEach((dataset, di) => {
            const container = boundaryPlots[mi]?.[di];
            if (!container) return;
            const heatmap = step?.boundaries?.[method]?.[dataset] ?? null;
            const points  = step?.scatter?.[dataset] ?? null;
            renderBoundaryPlot(container, heatmap, points, method);
        });
    });

    if (chartEmpty) chartEmpty.hidden = true;
    if (lossChart)  lossChart.style.display = "";
    setLoadingState({ visible: false });
}

// ── run loading ────────────────────────────────────────────────────────────────

function populateExperimentMeta(data, run) {
    if (!experimentMeta) return;
    experimentMeta.innerHTML = "";
    appendMetaRow("Model",   data.experiment?.model);
    appendMetaRow("Task",    data.experiment?.task);
    appendMetaRow("Methods", (data.experiment?.methods ?? []).join(", "));
    appendMetaRow("Datasets",(data.experiment?.datasets ?? []).join(", "));
    appendMetaRow("Device",  data.experiment?.device);
    if (run?.num_qubits !== undefined) appendMetaRow("Qubits",  run.num_qubits);
    if (run?.num_layers_explicit !== undefined || run?.num_layers_reuploading !== undefined) {
        appendMetaRow(
            "Layers (explicit / reupload)",
            `${formatInteger(run?.num_layers_explicit)} / ${formatInteger(run?.num_layers_reuploading)}`,
        );
    } else if (run?.num_layers !== undefined) {
        appendMetaRow("Layers",  run.num_layers);
    }
    if (run?.num_params !== undefined) appendMetaRow("Params",  formatInteger(run.num_params));
    if (run?.train_time !== undefined) appendMetaRow("Train time", run.train_time);
    appendMetaRow("Note", data.experiment?.note);
}

async function applyRun(runId) {
    state.currentRunId = runId;
    state.currentRunChunkCache = {};
    state.currentRunChunkInflight = {};
    const loadToken = ++state.activeLoadToken;

    const selectedRun =
        state.currentManifest.runs.find((r) => r.id === runId) ||
        state.currentManifest.runs[0];

    setLoadingState({
        visible: true,
        label: `Preparing ${selectedRun.label}`,
        percent: 20,
        status: "loading",
    });

    state.currentData = await loadRunData(selectedRun.path, loadToken);
    if (loadToken !== state.activeLoadToken) return;

    setLoadingState({ visible: true, label: `Rendering ${selectedRun.label}`, percent: 82, status: "rendering" });

    pageTitle.textContent    = state.currentData.title;
    pageSubtitle.textContent = state.currentData.subtitle;
    exportStatus.textContent = state.currentData.status;
    runNote.textContent      = `Loaded ${selectedRun.label} with ${selectedRun.steps} exported steps.`;

    renderResultsTable(state.currentManifest.runs, selectedRun.id);

    if (datasetsImage) {
        const src = state.currentData.assets?.datasets_overview;
        if (src) {
            datasetsImage.src = withCacheBust(src);
        } else {
            datasetsImage.removeAttribute("src");
        }
    }

    populateExperimentMeta(state.currentData, selectedRun);

    const steps = state.currentData.timeline_steps || [];
    if (steps.length > 1) {
        stepSlider.disabled = false;
        stepSlider.min   = "0";
        stepSlider.max   = String(steps.length - 1);
        stepSlider.value = "0";
    } else {
        stepSlider.disabled = true;
        stepSlider.min = stepSlider.max = stepSlider.value = "0";
    }

    await refreshStepState(state.currentData, 0, ++state.activeStepToken);
    setLoadingState({ visible: false, label: "Viewer ready", percent: 100, status: "ready" });
}

// ── main ───────────────────────────────────────────────────────────────────────

async function main() {
    bindImageLightbox();
    bindAnalysisModal();
    maybeShowAnalysisHint();

    setLoadingState({ visible: true, label: "Booting viewer", percent: 2, status: "loading" });
    await loadRuntimeSource();
    state.currentManifest = await loadManifest();

    const runs = state.currentManifest.runs || [];
    runSelect.innerHTML = "";
    runs.forEach((run) => {
        const opt = document.createElement("option");
        opt.value       = run.id;
        opt.textContent = run.label;
        runSelect.appendChild(opt);
    });

    if (!runs.length) throw new Error("Viewer manifest contains no runs.");

    const defaultRunId = state.currentManifest.default_run || runs[0].id;
    runSelect.value = defaultRunId;
    renderResultsTable(runs, defaultRunId);
    await applyRun(defaultRunId);

    runSelect.addEventListener("change", async (e) => applyRun(e.target.value));

    stepSlider.addEventListener("input", async (e) => {
        if (state.currentData) {
            await refreshStepState(
                state.currentData,
                Number(e.target.value),
                ++state.activeStepToken,
            );
        }
    });
}

main().catch((error) => {
    console.error(error);
    pageSubtitle.textContent = "Failed to load static export.";
    runNote.textContent      = "Unable to load a viewer manifest.";
    if (chartEmpty) {
        chartEmpty.textContent  = "The static viewer failed to load its export data.";
        chartEmpty.hidden       = false;
        chartEmpty.style.display = "flex";
    }
    setLoadingState({ visible: true, label: "Load failed", percent: 100, status: "error" });
});
