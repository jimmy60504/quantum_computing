import {
    pageTitle, pageSubtitle, exportStatus,
    runSelect, runNote, resultsTableBody,
    stepSlider, currentStepLabel,
    mlpAccPill, qnnAccPill, chartEmpty,
    experimentMeta, cmPlots, lossChart,
    state,
} from "./dom.js";
import { bindImageLightbox, bindAnalysisModal } from "./overlays.js";
import { renderTrainingCurves, renderConfusionMatrix, renderEmptyState } from "./charts.js";
import {
    formatAcc, formatParams, formatTime, formatInteger,
    appendMetaRow, withCacheBust,
    setLoadingState, loadManifest, loadRunData, loadRuntimeSource,
} from "./data.js";

const METHODS = ["mlp", "qnn"];

// ── comparison table ─────────────────────────────────────────────────────────

function renderComparisonTable(summary) {
    if (!resultsTableBody || !summary) return;
    resultsTableBody.innerHTML = "";

    for (const method of METHODS) {
        const info = summary[method];
        if (!info) continue;

        const row = document.createElement("tr");
        const values = [
            { text: info.label || method, className: "" },
            { text: formatAcc(info.best_test_acc), className: "metric-cell metric-cell-strong" },
            { text: formatParams(info.num_params), className: "metric-cell" },
            { text: formatTime(info.train_time_s), className: "metric-cell" },
        ];

        values.forEach(({ text, className }) => {
            const cell = document.createElement("td");
            cell.textContent = text;
            if (className) cell.className = className;
            row.appendChild(cell);
        });

        resultsTableBody.appendChild(row);
    }
}

// ── step rendering ───────────────────────────────────────────────────────────

function refreshStepState(data, index) {
    const steps = data.timeline_steps || [];
    if (!steps.length) {
        renderEmptyState();
        return;
    }

    const step = steps[index];
    if (currentStepLabel) currentStepLabel.textContent = step.label || `Epoch ${step.epoch ?? index}`;

    if (mlpAccPill) {
        mlpAccPill.textContent = `MLP ${formatAcc(step.mlp_test_acc)}`;
        mlpAccPill.hidden = step.mlp_test_acc === undefined;
    }
    if (qnnAccPill) {
        qnnAccPill.textContent = `QNN ${formatAcc(step.qnn_test_acc)}`;
        qnnAccPill.hidden = step.qnn_test_acc === undefined;
    }

    renderTrainingCurves(steps, index);

    // Confusion matrices
    for (const method of METHODS) {
        const cm = step[`${method}_confusion`] ?? null;
        renderConfusionMatrix(cmPlots[method], cm, method);
    }

    if (chartEmpty) chartEmpty.hidden = true;
    if (lossChart) lossChart.style.display = "";
    setLoadingState({ visible: false });
}

// ── run loading ──────────────────────────────────────────────────────────────

function populateExperimentMeta(data, run) {
    if (!experimentMeta) return;
    experimentMeta.innerHTML = "";
    appendMetaRow("Model", data.experiment?.model);
    appendMetaRow("Task", data.experiment?.task);
    appendMetaRow("Methods", (data.experiment?.methods ?? []).join(", "));
    appendMetaRow("Dataset", data.experiment?.dataset);
    appendMetaRow("Device", data.experiment?.device);
    if (run?.num_qubits !== undefined) appendMetaRow("Qubits", run.num_qubits);
    if (run?.num_layers !== undefined) appendMetaRow("Layers", run.num_layers);
    appendMetaRow("Note", data.experiment?.note);
}

async function applyRun(runId) {
    state.currentRunId = runId;
    const loadToken = ++state.activeLoadToken;

    const selectedRun =
        state.currentManifest.runs.find((r) => r.id === runId) ||
        state.currentManifest.runs[0];

    setLoadingState({ visible: true, label: `Preparing ${selectedRun.label}`, percent: 20, status: "loading" });

    state.currentData = await loadRunData(selectedRun.path, loadToken);
    if (loadToken !== state.activeLoadToken) return;

    setLoadingState({ visible: true, label: `Rendering ${selectedRun.label}`, percent: 82, status: "rendering" });

    pageTitle.textContent = state.currentData.title;
    pageSubtitle.textContent = state.currentData.subtitle;
    exportStatus.textContent = state.currentData.status;
    runNote.textContent = `Loaded ${selectedRun.label} with ${selectedRun.epochs} epochs.`;

    renderComparisonTable(state.currentData.summary);
    populateExperimentMeta(state.currentData, selectedRun);

    const steps = state.currentData.timeline_steps || [];
    if (steps.length > 1) {
        stepSlider.disabled = false;
        stepSlider.min = "0";
        stepSlider.max = String(steps.length - 1);
        stepSlider.value = String(steps.length - 1);
    } else {
        stepSlider.disabled = true;
        stepSlider.min = stepSlider.max = stepSlider.value = "0";
    }

    refreshStepState(state.currentData, steps.length > 0 ? steps.length - 1 : 0);
    setLoadingState({ visible: false, label: "Viewer ready", percent: 100, status: "ready" });
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main() {
    bindImageLightbox();
    bindAnalysisModal();

    setLoadingState({ visible: true, label: "Booting viewer", percent: 2, status: "loading" });
    await loadRuntimeSource();
    state.currentManifest = await loadManifest();

    const runs = state.currentManifest.runs || [];
    runSelect.innerHTML = "";
    runs.forEach((run) => {
        const opt = document.createElement("option");
        opt.value = run.id;
        opt.textContent = run.label;
        runSelect.appendChild(opt);
    });

    if (!runs.length) throw new Error("Viewer manifest contains no runs.");

    const defaultRunId = state.currentManifest.default_run || runs[0].id;
    runSelect.value = defaultRunId;
    await applyRun(defaultRunId);

    runSelect.addEventListener("change", async (e) => applyRun(e.target.value));

    stepSlider.addEventListener("input", (e) => {
        if (state.currentData) {
            refreshStepState(state.currentData, Number(e.target.value));
        }
    });
}

main().catch((error) => {
    console.error(error);
    pageSubtitle.textContent = "Failed to load static export.";
    runNote.textContent = "Unable to load a viewer manifest.";
    if (chartEmpty) {
        chartEmpty.textContent = "The static viewer failed to load its export data.";
        chartEmpty.hidden = false;
        chartEmpty.style.display = "flex";
    }
    setLoadingState({ visible: true, label: "Load failed", percent: 100, status: "error" });
});
