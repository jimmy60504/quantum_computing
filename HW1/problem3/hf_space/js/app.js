import {
    pageTitle, pageSubtitle, exportStatus,
    runSelect, runNote, resultsTableBody,
    stepSlider, currentStepLabel,
    mlpAccPill, qnnAccPill, chartEmpty,
    experimentMeta, cmPlots, lossChart,
    tsneSection, tsnePlots,
    state,
} from "./dom.js";
import { bindImageLightbox, bindAnalysisModal } from "./overlays.js";
import { renderTrainingCurves, renderConfusionMatrix, renderTsneChart, renderEmptyState } from "./charts.js";
import {
    formatAcc, formatParams, formatTime, formatInteger,
    appendMetaRow, withCacheBust,
    setLoadingState, loadManifest, loadRunData, loadRuntimeSource, loadTsneData,
} from "./data.js";

const METHODS = ["mlp", "qnn"];
const BATCHES_PER_EPOCH = 781;

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

// ── step-level accuracy interpolated from epoch history (10000-based) ────────

function _interpEpochAcc(epochSteps, epochFrac, key) {
    // epochSteps[i] = record for end of epoch (i+1); epochFrac is the epoch fraction
    // e.g. epochFrac=1.0 → end of epoch 1 → epochSteps[0]
    const epochIdx = epochFrac - 1;  // 0-based index into epochSteps
    if (epochIdx <= 0) return epochSteps[0]?.[key] ?? 0;
    if (epochIdx >= epochSteps.length - 1) return epochSteps[epochSteps.length - 1]?.[key] ?? 0;
    const lo = Math.floor(epochIdx);
    const t  = epochIdx - lo;
    const a0 = epochSteps[lo]?.[key] ?? 0;
    const a1 = epochSteps[lo + 1]?.[key] ?? 0;
    return a0 + t * (a1 - a0);
}

function computeStepAccData(tsneData) {
    if (!tsneData?.methods) return null;
    const epochSteps = state.currentEpochSteps ?? [];
    if (!epochSteps.length) return null;
    const result = {};
    for (const [method, md] of Object.entries(tsneData.methods)) {
        if (!md.steps?.length) continue;
        const steps     = md.steps;
        const accs      = steps.map((s) => _interpEpochAcc(epochSteps, s / BATCHES_PER_EPOCH, `${method}_test_acc`));
        const trainAccs = steps.map((s) => _interpEpochAcc(epochSteps, s / BATCHES_PER_EPOCH, `${method}_train_acc`));
        result[method]  = { steps, accs, trainAccs };
    }
    return Object.keys(result).length ? result : null;
}

// ── unified frame refresh ─────────────────────────────────────────────────────
// frameIdx: index into tsne frames (0‥maxFrames-1).
// In epoch-only fallback mode, frameIdx is an epoch index.

function refreshAtFrame(frameIdx) {
    const tsneData   = state.currentTsneData;
    const epochSteps = state.currentEpochSteps ?? [];
    if (!epochSteps.length) { renderEmptyState(); return; }

    // Map frame → epoch position for the vertical line
    let currentStep   = null;
    let epochFrac     = null;

    if (tsneData) {
        const firstKey = Object.keys(tsneData.methods)[0];
        const stepNums = tsneData.methods[firstKey]?.steps ?? [];
        currentStep = stepNums[frameIdx] ?? frameIdx;
        epochFrac   = currentStep / BATCHES_PER_EPOCH;
    } else {
        epochFrac = epochSteps[frameIdx]?.epoch ?? frameIdx;
    }

    // Nearest epoch record for confusion matrix
    const nearestEpochIdx = Math.min(
        epochSteps.length - 1,
        Math.max(0, Math.round(epochFrac) - 1),
    );
    const epochRecord = epochSteps[nearestEpochIdx];

    // Epoch pill label
    if (currentStepLabel) {
        currentStepLabel.textContent = epochFrac !== null
            ? `Epoch ${epochFrac.toFixed(1)}`
            : (epochRecord?.label ?? `Epoch ${nearestEpochIdx + 1}`);
    }

    // Scatter step label in tsne heading
    const tsneLabel = document.getElementById("tsne-step-label");
    if (tsneLabel) {
        tsneLabel.textContent = epochFrac !== null ? `Epoch ${epochFrac.toFixed(1)}` : "—";
    }

    // Acc pills — checkpoint-level when available, else epoch-level
    const sAcc  = state.stepAccData;
    const mlpAcc = sAcc?.mlp ? sAcc.mlp.accs[frameIdx] : epochRecord?.mlp_test_acc;
    const qnnAcc = sAcc?.qnn ? sAcc.qnn.accs[frameIdx] : epochRecord?.qnn_test_acc;
    if (mlpAccPill) { mlpAccPill.textContent = `MLP ${formatAcc(mlpAcc)}`; mlpAccPill.hidden = mlpAcc === undefined; }
    if (qnnAccPill) { qnnAccPill.textContent = `QNN ${formatAcc(qnnAcc)}`; qnnAccPill.hidden = qnnAcc === undefined; }

    // Training curves: 400-pt step acc + vertical line
    renderTrainingCurves(epochSteps, epochFrac, state.stepAccData);

    // Scatter plots
    if (tsneData) {
        for (const method of METHODS) {
            const methodData = tsneData.methods?.[method];
            const container  = tsnePlots[method] ?? document.getElementById(`tsne-${method}`);
            if (!methodData || !container) continue;
            renderTsneChart(container, methodData, tsneData.samples, frameIdx, method);
        }
    }

    // Confusion matrices — always use full 10k test set from nearest epoch history
    for (const method of METHODS) {
        renderConfusionMatrix(cmPlots[method], epochRecord?.[`${method}_confusion`] ?? null, method);
    }

    state.activeTsneStep = frameIdx;
    if (chartEmpty) chartEmpty.hidden = true;
    if (lossChart) lossChart.style.display = "";
    setLoadingState({ visible: false });
}

// ── animation (Play button in scatter heading) ────────────────────────────────

let tsneAnimTimer = null;
let tsneObserver  = null;

async function lazyLoadTsne(data) {
    if (state.currentTsneData || !data.tsne_path) return;
    try {
        const tsne = await loadTsneData(data.tsne_path);
        state.stepAccData     = computeStepAccData(tsne);
        state.currentTsneData = tsne;
        initTsneSection({ ...data, tsne_probes: tsne });
        const initFrame = stepSlider.disabled ? 0 : Number(stepSlider.value);
        refreshAtFrame(initFrame);
    } catch (e) {
        console.warn("t-SNE lazy load failed:", e);
    }
}

function initTsneSection(data) {
    const tsneData = data.tsne_probes ?? state.currentTsneData;

    // Compute step-level accuracy from tsne preds
    state.stepAccData    = computeStepAccData(tsneData);
    state.currentTsneData = tsneData ?? null;

    if (tsneAnimTimer) { clearInterval(tsneAnimTimer); tsneAnimTimer = null; }

    const playBtn = document.getElementById("tsne-play");

    if (!tsneData?.methods || !tsneSection) {
        if (playBtn) playBtn.style.display = "none";
        return;
    }
    if (playBtn) playBtn.style.display = "";

    const firstKey   = Object.keys(tsneData.methods)[0];
    const firstMd    = tsneData.methods[firstKey];
    const maxFrames  = firstMd?.coords?.length ?? firstMd?.preds?.length ?? 1;

    // step-slider takes on the full frame range
    if (stepSlider) {
        stepSlider.disabled = false;
        stepSlider.min   = "0";
        stepSlider.max   = String(maxFrames - 1);
        stepSlider.value = String(maxFrames - 1);
    }

    // Play drives the step-slider (which fires "input" → refreshAtFrame)
    if (playBtn) {
        playBtn.textContent = "▶ Play";
        playBtn.onclick = () => {
            if (tsneAnimTimer) {
                clearInterval(tsneAnimTimer); tsneAnimTimer = null;
                playBtn.textContent = "▶ Play";
            } else {
                playBtn.textContent = "⏸ Pause";
                let step = state.activeTsneStep ?? 0;
                tsneAnimTimer = setInterval(() => {
                    step = (step + 1) % maxFrames;
                    if (stepSlider) stepSlider.value = String(step);
                    refreshAtFrame(step);
                    if (step === maxFrames - 1) {
                        clearInterval(tsneAnimTimer); tsneAnimTimer = null;
                        playBtn.textContent = "▶ Play";
                    }
                }, 120);
            }
        };
    }
}

// ── run loading ──────────────────────────────────────────────────────────────

function populateExperimentMeta(data, run) {
    if (!experimentMeta) return;
    experimentMeta.innerHTML = "";
    appendMetaRow("Model",   data.experiment?.model);
    appendMetaRow("Task",    data.experiment?.task);
    appendMetaRow("Methods", (data.experiment?.methods ?? []).join(", "));
    appendMetaRow("Dataset", data.experiment?.dataset);
    appendMetaRow("Device",  data.experiment?.device);
    if (run?.num_qubits !== undefined) appendMetaRow("Qubits",  run.num_qubits);
    if (run?.num_layers !== undefined) appendMetaRow("Layers",  run.num_layers);
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

    pageTitle.textContent     = state.currentData.title;
    pageSubtitle.textContent  = state.currentData.subtitle;
    exportStatus.textContent  = state.currentData.status;
    runNote.textContent       = `Loaded ${selectedRun.label} with ${selectedRun.epochs} epochs.`;

    const epochSteps = state.currentData.timeline_steps || [];
    state.currentEpochSteps = epochSteps;

    renderComparisonTable(state.currentData.summary);
    populateExperimentMeta(state.currentData, selectedRun);

    // Reset tsne state for new run
    state.currentTsneData = null;
    state.stepAccData = null;
    if (tsneObserver) { tsneObserver.disconnect(); tsneObserver = null; }

    if (state.currentData.tsne_probes) {
        // Embedded (old format) — load immediately
        initTsneSection(state.currentData);
    } else if (state.currentData.tsne_path) {
        // Lazy-load tsne after main render completes
        lazyLoadTsne(state.currentData);
    }

    // Epoch-only fallback: step-slider covers epochs
    if (!state.currentTsneData && epochSteps.length > 1) {
        stepSlider.disabled = false;
        stepSlider.min   = "0";
        stepSlider.max   = String(epochSteps.length - 1);
        stepSlider.value = String(epochSteps.length - 1);
    } else if (!state.currentTsneData) {
        stepSlider.disabled = true;
        stepSlider.min = stepSlider.max = stepSlider.value = "0";
    }

    const initFrame = stepSlider.disabled ? 0 : Number(stepSlider.value);
    refreshAtFrame(initFrame);
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
        opt.value       = run.id;
        opt.textContent = run.label;
        runSelect.appendChild(opt);
    });

    if (!runs.length) throw new Error("Viewer manifest contains no runs.");

    const defaultRunId = state.currentManifest.default_run || runs[0].id;
    runSelect.value = defaultRunId;
    await applyRun(defaultRunId);

    runSelect.addEventListener("change", async (e) => applyRun(e.target.value));

    stepSlider.addEventListener("input", (e) => {
        if (tsneAnimTimer) {
            clearInterval(tsneAnimTimer); tsneAnimTimer = null;
            const playBtn = document.getElementById("tsne-play");
            if (playBtn) playBtn.textContent = "▶ Play";
        }
        refreshAtFrame(Number(e.target.value));
    });
}

main().catch((error) => {
    console.error(error);
    pageSubtitle.textContent = "Failed to load static export.";
    runNote.textContent      = "Unable to load a viewer manifest.";
    if (chartEmpty) {
        chartEmpty.textContent    = "The static viewer failed to load its export data.";
        chartEmpty.hidden         = false;
        chartEmpty.style.display  = "flex";
    }
    setLoadingState({ visible: true, label: "Load failed", percent: 100, status: "error" });
});
