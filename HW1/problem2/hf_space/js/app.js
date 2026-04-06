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
    setLoadingState, loadManifest, loadRunData, loadRunChunk, loadRuntimeSource,
} from "./data.js";

const METHODS = ["explicit", "kernel", "reuploading"];
const DATASETS = ["circle", "moons"];

// ── helpers ───────────────────────────────────────────────────────────────────

function formatAcc(v) {
    if (v === null || v === undefined) return "—";
    return (v * 100).toFixed(1) + "%";
}

// ── chunk helpers ─────────────────────────────────────────────────────────────

function getChunkPaths(data) {
    return (data.timeline_chunks || []).map((c) => c.path).filter(Boolean);
}

function updatePrefetchProgress(completed, total) {
    if (!total) return;
    const percent = 82 + (completed / total) * 18;
    setLoadingState({
        visible: completed < total,
        label: `Streaming epoch chunks ${completed}/${total}`,
        percent,
        status: completed < total ? "streaming" : "ready",
    });
}

async function fetchChunkIntoCache(chunkPath, loadToken, { announce = false } = {}) {
    if (state.currentRunChunkCache[chunkPath]) {
        return state.currentRunChunkCache[chunkPath];
    }
    if (state.currentRunChunkInflight[chunkPath]) {
        return state.currentRunChunkInflight[chunkPath];
    }

    if (announce) {
        const epochMatch = chunkPath.match(/_epoch_(\d+)\.json$/);
        const epochLabel = epochMatch ? String(Number(epochMatch[1])) : "?";
        setLoadingState({
            visible: true,
            label: `Loading epoch ${epochLabel} boundaries`,
            percent: 92,
            status: "rendering",
        });
    }

    const promise = loadRunChunk(chunkPath, loadToken)
        .then((chunk) => {
            state.currentRunChunkCache[chunkPath] = chunk;
            delete state.currentRunChunkInflight[chunkPath];
            return chunk;
        })
        .catch((error) => {
            delete state.currentRunChunkInflight[chunkPath];
            throw error;
        });

    state.currentRunChunkInflight[chunkPath] = promise;
    return promise;
}

function prefetchChunkStream(data, loadToken, prefetchToken) {
    const chunkPaths = getChunkPaths(data);
    if (!chunkPaths.length) return;

    const pump = async () => {
        updatePrefetchProgress(Object.keys(state.currentRunChunkCache).length, chunkPaths.length);
        for (const chunkPath of chunkPaths) {
            if (prefetchToken !== state.activePrefetchToken || loadToken !== state.activeLoadToken) return;
            await fetchChunkIntoCache(chunkPath, loadToken);
            updatePrefetchProgress(Object.keys(state.currentRunChunkCache).length, chunkPaths.length);
            if (prefetchToken !== state.activePrefetchToken || loadToken !== state.activeLoadToken) return;
            await new Promise((resolve) => window.setTimeout(resolve, 0));
        }
        updatePrefetchProgress(chunkPaths.length, chunkPaths.length);
    };

    pump().catch((error) => console.warn("Chunk prefetch stopped:", error));
}

/**
 * Resolve the full step payload for a given index.
 * - Inline mode:  step already has `boundaries` → return as-is.
 * - Chunked mode: step has `chunk_path`          → load chunk, find step by epoch.
 */
async function resolveStepPayload(data, index, loadToken) {
    const steps = data.timeline_steps || [];
    const summaryStep = steps[index];
    if (!summaryStep) return null;

    // Inline: boundaries already present
    if (summaryStep.boundaries) return summaryStep;

    // Chunked: fetch from chunk file
    if (!summaryStep.chunk_path) return summaryStep;

    const chunkPath = summaryStep.chunk_path;
    const chunk = await fetchChunkIntoCache(chunkPath, loadToken, { announce: true });
    const chunkSteps = chunk.timeline_steps || [];
    const targetEpoch = Number(summaryStep.epoch ?? summaryStep.global_step ?? index);
    const fullStep = chunkSteps.find(
        (s) => Number(s.epoch ?? s.global_step) === targetEpoch
    );
    if (!fullStep) {
        throw new Error(`Epoch ${targetEpoch} missing from chunk ${chunkPath}`);
    }
    return { ...fullStep, chunk_path: chunkPath };
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

    // Render summary metrics immediately (no chunk needed)
    const summaryStep = steps[index];
    if (currentStepLabel) currentStepLabel.textContent = summaryStep.label || `Epoch ${summaryStep.epoch ?? index}`;
    if (timelineCaption) timelineCaption.hidden = true;

    if (trainAccPill) {
        trainAccPill.textContent = `Train ${formatAcc(summaryStep.train_acc)}`;
        trainAccPill.hidden = false;
    }
    if (testAccPill) {
        testAccPill.textContent = `Test ${formatAcc(summaryStep.test_acc)}`;
        testAccPill.hidden = false;
    }

    renderAccuracyChart(steps, index);
    updateAccPills(summaryStep);

    // Resolve full step payload (may require a chunk fetch)
    const step = await resolveStepPayload(data, index, state.activeLoadToken);
    if (stepToken !== state.activeStepToken) return;

    if (!step?.boundaries) {
        // Chunked data not yet available — leave boundary panels as-is
        return;
    }

    // Scatter lives at top level (fixed) or falls back to per-step copy
    const scatterSource = data.scatter ?? {};

    // Decision boundaries
    METHODS.forEach((method, mi) => {
        DATASETS.forEach((dataset, di) => {
            const container = boundaryPlots[mi]?.[di];
            if (!container) return;
            const heatmap = step.boundaries?.[method]?.[dataset] ?? null;
            const points = scatterSource[dataset] ?? step.scatter?.[dataset] ?? null;
            renderBoundaryPlot(container, heatmap, points, method);
        });
    });

    if (chartEmpty) chartEmpty.hidden = true;
    if (lossChart) lossChart.style.display = "";
    setLoadingState({ visible: false });
}

// ── run loading ────────────────────────────────────────────────────────────────

function populateExperimentMeta(data, run) {
    if (!experimentMeta) return;
    experimentMeta.innerHTML = "";
    appendMetaRow("Model", data.experiment?.model);
    appendMetaRow("Task", data.experiment?.task);
    appendMetaRow("Methods", (data.experiment?.methods ?? []).join(", "));
    appendMetaRow("Datasets", (data.experiment?.datasets ?? []).join(", "));
    appendMetaRow("Device", data.experiment?.device);
    if (run?.num_qubits !== undefined) appendMetaRow("Qubits", run.num_qubits);
    if (run?.num_layers_explicit !== undefined || run?.num_layers_reuploading !== undefined) {
        appendMetaRow(
            "Layers (explicit / reupload)",
            `${formatInteger(run?.num_layers_explicit)} / ${formatInteger(run?.num_layers_reuploading)}`,
        );
    } else if (run?.num_layers !== undefined) {
        appendMetaRow("Layers", run.num_layers);
    }
    if (run?.num_params !== undefined) appendMetaRow("Params", formatInteger(run.num_params));
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

    pageTitle.textContent = state.currentData.title;
    pageSubtitle.textContent = state.currentData.subtitle;
    exportStatus.textContent = state.currentData.status;
    runNote.textContent = `Loaded ${selectedRun.label} with ${selectedRun.steps} exported steps.`;

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
        stepSlider.min = "0";
        stepSlider.max = String(steps.length - 1);
        stepSlider.value = "0";
    } else {
        stepSlider.disabled = true;
        stepSlider.min = stepSlider.max = stepSlider.value = "0";
    }

    await refreshStepState(state.currentData, 0, ++state.activeStepToken);
    prefetchChunkStream(state.currentData, loadToken, ++state.activePrefetchToken);
    if (!getChunkPaths(state.currentData).length) {
        setLoadingState({ visible: false, label: "Viewer ready", percent: 100, status: "ready" });
    }
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
        opt.value = run.id;
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
    runNote.textContent = "Unable to load a viewer manifest.";
    if (chartEmpty) {
        chartEmpty.textContent = "The static viewer failed to load its export data.";
        chartEmpty.hidden = false;
        chartEmpty.style.display = "flex";
    }
    setLoadingState({ visible: true, label: "Load failed", percent: 100, status: "error" });
});
