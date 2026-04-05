import {
    pageTitle, pageSubtitle, exportStatus,
    runSelect, runNote, resultsTableBody,
    stepSlider, currentStepLabel, experimentMeta,
    overviewImage, circuitImage, fourierImage,
    trainOverlayPlot, testOverlayPlot,
    trainErrorPlot, testErrorPlot,
    timelineCaption, trainMsePill, testMsePill, chartEmpty,
    state,
} from "./dom.js";
import { bindImageLightbox, bindAnalysisModal, maybeShowAnalysisHint } from "./overlays.js";
import {
    renderOverlayPlot, renderHeatmapPlot, renderLossChart,
    renderEmptyState, getDomainPoints,
} from "./charts.js";
import {
    formatMetric, formatInteger, appendMetaRow, withCacheBust,
    setLoadingState, loadManifest, loadRunData, loadRunChunk, loadRuntimeSource,
    resolveRuntimeAssetPath,
} from "./data.js";

function getRunEncoding(run, data) {
    return run?.encoding_mode ?? run?.encoding ?? data?.experiment?.encoding ?? null;
}

function getChunkPaths(data) {
    return (data.timeline_chunks || []).map((chunk) => chunk.path).filter(Boolean);
}

function setImageSourceWithFallback(imageElement, candidatePaths) {
    if (!imageElement) {
        return;
    }

    const candidates = candidatePaths.filter(Boolean);
    if (!candidates.length) {
        imageElement.removeAttribute("src");
        return;
    }

    let index = 0;
    imageElement.onerror = () => {
        index += 1;
        if (index >= candidates.length) {
            imageElement.onerror = null;
            imageElement.removeAttribute("src");
            return;
        }
        imageElement.src = withCacheBust(candidates[index]);
    };
    imageElement.src = withCacheBust(candidates[index]);
}

function updatePrefetchProgress(completed, total) {
    if (!total) {
        return;
    }
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
            label: `Loading epoch ${epochLabel} details`,
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
    if (!chunkPaths.length) {
        return;
    }

    const pump = async () => {
        updatePrefetchProgress(Object.keys(state.currentRunChunkCache).length, chunkPaths.length);
        for (const chunkPath of chunkPaths) {
            if (prefetchToken !== state.activePrefetchToken || loadToken !== state.activeLoadToken) {
                return;
            }
            await fetchChunkIntoCache(chunkPath, loadToken);
            updatePrefetchProgress(Object.keys(state.currentRunChunkCache).length, chunkPaths.length);
            if (prefetchToken !== state.activePrefetchToken || loadToken !== state.activeLoadToken) {
                return;
            }
            await new Promise((resolve) => window.setTimeout(resolve, 0));
        }
        updatePrefetchProgress(chunkPaths.length, chunkPaths.length);
    };

    pump().catch((error) => {
        console.warn("Chunk prefetch stopped:", error);
    });
}

function renderResultsTable(runs, selectedRunId) {
    if (!resultsTableBody) {
        return;
    }

    resultsTableBody.innerHTML = "";
    runs.forEach((run) => {
        const row = document.createElement("tr");
        row.dataset.runId = run.id;
        if (run.id === selectedRunId) {
            row.classList.add("is-selected");
        }

        const values = [
            {
                text: formatMetric(run.final_test_mse ?? run.best_test_mse),
                className: "metric-cell metric-cell-strong",
            },
            {
                text: formatMetric(run.final_train_mse),
                className: "metric-cell",
            },
            { text: run.label || run.id, className: "run-cell" },
            { text: getRunEncoding(run) || "—" },
            { text: formatInteger(run.num_qubits) },
            { text: formatInteger(run.num_layers) },
        ];

        values.forEach((value) => {
            const cell = document.createElement("td");
            cell.textContent = value.text;
            if (value.className) {
                cell.className = value.className;
            }
            row.appendChild(cell);
        });

        row.addEventListener("click", async () => {
            runSelect.value = run.id;
            await applyRun(run.id);
        });

        resultsTableBody.appendChild(row);
    });
}

async function resolveStepPayload(data, index, loadToken) {
    const steps = data.timeline_steps || [];
    const summaryStep = steps[index];
    if (!summaryStep) {
        return null;
    }
    if (summaryStep.heatmaps) {
        return summaryStep;
    }
    if (!summaryStep.chunk_path) {
        return summaryStep;
    }

    const chunkPath = summaryStep.chunk_path;
    const chunk = await fetchChunkIntoCache(chunkPath, loadToken, { announce: true });
    const chunkSteps = chunk.timeline_steps || [];
    const fullStep = chunkSteps.find(
        (step) => Number(step.global_step) === Number(summaryStep.global_step)
    );
    if (!fullStep) {
        throw new Error(`Step ${summaryStep.global_step} missing from chunk ${chunkPath}`);
    }
    return { ...fullStep, chunk_path: chunkPath };
}

async function refreshStepState(data, index, stepToken) {
    const steps = data.timeline_steps || [];
    if (!steps.length) {
        renderEmptyState();
        return;
    }

    const summaryStep = steps[index];
    currentStepLabel.textContent = summaryStep.label || `Step ${index + 1}`;
    if (timelineCaption) {
        timelineCaption.hidden = true;
    }
    if (trainMsePill) {
        trainMsePill.textContent = `Train MSE ${formatMetric(summaryStep.train_mse)}`;
        trainMsePill.hidden = false;
    }
    if (testMsePill) {
        testMsePill.textContent = `Test MSE ${formatMetric(summaryStep.test_mse)}`;
        testMsePill.hidden = false;
    }
    renderLossChart(steps, index);

    const current = await resolveStepPayload(data, index, state.activeLoadToken);
    if (stepToken !== state.activeStepToken) {
        return;
    }
    if (!current?.heatmaps) {
        renderEmptyState();
        return;
    }

    const trainHeatmaps = current.heatmaps.train;
    const testHeatmaps = current.heatmaps.test;
    const trainPredictionGrid = trainHeatmaps?.prediction;
    const testPredictionGrid = testHeatmaps?.prediction;
    const trainTargetGrid = trainHeatmaps?.target;
    const testTargetGrid = testHeatmaps?.target;

    renderOverlayPlot(
        trainOverlayPlot,
        "train",
        trainPredictionGrid,
        getDomainPoints(data, "train", trainTargetGrid),
        "Train surface vs train samples"
    );
    renderOverlayPlot(
        testOverlayPlot,
        "test",
        testPredictionGrid,
        getDomainPoints(data, "test", testTargetGrid),
        "Test surface vs test samples"
    );
    renderHeatmapPlot(trainErrorPlot, {
        title: "Train Absolute Error",
        x: trainHeatmaps.error.x,
        y: trainHeatmaps.error.y,
        z: trainHeatmaps.error.z,
        colorscale: "Magma",
        showTitle: false,
    });
    renderHeatmapPlot(testErrorPlot, {
        title: "Test Absolute Error",
        x: testHeatmaps.error.x,
        y: testHeatmaps.error.y,
        z: testHeatmaps.error.z,
        colorscale: "Magma",
        showTitle: false,
    });
    setLoadingState({ visible: false });
}

function populateExperimentMeta(data, selectedRun) {
    experimentMeta.innerHTML = "";
    appendMetaRow("Model", data.experiment.model);
    appendMetaRow("Task", data.experiment.task);
    appendMetaRow("Train domain", data.experiment.train_domain);
    appendMetaRow("Test domain", data.experiment.test_domain);
    appendMetaRow("Device", data.experiment.device);
    if (selectedRun?.num_qubits !== undefined) {
        appendMetaRow("Qubits", selectedRun.num_qubits);
        appendMetaRow("Layers", selectedRun.num_layers);
    }
    const encoding = getRunEncoding(selectedRun, data);
    if (encoding !== null) {
        appendMetaRow("Encoding", encoding || "—");
    }
    if (selectedRun?.trainable_parameters !== undefined) {
        appendMetaRow("Parameters", formatInteger(selectedRun.trainable_parameters));
    }
    if (selectedRun?.learning_rate !== undefined) {
        appendMetaRow("Learning rate", selectedRun.learning_rate);
    }
    if (selectedRun?.batch_size !== undefined) {
        appendMetaRow("Batch size", selectedRun.batch_size);
    }
    if (selectedRun?.epochs !== undefined) {
        appendMetaRow("Epochs", selectedRun.epochs);
    }
    if (selectedRun?.final_train_mse !== undefined || selectedRun?.final_test_mse !== undefined) {
        appendMetaRow("Train MSE", formatMetric(selectedRun.final_train_mse));
        appendMetaRow(
            "Test MSE",
            formatMetric(selectedRun.final_test_mse ?? selectedRun.best_test_mse)
        );
    }
    appendMetaRow("Note", data.experiment.note);
}

async function applyRun(runId) {
    state.currentRunId = runId;
    state.currentRunChunkCache = {};
    state.currentRunChunkInflight = {};
    state.overlayCameraStates.train = null;
    state.overlayCameraStates.test = null;
    const loadToken = ++state.activeLoadToken;
    const selectedRun =
        state.currentManifest.runs.find((run) => run.id === runId) || state.currentManifest.runs[0];
    setLoadingState({
        visible: true,
        label: `Preparing ${selectedRun.label}`,
        percent: 20,
        status: "loading",
    });
    state.currentData = await loadRunData(selectedRun.path, loadToken);
    if (loadToken !== state.activeLoadToken) {
        return;
    }

    const steps = state.currentData.timeline_steps || [];
    const finalStep = steps.length ? steps[steps.length - 1] : null;
    if (finalStep) {
        if (selectedRun.final_train_mse === undefined || selectedRun.final_train_mse === null) {
            selectedRun.final_train_mse = finalStep.train_mse;
        }
        if (selectedRun.final_test_mse === undefined || selectedRun.final_test_mse === null) {
            selectedRun.final_test_mse = finalStep.test_mse;
        }
    }

    setLoadingState({
        visible: true,
        label: `Rendering ${selectedRun.label}`,
        percent: 82,
        status: "rendering",
    });

    pageTitle.textContent = state.currentData.title;
    pageSubtitle.textContent = state.currentData.subtitle;
    exportStatus.textContent = state.currentData.status;
    runNote.textContent = `Loaded ${selectedRun.label} with ${selectedRun.steps} exported steps.`;
    renderResultsTable(state.currentManifest.runs || [], selectedRun.id);
    if (overviewImage) {
        setImageSourceWithFallback(overviewImage, [state.currentData.assets.data_overview]);
    }
    setImageSourceWithFallback(circuitImage, [state.currentData.assets.circuit]);
    if (fourierImage) {
        setImageSourceWithFallback(fourierImage, [
            resolveRuntimeAssetPath(`./runtime/${selectedRun.id}_fourier_spectrum.png`),
            `./runtime/${selectedRun.id}_fourier_spectrum.png`,
            `./HW1/problem1/artifacts/${selectedRun.id}_fourier_spectrum.png`,
        ]);
    }
    populateExperimentMeta(state.currentData, selectedRun);

    if (steps.length > 1) {
        stepSlider.disabled = false;
        stepSlider.min = "0";
        stepSlider.max = String(steps.length - 1);
        stepSlider.value = "0";
    } else {
        stepSlider.disabled = true;
        stepSlider.min = "0";
        stepSlider.max = "0";
        stepSlider.value = "0";
    }

    await refreshStepState(state.currentData, 0, ++state.activeStepToken);
    prefetchChunkStream(state.currentData, loadToken, ++state.activePrefetchToken);
    if (!getChunkPaths(state.currentData).length) {
        setLoadingState({
            visible: false,
            label: "Viewer ready",
            percent: 100,
            status: "ready",
        });
    }
}

async function main() {
    bindImageLightbox();
    bindAnalysisModal();
    maybeShowAnalysisHint();
    setLoadingState({
        visible: true,
        label: "Booting viewer",
        percent: 2,
        status: "loading",
    });
    await loadRuntimeSource();
    state.currentManifest = await loadManifest();

    const runs = state.currentManifest.runs || [];
    runSelect.innerHTML = "";
    runs.forEach((run) => {
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = run.label;
        runSelect.appendChild(option);
    });

    if (!runs.length) {
        throw new Error("Viewer manifest contains no runs.");
    }

    const defaultRunId = state.currentManifest.default_run || runs[0].id;
    runSelect.value = defaultRunId;
    renderResultsTable(runs, defaultRunId);
    await applyRun(defaultRunId);

    runSelect.addEventListener("change", async (event) => {
        await applyRun(event.target.value);
    });

    stepSlider.addEventListener("input", async (event) => {
        if (state.currentData) {
            await refreshStepState(
                state.currentData,
                Number(event.target.value),
                ++state.activeStepToken
            );
        }
    });
}

main().catch((error) => {
    console.error(error);
    pageSubtitle.textContent = "Failed to load static export.";
    runNote.textContent = "Unable to load a viewer manifest.";
    chartEmpty.textContent = "The static viewer failed to load its export data.";
    chartEmpty.hidden = false;
    chartEmpty.style.display = "flex";
    setLoadingState({
        visible: true,
        label: "Load failed",
        percent: 100,
        status: "error",
    });
});
