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
    formatMetric, formatInteger, appendMetaRow,
    setLoadingState, loadManifest, loadRunData,
} from "./data.js";

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

function refreshStepState(data, index) {
    const steps = data.timeline_steps || [];
    if (!steps.length) {
        renderEmptyState();
        return;
    }

    const current = steps[index];
    const trainHeatmaps = current.heatmaps?.train;
    const testHeatmaps = current.heatmaps?.test;
    const trainPredictionGrid = trainHeatmaps?.prediction;
    const testPredictionGrid = testHeatmaps?.prediction;
    const trainTargetGrid = trainHeatmaps?.target;
    const testTargetGrid = testHeatmaps?.target;

    currentStepLabel.textContent = current.label || `Step ${index + 1}`;
    if (timelineCaption) {
        timelineCaption.hidden = true;
    }
    if (trainMsePill) {
        trainMsePill.textContent = `Train MSE ${current.train_mse.toFixed(6)}`;
        trainMsePill.hidden = false;
    }
    if (testMsePill) {
        testMsePill.textContent = `Test MSE ${current.test_mse.toFixed(6)}`;
        testMsePill.hidden = false;
    }

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
    renderLossChart(steps, index);
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
    if (selectedRun?.encoding !== undefined) {
        appendMetaRow("Encoding", selectedRun.encoding || "—");
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
        overviewImage.src = state.currentData.assets.data_overview;
    }
    circuitImage.src = state.currentData.assets.circuit;
    if (fourierImage) {
        fourierImage.src = `./runtime/${selectedRun.id}_fourier_spectrum.png`;
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

    refreshStepState(state.currentData, 0);
    setLoadingState({
        visible: false,
        label: "Viewer ready",
        percent: 100,
        status: "ready",
    });
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

    stepSlider.addEventListener("input", (event) => {
        if (state.currentData) {
            refreshStepState(state.currentData, Number(event.target.value));
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
