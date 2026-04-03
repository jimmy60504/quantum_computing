const manifestUrls = [
  "./runtime/viewer_manifest.json",
  "./data/viewer_manifest.template.json",
];

const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const exportStatus = document.getElementById("export-status");
const runSelect = document.getElementById("run-select");
const runNote = document.getElementById("run-note");
const resultsTableBody = document.getElementById("results-table-body");
const stepSlider = document.getElementById("step-slider");
const currentStepLabel = document.getElementById("current-step-label");
const experimentMeta = document.getElementById("experiment-meta");
const overviewImage = document.getElementById("overview-image");
const circuitImage = document.getElementById("circuit-image");
const previewableImages = Array.from(document.querySelectorAll(".previewable-image"));
const timelineCaption = document.getElementById("timeline-caption");
const chartEmpty = document.getElementById("chart-empty");
const loadingPanel = document.getElementById("loading-panel");
const loadingLabel = document.getElementById("loading-label");
const loadingPercent = document.getElementById("loading-percent");
const loadingBar = document.getElementById("loading-bar");
const imageLightbox = document.getElementById("image-lightbox");
const imageLightboxImage = document.getElementById("image-lightbox-image");
const imageLightboxCaption = document.getElementById("image-lightbox-caption");
const imageLightboxClose = document.getElementById("image-lightbox-close");

const trainOverlayPlot = document.getElementById("train-overlay-plot");
const testOverlayPlot = document.getElementById("test-overlay-plot");
const trainErrorPlot = document.getElementById("train-error-plot");
const testErrorPlot = document.getElementById("test-error-plot");
const lossChart = document.getElementById("loss-chart");

let currentManifest = null;
let currentData = null;
let currentRunId = null;
const overlayCameraStates = {
  train: null,
  test: null,
};
let cameraSyncLocked = false;
let activeLoadToken = 0;

const defaultOverlayCamera = {
  eye: { x: 1.5, y: 1.3, z: 0.95 },
};

function cloneCamera(camera) {
  if (!camera) {
    return null;
  }
  return JSON.parse(JSON.stringify(camera));
}

function openImageLightbox(sourceImage) {
  if (!imageLightbox || !imageLightboxImage || !sourceImage?.src) {
    return;
  }

  imageLightboxImage.src = sourceImage.src;
  imageLightboxImage.alt = sourceImage.alt || "Preview";
  if (imageLightboxCaption) {
    imageLightboxCaption.textContent =
      sourceImage.dataset.previewCaption || sourceImage.alt || "";
  }
  imageLightbox.hidden = false;
  imageLightbox.setAttribute("aria-hidden", "false");
}

function closeImageLightbox() {
  if (!imageLightbox || !imageLightboxImage) {
    return;
  }

  imageLightbox.hidden = true;
  imageLightbox.setAttribute("aria-hidden", "true");
  imageLightboxImage.removeAttribute("src");
}

function bindImageLightbox() {
  previewableImages.forEach((image) => {
    image.addEventListener("click", () => openImageLightbox(image));
  });

  imageLightbox?.addEventListener("click", (event) => {
    const closeRequested =
      event.target === imageLightbox || event.target?.dataset?.lightboxClose === "true";
    if (closeRequested) {
      closeImageLightbox();
    }
  });

  imageLightboxClose?.addEventListener("click", closeImageLightbox);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && imageLightbox && !imageLightbox.hidden) {
      closeImageLightbox();
    }
  });
}

function appendMetaRow(label, value) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrapper.append(dt, dd);
  experimentMeta.appendChild(wrapper);
}

function formatMetric(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(4);
}

function formatInteger(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "—";
  }
  return String(Math.round(Number(value)));
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

function setLoadingState({ visible, label, percent, status }) {
  if (loadingPanel) {
    loadingPanel.hidden = !visible;
  }
  if (loadingLabel && label) {
    loadingLabel.textContent = label;
  }
  if (typeof percent === "number") {
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    if (loadingPercent) {
      loadingPercent.textContent = `${clamped}%`;
    }
    if (loadingBar) {
      loadingBar.style.width = `${clamped}%`;
    }
  }
  if (status) {
    exportStatus.textContent = status;
  }
}

function makeAxisLayout(title, xRange, yRange) {
  return {
    title: title ? { text: title, x: 0.03, xanchor: "left" } : undefined,
    margin: { l: 48, r: 16, t: 46, b: 42 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff",
    xaxis: {
      title: "x1",
      range: xRange,
      gridcolor: "rgba(23,33,29,0.08)",
      zeroline: false,
    },
    yaxis: {
      title: "x2",
      range: yRange,
      gridcolor: "rgba(23,33,29,0.08)",
      zeroline: false,
      scaleanchor: "x",
      scaleratio: 1,
    },
  };
}

function renderHeatmapPlot(element, spec) {
  if (!element) {
    return;
  }

  Plotly.react(
    element,
    [
      {
        type: "heatmap",
        x: spec.x,
        y: spec.y,
        z: spec.z,
        colorscale: spec.colorscale || "Magma",
        zsmooth: false,
        hovertemplate:
          "x1=%{x:.3f}<br>x2=%{y:.3f}<br>value=%{z:.6f}<extra></extra>",
        colorbar: {
          thickness: 10,
          len: 0.78,
        },
      },
    ],
    makeAxisLayout(
      spec.showTitle === false ? null : spec.title,
      [Math.min(...spec.x), Math.max(...spec.x)],
      [Math.min(...spec.y), Math.max(...spec.y)]
    ),
    { displayModeBar: false, responsive: true }
  );
}

function getDomainPoints(data, domain, fallbackGrid) {
  const domainSamples = data.samples?.[domain];
  if (domainSamples?.x1?.length && domainSamples?.x2?.length && domainSamples?.y?.length) {
    return {
      x: domainSamples.x1,
      y: domainSamples.x2,
      z: domainSamples.y,
    };
  }

  const x = fallbackGrid.x || [];
  const y = fallbackGrid.y || [];
  const z = fallbackGrid.z || [];
  const totalPoints = x.length * y.length;
  if (!totalPoints) {
    return { x: [], y: [], z: [] };
  }

  const stride = Math.max(1, Math.ceil(Math.sqrt(totalPoints / 225)));
  const xs = [];
  const ys = [];
  const zs = [];

  for (let rowIndex = 0; rowIndex < y.length; rowIndex += stride) {
    for (let colIndex = 0; colIndex < x.length; colIndex += stride) {
      xs.push(x[colIndex]);
      ys.push(y[rowIndex]);
      zs.push(z[rowIndex][colIndex]);
    }
  }

  return { x: xs, y: ys, z: zs };
}

function makeSurfaceLayout(title, domainKey, xRange, yRange) {
  return {
    title: title ? { text: title, x: 0.03, xanchor: "left" } : undefined,
    margin: { l: 0, r: 0, t: 46, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      dragmode: "turntable",
      xaxis: {
        title: "x1",
        range: xRange,
        gridcolor: "rgba(23,33,29,0.10)",
        linecolor: "rgba(23,33,29,0.42)",
        tickcolor: "rgba(23,33,29,0.42)",
        zerolinecolor: "rgba(23,33,29,0.26)",
        zeroline: false,
      },
      yaxis: {
        title: "x2",
        range: yRange,
        gridcolor: "rgba(23,33,29,0.10)",
        linecolor: "rgba(23,33,29,0.42)",
        tickcolor: "rgba(23,33,29,0.42)",
        zerolinecolor: "rgba(23,33,29,0.26)",
        zeroline: false,
      },
      zaxis: {
        title: "value",
        range: [-1.5, 1.5],
        gridcolor: "rgba(23,33,29,0.10)",
        linecolor: "rgba(23,33,29,0.42)",
        tickcolor: "rgba(23,33,29,0.42)",
        zerolinecolor: "rgba(23,33,29,0.26)",
        zeroline: false,
      },
      camera: cloneCamera(overlayCameraStates[domainKey] || defaultOverlayCamera),
      aspectratio: { x: 1.15, y: 1.15, z: 0.7 },
    },
  };
}

function bindOverlayCameraTracking(element, domainKey) {
  if (!element || typeof element.on !== "function") {
    return;
  }

  element.on("plotly_relayout", (eventData) => {
    if (cameraSyncLocked) {
      return;
    }

    const camera =
      eventData?.["scene.camera"] ||
      element.layout?.scene?.camera ||
      element._fullLayout?.scene?.camera;
    if (camera) {
      const nextCamera = cloneCamera(camera);
      overlayCameraStates.train = cloneCamera(nextCamera);
      overlayCameraStates.test = cloneCamera(nextCamera);

      const siblingElement = domainKey === "train" ? testOverlayPlot : trainOverlayPlot;
      if (siblingElement) {
        cameraSyncLocked = true;
        Plotly.relayout(siblingElement, { "scene.camera": cloneCamera(nextCamera) })
          .catch(() => {})
          .finally(() => {
            cameraSyncLocked = false;
          });
      }
    }
  });
}

function renderOverlayPlot(element, domainKey, predictionGrid, domainPoints) {
  if (!element) {
    return;
  }

  const pointDisplayZ = domainPoints.z || [];
  Plotly.react(
    element,
    [
      {
        type: "surface",
        x: predictionGrid.x,
        y: predictionGrid.y,
        z: predictionGrid.z,
        colorscale: "Viridis",
        opacity: 0.62,
        showscale: false,
        contours: {
          z: {
            show: true,
            usecolormap: false,
            color: "rgba(255,255,255,0.35)",
            width: 1,
          },
        },
        hovertemplate:
          "Prediction surface<br>x1=%{x:.3f}<br>x2=%{y:.3f}<br>value=%{z:.6f}<extra></extra>",
      },
      {
        type: "scatter3d",
        mode: "markers",
        x: domainPoints.x,
        y: domainPoints.y,
        z: pointDisplayZ,
        customdata: domainPoints.z,
        name: `${domainKey} targets`,
        marker: {
          size: 2.1,
          color: domainKey === "train" ? "rgba(13,143,113,0.88)" : "rgba(239,131,84,0.92)",
          opacity: 0.82,
          line: {
            width: 0.45,
            color: "rgba(255,255,255,0.6)",
          },
        },
        hovertemplate:
          `${domainKey} sample<br>x1=%{x:.3f}<br>x2=%{y:.3f}<br>value=%{customdata:.6f}<extra></extra>`,
      },
    ],
    makeSurfaceLayout(
      null,
      domainKey,
      [Math.min(...predictionGrid.x), Math.max(...predictionGrid.x)],
      [Math.min(...predictionGrid.y), Math.max(...predictionGrid.y)]
    ),
    { displayModeBar: false, responsive: true }
  );
  bindOverlayCameraTracking(element, domainKey);
}

function renderLossChart(steps, currentIndex) {
  if (!lossChart || !chartEmpty) {
    return;
  }

  if (!steps.length) {
    chartEmpty.hidden = false;
    chartEmpty.style.display = "flex";
    Plotly.react(
      lossChart,
      [],
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#ffffff",
        margin: { l: 52, r: 20, t: 20, b: 44 },
        xaxis: { visible: false },
        yaxis: { visible: false },
      },
      { displayModeBar: false, responsive: true }
    );
    return;
  }

  chartEmpty.hidden = true;
  chartEmpty.style.display = "none";
  const x = steps.map((step, index) => step.global_step || index + 1);
  const currentStep = steps[currentIndex];
  const LOG_EPSILON = 1e-6;
  const primarySeries = steps.map((step) =>
    Math.max(step.batch_loss ?? step.train_mse ?? LOG_EPSILON, LOG_EPSILON)
  );
  const secondarySeries = steps.map((step) => Math.max(step.test_mse ?? LOG_EPSILON, LOG_EPSILON));
  const tickCount = Math.min(8, x.length);
  const tickvals = Array.from({ length: tickCount }, (_, index) => {
    const stepIndex = Math.round((index / Math.max(tickCount - 1, 1)) * (x.length - 1));
    return x[stepIndex];
  });

  Plotly.react(
    lossChart,
    [
      {
        type: "scatter",
        mode: "lines+markers",
        name: steps[0].batch_loss !== undefined ? "Batch loss" : "Train MSE",
        x,
        y: primarySeries,
        line: { color: "#0d8f71", width: 3 },
        marker: { size: 6 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Test MSE",
        x,
        y: secondarySeries,
        line: { color: "#ef8354", width: 3 },
        marker: { size: 6 },
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#ffffff",
      margin: { l: 56, r: 18, t: 16, b: 56 },
      legend: { orientation: "h", x: 0, y: 1.14 },
      xaxis: {
        title: "Step",
        tickmode: "array",
        tickvals,
        gridcolor: "rgba(23,33,29,0.08)",
      },
      yaxis: {
        title: "MSE",
        type: "log",
        gridcolor: "rgba(23,33,29,0.08)",
      },
      shapes: [
        {
          type: "line",
          x0: x[currentIndex],
          x1: x[currentIndex],
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "rgba(23,33,29,0.25)", width: 2, dash: "dot" },
        },
      ],
      annotations: [
        {
          x: x[currentIndex],
          y: Math.max(primarySeries[currentIndex], secondarySeries[currentIndex]),
          yshift: 18,
          text: currentStep.label || `step ${currentIndex + 1}`,
          showarrow: false,
          bgcolor: "rgba(255,252,245,0.92)",
          bordercolor: "rgba(23,33,29,0.12)",
          borderpad: 6,
        },
      ],
    },
    { displayModeBar: false, responsive: true }
  );
}

function renderEmptyState() {
  currentStepLabel.textContent = "Final snapshot";
  timelineCaption.textContent = "Waiting for raw step grids.";

  for (const [element, domainKey, title] of [
    [trainOverlayPlot, "train", "Train surface vs train samples"],
    [testOverlayPlot, "test", "Test surface vs test samples"],
  ]) {
    if (element) {
      Plotly.react(element, [], makeSurfaceLayout(title, domainKey, [0, 1], [0, 1]), {
        displayModeBar: false,
        responsive: true,
      });
    }
  }

  for (const [element, title] of [
    [trainErrorPlot, "Train Absolute Error"],
    [testErrorPlot, "Test Absolute Error"],
  ]) {
    renderHeatmapPlot(element, {
      title,
      x: [0, 1],
      y: [0, 1],
      z: [
        [0, 0],
        [0, 0],
      ],
      colorscale: "Magma",
      showTitle: false,
    });
  }

  renderLossChart([], 0);
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
  timelineCaption.textContent =
    `Train MSE ${current.train_mse.toFixed(6)} | Test MSE ${current.test_mse.toFixed(6)}`;

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

async function loadManifest() {
  for (const url of manifestUrls) {
    setLoadingState({
      visible: true,
      label: `Loading manifest from ${url.replace("./", "")}`,
      percent: 8,
      status: "loading",
    });
    const response = await fetch(url);
    if (response.ok) {
      const data = await response.json();
      setLoadingState({
        visible: true,
        label: "Manifest ready",
        percent: 18,
        status: "loading",
      });
      return data;
    }
  }
  throw new Error("No viewer manifest available.");
}

async function loadRunData(path, loadToken) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load run data: ${path}`);
  }
  const contentLength = Number(response.headers.get("content-length") || 0);

  if (!response.body || !contentLength) {
    setLoadingState({
      visible: true,
      label: `Loading ${path.replace("./runtime/", "")}`,
      percent: 45,
      status: "loading",
    });
    return response.json();
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let received = 0;
  let text = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (loadToken !== activeLoadToken) {
      throw new Error("Stale run load aborted.");
    }
    received += value.byteLength;
    text += decoder.decode(value, { stream: true });
    const progress = 20 + (received / contentLength) * 55;
    setLoadingState({
      visible: true,
      label: `Downloading ${path.replace("./runtime/", "")}`,
      percent: progress,
      status: "loading",
    });
  }

  text += decoder.decode();
  return JSON.parse(text);
}

async function applyRun(runId) {
  currentRunId = runId;
  overlayCameraStates.train = null;
  overlayCameraStates.test = null;
  const loadToken = ++activeLoadToken;
  const selectedRun =
    currentManifest.runs.find((run) => run.id === runId) || currentManifest.runs[0];
  setLoadingState({
    visible: true,
    label: `Preparing ${selectedRun.label}`,
    percent: 20,
    status: "loading",
  });
  currentData = await loadRunData(selectedRun.path, loadToken);
  if (loadToken !== activeLoadToken) {
    return;
  }

  setLoadingState({
    visible: true,
    label: `Rendering ${selectedRun.label}`,
    percent: 82,
    status: "rendering",
  });

  pageTitle.textContent = currentData.title;
  pageSubtitle.textContent = currentData.subtitle;
  exportStatus.textContent = currentData.status;
  runNote.textContent = `Loaded ${selectedRun.label} with ${selectedRun.steps} exported steps.`;
  renderResultsTable(currentManifest.runs || [], selectedRun.id);
  overviewImage.src = currentData.assets.data_overview;
  circuitImage.src = currentData.assets.circuit;
  populateExperimentMeta(currentData, selectedRun);

  const steps = currentData.timeline_steps || [];
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

  refreshStepState(currentData, 0);
  setLoadingState({
    visible: false,
    label: "Viewer ready",
    percent: 100,
    status: "ready",
  });
}

async function main() {
  bindImageLightbox();
  setLoadingState({
    visible: true,
    label: "Booting viewer",
    percent: 2,
    status: "loading",
  });
  currentManifest = await loadManifest();

  const runs = currentManifest.runs || [];
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

  const defaultRunId = currentManifest.default_run || runs[0].id;
  runSelect.value = defaultRunId;
  renderResultsTable(runs, defaultRunId);
  await applyRun(defaultRunId);

  runSelect.addEventListener("change", async (event) => {
    await applyRun(event.target.value);
  });

  stepSlider.addEventListener("input", (event) => {
    if (currentData) {
      refreshStepState(currentData, Number(event.target.value));
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
