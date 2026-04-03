const manifestUrls = [
  "./runtime/viewer_manifest.json",
  "./data/viewer_manifest.template.json",
];

const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const exportStatus = document.getElementById("export-status");
const runSelect = document.getElementById("run-select");
const runNote = document.getElementById("run-note");
const stepSlider = document.getElementById("step-slider");
const currentStepLabel = document.getElementById("current-step-label");
const playbackMode = document.getElementById("playback-mode");
const playbackNote = document.getElementById("playback-note");
const experimentMeta = document.getElementById("experiment-meta");
const overviewImage = document.getElementById("overview-image");
const circuitImage = document.getElementById("circuit-image");
const timelineCaption = document.getElementById("timeline-caption");
const chartEmpty = document.getElementById("chart-empty");

const overlayPlot =
  document.getElementById("overlay-plot") ||
  document.getElementById("prediction-plot");
const errorPlot = document.getElementById("error-plot");
const lossChart = document.getElementById("loss-chart");

let currentManifest = null;
let currentData = null;
let overlayCameraState = null;

const defaultOverlayCamera = {
  eye: { x: 1.5, y: 1.3, z: 0.95 },
};

function appendMetaRow(label, value) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrapper.append(dt, dd);
  experimentMeta.appendChild(wrapper);
}

function linspace(start, stop, count) {
  if (count === 1) {
    return [start];
  }
  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => start + index * step);
}

function makeAxisLayout(title) {
  return {
    title: { text: title, x: 0.03, xanchor: "left" },
    margin: { l: 48, r: 16, t: 46, b: 42 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff",
    xaxis: {
      title: "x1",
      range: [0.5, 1.0],
      gridcolor: "rgba(23,33,29,0.08)",
      zeroline: false,
    },
    yaxis: {
      title: "x2",
      range: [0.5, 1.0],
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
        colorscale: spec.colorscale || "Viridis",
        zsmooth: false,
        hovertemplate: "x1=%{x:.3f}<br>x2=%{y:.3f}<br>value=%{z:.6f}<extra></extra>",
        colorbar: {
          thickness: 10,
          len: 0.8,
        },
      },
    ],
    makeAxisLayout(spec.title),
    { displayModeBar: false, responsive: true }
  );
}

function getTestPoints(data, fallbackGrid) {
  const testSamples = data.samples?.test;
  if (testSamples?.x1?.length && testSamples?.x2?.length && testSamples?.y?.length) {
    return {
      x: testSamples.x1,
      y: testSamples.x2,
      z: testSamples.y,
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

function flattenGridValues(grid) {
  return (grid.z || []).flatMap((row) => row);
}

function makeSurfaceLayout(title) {
  return {
    title: { text: title, x: 0.03, xanchor: "left" },
    margin: { l: 0, r: 0, t: 46, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        title: "x1",
        range: [0.5, 1.0],
        gridcolor: "rgba(23,33,29,0.10)",
        zeroline: false,
      },
      yaxis: {
        title: "x2",
        range: [0.5, 1.0],
        gridcolor: "rgba(23,33,29,0.10)",
        zeroline: false,
      },
      zaxis: {
        title: "value",
        range: [-1.5, 1.5],
        gridcolor: "rgba(23,33,29,0.10)",
        zeroline: false,
      },
      camera: overlayCameraState || defaultOverlayCamera,
      aspectratio: { x: 1.15, y: 1.15, z: 0.7 },
    },
    uirevision: "overlay-camera",
  };
}

function bindOverlayCameraTracking(element) {
  if (!element || element.dataset.cameraBound === "true" || typeof element.on !== "function") {
    return;
  }

  element.on("plotly_relayout", (eventData) => {
    if (eventData?.["scene.camera"]) {
      overlayCameraState = eventData["scene.camera"];
      return;
    }

    const liveCamera =
      element.layout?.scene?.camera || element._fullLayout?.scene?.camera;
    if (liveCamera) {
      overlayCameraState = liveCamera;
    }
  });

  element.dataset.cameraBound = "true";
}

function renderOverlayPlot(element, predictionGrid, targetGrid) {
  if (!element) {
    return;
  }
  const sampledTarget = getTestPoints(currentData || {}, targetGrid);

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
        x: sampledTarget.x,
        y: sampledTarget.y,
        z: sampledTarget.z,
        name: "Test targets",
        marker: {
          size: 2.2,
          color: "rgba(239,131,84,0.92)",
          opacity: 0.82,
          line: { width: 0.4, color: "rgba(255,255,255,0.55)" },
        },
        hovertemplate:
          "Test target<br>x1=%{x:.3f}<br>x2=%{y:.3f}<br>value=%{z:.6f}<extra></extra>",
      },
    ],
    makeSurfaceLayout("Prediction surface vs test targets"),
    { displayModeBar: false, responsive: true }
  );
  bindOverlayCameraTracking(element);
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
  const primarySeries = steps.map((step) => step.batch_loss ?? step.train_mse ?? 0);
  const secondarySeries = steps.map((step) => step.test_mse ?? 0);
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
        marker: { size: 8 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Test MSE",
        x,
        y: secondarySeries,
        line: { color: "#ef8354", width: 3 },
        marker: { size: 8 },
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

function refreshStepState(data, index) {
  const steps = data.timeline_steps || [];
  const targetGrid = steps[0]?.heatmaps?.target || {
    x: [],
    y: [],
    z: [],
  };

  if (!steps.length) {
    currentStepLabel.textContent = "Final snapshot";
    playbackMode.textContent = "Static export";
    timelineCaption.textContent = "Waiting for raw step grids.";
    if (overlayPlot) {
      Plotly.react(
        overlayPlot,
        [],
        makeSurfaceLayout("Prediction surface vs test targets"),
        {
          displayModeBar: false,
          responsive: true,
        }
      );
    }
    renderHeatmapPlot(errorPlot, {
      title: "Absolute Error",
      x: [],
      y: [],
      z: [],
      colorscale: "Magma",
    });
    renderLossChart([], 0);
    return;
  }

  const current = steps[index];
  const currentTargetGrid = {
    title: "Target",
    x: current.heatmaps?.target?.x || targetGrid.x,
    y: current.heatmaps?.target?.y || targetGrid.y,
    z: current.heatmaps?.target?.z || targetGrid.z,
  };
  const currentPredictionGrid = {
    title: "Prediction",
    x: current.heatmaps?.prediction?.x || targetGrid.x,
    y: current.heatmaps?.prediction?.y || targetGrid.y,
    z: current.heatmaps?.prediction?.z || [],
  };

  currentStepLabel.textContent = current.label || `Step ${index + 1}`;
  playbackMode.textContent = "Trajectory replay";
  if (current.batch_loss !== undefined) {
    timelineCaption.textContent = `Batch loss ${current.batch_loss.toFixed(6)} | Test MSE ${current.test_mse.toFixed(6)}`;
  } else {
    timelineCaption.textContent = `Train MSE ${current.train_mse.toFixed(6)} | Test MSE ${current.test_mse.toFixed(6)}`;
  }

  renderOverlayPlot(overlayPlot, currentPredictionGrid, currentTargetGrid);
  renderHeatmapPlot(errorPlot, {
    title: "Absolute Error",
    x: current.heatmaps?.error?.x || targetGrid.x,
    y: current.heatmaps?.error?.y || targetGrid.y,
    z: current.heatmaps?.error?.z,
    colorscale: "Magma",
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
    appendMetaRow("Hyperparams", `q=${selectedRun.num_qubits}, layers=${selectedRun.num_layers}, lr=${selectedRun.learning_rate}, batch=${selectedRun.batch_size}, epochs=${selectedRun.epochs}`);
  }
  appendMetaRow("Note", data.experiment.note);
}

async function loadManifest() {
  for (const url of manifestUrls) {
    const response = await fetch(url);
    if (response.ok) {
      return response.json();
    }
  }
  throw new Error("No viewer manifest available.");
}

async function loadRunData(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load run data: ${path}`);
  }
  return response.json();
}

async function applyRun(runId) {
  const selectedRun = currentManifest.runs.find((run) => run.id === runId) || currentManifest.runs[0];
  currentData = await loadRunData(selectedRun.path);

  pageTitle.textContent = currentData.title;
  pageSubtitle.textContent = currentData.subtitle;
  exportStatus.textContent = currentData.status;
  playbackNote.textContent = currentData.description;
  runNote.textContent = `Loaded ${selectedRun.label} with ${selectedRun.steps} exported steps.`;
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
}

async function main() {
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
});
