const dataUrl = "./data/viewer_data.json";

const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const exportStatus = document.getElementById("export-status");
const stepSlider = document.getElementById("step-slider");
const currentStepLabel = document.getElementById("current-step-label");
const playbackMode = document.getElementById("playback-mode");
const playbackNote = document.getElementById("playback-note");
const experimentMeta = document.getElementById("experiment-meta");
const overviewImage = document.getElementById("overview-image");
const circuitImage = document.getElementById("circuit-image");
const timelineCaption = document.getElementById("timeline-caption");
const chartEmpty = document.getElementById("chart-empty");

const targetPlot = document.getElementById("target-plot");
const predictionPlot = document.getElementById("prediction-plot");
const errorPlot = document.getElementById("error-plot");
const lossChart = document.getElementById("loss-chart");

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

function computeTargetGrid(grid) {
  const x = linspace(grid.x_min, grid.x_max, grid.grid_size);
  const y = linspace(grid.y_min, grid.y_max, grid.grid_size);
  const z = y.map((yValue) =>
    x.map((xValue) => Math.sin(Math.exp(xValue) + yValue))
  );
  return { x, y, z };
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
      zeroline: false
    },
    yaxis: {
      title: "x2",
      range: [0.5, 1.0],
      gridcolor: "rgba(23,33,29,0.08)",
      zeroline: false,
      scaleanchor: "x",
      scaleratio: 1
    }
  };
}

function renderHeatmapPlot(element, spec) {
  if (spec?.z) {
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
            len: 0.8
          }
        }
      ],
      makeAxisLayout(spec.title),
      { displayModeBar: false, responsive: true }
    );
    return;
  }

  Plotly.react(
    element,
    [],
    {
      ...makeAxisLayout(spec.title),
      images: [
        {
          source: spec.fallback,
          xref: "paper",
          yref: "paper",
          x: 0,
          y: 1,
          sizex: 1,
          sizey: 1,
          sizing: "stretch",
          layer: "below",
          opacity: 1
        }
      ],
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: -0.16,
          text: "Raster fallback until raw grid export is available",
          showarrow: false,
          font: { size: 12, color: "#51615b" }
        }
      ]
    },
    { displayModeBar: false, responsive: true }
  );
}

function renderLossChart(steps, currentIndex) {
  if (!steps.length) {
    chartEmpty.hidden = false;
    Plotly.react(
      lossChart,
      [],
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#ffffff",
        margin: { l: 52, r: 20, t: 20, b: 44 },
        xaxis: { visible: false },
        yaxis: { visible: false }
      },
      { displayModeBar: false, responsive: true }
    );
    return;
  }

  chartEmpty.hidden = true;
  const x = steps.map((_, index) => index);
  const tickText = steps.map((step, index) => step.label || `step ${index}`);
  const currentStep = steps[currentIndex];

  Plotly.react(
    lossChart,
    [
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Train MSE",
        x,
        y: steps.map((step) => step.train_mse),
        line: { color: "#0d8f71", width: 3 },
        marker: { size: 8 }
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Test MSE",
        x,
        y: steps.map((step) => step.test_mse),
        line: { color: "#ef8354", width: 3 },
        marker: { size: 8 }
      }
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#ffffff",
      margin: { l: 56, r: 18, t: 16, b: 56 },
      legend: { orientation: "h", x: 0, y: 1.14 },
      xaxis: {
        title: "Step",
        tickmode: "array",
        tickvals: x,
        ticktext: tickText,
        gridcolor: "rgba(23,33,29,0.08)"
      },
      yaxis: {
        title: "MSE",
        gridcolor: "rgba(23,33,29,0.08)"
      },
      shapes: [
        {
          type: "line",
          x0: currentIndex,
          x1: currentIndex,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "rgba(23,33,29,0.25)", width: 2, dash: "dot" }
        }
      ],
      annotations: [
        {
          x: currentIndex,
          y: Math.max(currentStep.train_mse, currentStep.test_mse),
          yshift: 18,
          text: currentStep.label || `step ${currentIndex}`,
          showarrow: false,
          bgcolor: "rgba(255,252,245,0.92)",
          bordercolor: "rgba(23,33,29,0.12)",
          borderpad: 6
        }
      ]
    },
    { displayModeBar: false, responsive: true }
  );
}

function refreshStepState(data, index) {
  const steps = data.timeline_steps || [];
  const targetGrid = computeTargetGrid(data.grid);

  if (!steps.length) {
    currentStepLabel.textContent = "Final snapshot";
    playbackMode.textContent = "Static export";
    timelineCaption.textContent = "Plotly viewer is ready; waiting for raw step grids.";

    renderHeatmapPlot(targetPlot, {
      title: "Target",
      ...targetGrid,
      colorscale: "Viridis"
    });
    renderHeatmapPlot(predictionPlot, {
      title: "Prediction",
      fallback: data.assets.prediction_fallback
    });
    renderHeatmapPlot(errorPlot, {
      title: "Absolute Error",
      fallback: data.assets.error_fallback
    });
    renderLossChart([], 0);
    return;
  }

  const current = steps[index];
  currentStepLabel.textContent = current.label || `Step ${index + 1}`;
  playbackMode.textContent = "Trajectory replay";
  timelineCaption.textContent = `Train MSE ${current.train_mse.toFixed(6)} | Test MSE ${current.test_mse.toFixed(6)}`;

  renderHeatmapPlot(targetPlot, {
    title: "Target",
    x: current.heatmaps?.target?.x || targetGrid.x,
    y: current.heatmaps?.target?.y || targetGrid.y,
    z: current.heatmaps?.target?.z || targetGrid.z,
    colorscale: "Viridis"
  });
  renderHeatmapPlot(predictionPlot, {
    title: "Prediction",
    x: current.heatmaps?.prediction?.x,
    y: current.heatmaps?.prediction?.y,
    z: current.heatmaps?.prediction?.z,
    colorscale: "Viridis",
    fallback: data.assets.prediction_fallback
  });
  renderHeatmapPlot(errorPlot, {
    title: "Absolute Error",
    x: current.heatmaps?.error?.x,
    y: current.heatmaps?.error?.y,
    z: current.heatmaps?.error?.z,
    colorscale: "Magma",
    fallback: data.assets.error_fallback
  });
  renderLossChart(steps, index);
}

async function main() {
  const response = await fetch(dataUrl);
  const data = await response.json();

  pageTitle.textContent = data.title;
  pageSubtitle.textContent = data.subtitle;
  exportStatus.textContent = data.status;
  playbackNote.textContent = data.description;

  overviewImage.src = data.assets.data_overview;
  circuitImage.src = data.assets.circuit;

  experimentMeta.innerHTML = "";
  appendMetaRow("Model", data.experiment.model);
  appendMetaRow("Task", data.experiment.task);
  appendMetaRow("Train domain", data.experiment.train_domain);
  appendMetaRow("Test domain", data.experiment.test_domain);
  appendMetaRow("Device", data.experiment.device);
  appendMetaRow("Note", data.experiment.note);

  const steps = data.timeline_steps || [];
  if (steps.length > 1) {
    stepSlider.disabled = false;
    stepSlider.min = "0";
    stepSlider.max = String(steps.length - 1);
    stepSlider.value = "0";
  }

  stepSlider.addEventListener("input", (event) => {
    refreshStepState(data, Number(event.target.value));
  });

  refreshStepState(data, 0);
}

main().catch((error) => {
  console.error(error);
  pageSubtitle.textContent = "Failed to load static export.";
  chartEmpty.textContent = "The static viewer failed to load its export data.";
  chartEmpty.hidden = false;
});
