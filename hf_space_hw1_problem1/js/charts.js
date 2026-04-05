import {
  trainOverlayPlot, testOverlayPlot, trainErrorPlot, testErrorPlot,
  lossChart, chartEmpty, currentStepLabel, timelineCaption,
  trainMsePill, testMsePill,
  state, defaultOverlayCamera, cloneCamera,
} from "./dom.js";

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

export function renderHeatmapPlot(element, spec) {
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

export function getDomainPoints(data, domain, fallbackGrid) {
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
      camera: cloneCamera(state.overlayCameraStates[domainKey] || defaultOverlayCamera),
      aspectratio: { x: 1.15, y: 1.15, z: 0.7 },
    },
  };
}

function bindOverlayCameraTracking(element, domainKey) {
  if (!element || typeof element.on !== "function") {
    return;
  }

  element.on("plotly_relayout", (eventData) => {
    if (state.cameraSyncLocked) {
      return;
    }

    const camera =
      eventData?.["scene.camera"] ||
      element.layout?.scene?.camera ||
      element._fullLayout?.scene?.camera;
    if (camera) {
      const nextCamera = cloneCamera(camera);
      state.overlayCameraStates.train = cloneCamera(nextCamera);
      state.overlayCameraStates.test = cloneCamera(nextCamera);

      const siblingElement = domainKey === "train" ? testOverlayPlot : trainOverlayPlot;
      if (siblingElement) {
        state.cameraSyncLocked = true;
        Plotly.relayout(siblingElement, { "scene.camera": cloneCamera(nextCamera) })
          .catch(() => {})
          .finally(() => {
            state.cameraSyncLocked = false;
          });
      }
    }
  });
}

export function renderOverlayPlot(element, domainKey, predictionGrid, domainPoints) {
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

export function renderLossChart(steps, currentIndex) {
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

export function renderEmptyState() {
  currentStepLabel.textContent = "Final snapshot";
  timelineCaption.textContent = "Waiting for raw step grids.";
  if (timelineCaption) {
    timelineCaption.hidden = false;
  }
  if (trainMsePill) {
    trainMsePill.hidden = true;
  }
  if (testMsePill) {
    testMsePill.hidden = true;
  }

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
