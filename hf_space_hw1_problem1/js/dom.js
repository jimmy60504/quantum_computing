export const manifestUrls = [
    "./runtime/viewer_manifest.json",
    "./data/viewer_manifest.template.json",
];

// DOM element references
export const pageTitle = document.getElementById("page-title");
export const pageSubtitle = document.getElementById("page-subtitle");
export const exportStatus = document.getElementById("export-status");
export const runSelect = document.getElementById("run-select");
export const runNote = document.getElementById("run-note");
export const resultsTableBody = document.getElementById("results-table-body");
export const stepSlider = document.getElementById("step-slider");
export const currentStepLabel = document.getElementById("current-step-label");
export const experimentMeta = document.getElementById("experiment-meta");
export const overviewImage = document.getElementById("overview-image");
export const circuitImage = document.getElementById("circuit-image");
export const fourierImage = document.getElementById("fourier-image");
export const previewableImages = Array.from(document.querySelectorAll(".previewable-image"));
export const timelineCaption = document.getElementById("timeline-caption");
export const trainMsePill = document.getElementById("train-mse-pill");
export const testMsePill = document.getElementById("test-mse-pill");
export const chartEmpty = document.getElementById("chart-empty");
export const loadingPanel = document.getElementById("loading-panel");
export const loadingLabel = document.getElementById("loading-label");
export const loadingPercent = document.getElementById("loading-percent");
export const loadingBar = document.getElementById("loading-bar");
export const analysisOpenButton = document.getElementById("analysis-open");
export const analysisModal = document.getElementById("analysis-modal");
export const analysisCloseButton = document.getElementById("analysis-close");
export const analysisHint = document.getElementById("analysis-hint");
export const analysisHintClose = document.getElementById("analysis-hint-close");
export const analysisMarkdown = document.getElementById("analysis-markdown");
export const imageLightbox = document.getElementById("image-lightbox");
export const imageLightboxImage = document.getElementById("image-lightbox-image");
export const imageLightboxCaption = document.getElementById("image-lightbox-caption");
export const imageLightboxClose = document.getElementById("image-lightbox-close");

export const trainOverlayPlot = document.getElementById("train-overlay-plot");
export const testOverlayPlot = document.getElementById("test-overlay-plot");
export const trainErrorPlot = document.getElementById("train-error-plot");
export const testErrorPlot = document.getElementById("test-error-plot");
export const lossChart = document.getElementById("loss-chart");

// Shared mutable state
export const state = {
    currentManifest: null,
    currentData: null,
    currentRunId: null,
    currentRunChunkCache: {},
    currentRunChunkInflight: {},
    overlayCameraStates: { train: null, test: null },
    cameraSyncLocked: false,
    activeLoadToken: 0,
    activeStepToken: 0,
    activePrefetchToken: 0,
    analysisMarkdownLoaded: false,
};

export const defaultOverlayCamera = {
    eye: { x: 1.5, y: 1.3, z: 0.95 },
};

export function cloneCamera(camera) {
    if (!camera) {
        return null;
    }
    return JSON.parse(JSON.stringify(camera));
}
