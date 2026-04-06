export const runtimeSourceUrls = [
    "./data/runtime_source.json",
    "./data/runtime_source.template.json",
];

export const loadingPanel   = document.getElementById("loading-panel");
export const loadingLabel   = document.getElementById("loading-label");
export const loadingPercent = document.getElementById("loading-percent");
export const loadingBar     = document.getElementById("loading-bar");

export const pageTitle        = document.getElementById("page-title");
export const pageSubtitle     = document.getElementById("page-subtitle");
export const exportStatus     = document.getElementById("export-status");

export const runSelect        = document.getElementById("run-select");
export const runNote          = document.getElementById("run-note");
export const resultsTableBody = document.getElementById("results-table-body");

export const stepSlider         = document.getElementById("step-slider");
export const currentStepLabel   = document.getElementById("current-step-label");
export const timelineCaption    = document.getElementById("timeline-caption");
export const trainAccPill       = document.getElementById("train-acc-pill");
export const testAccPill        = document.getElementById("test-acc-pill");
export const chartEmpty         = document.getElementById("chart-empty");

export const experimentMeta   = document.getElementById("experiment-meta");
export const datasetsImage    = document.getElementById("datasets-image");

// Decision-boundary plot containers: [method][dataset]
// method: 0=explicit, 1=kernel, 2=reuploading
// dataset: 0=circle, 1=moons
export const boundaryPlots = [
    [document.getElementById("bp-explicit-circle"),  document.getElementById("bp-explicit-moons")],
    [document.getElementById("bp-kernel-circle"),    document.getElementById("bp-kernel-moons")],
    [document.getElementById("bp-reupload-circle"),  document.getElementById("bp-reupload-moons")],
];

export const accPills = [
    [document.getElementById("acc-explicit-circle"),  document.getElementById("acc-explicit-moons")],
    [document.getElementById("acc-kernel-circle"),    document.getElementById("acc-kernel-moons")],
    [document.getElementById("acc-reupload-circle"),  document.getElementById("acc-reupload-moons")],
];

export const lossChart = document.getElementById("loss-chart");

// Image lightbox
export const imageLightbox        = document.getElementById("image-lightbox");
export const imageLightboxImage   = document.getElementById("image-lightbox-image");
export const imageLightboxCaption = document.getElementById("image-lightbox-caption");
export const imageLightboxClose   = document.getElementById("image-lightbox-close");

// Analysis / answers modal
export const analysisOpenButton = document.getElementById("analysis-open");
export const answersOpenButton  = document.getElementById("answers-open");
export const analysisModal      = document.getElementById("analysis-modal");
export const analysisCloseButton = document.getElementById("analysis-close");
export const analysisHint       = document.getElementById("analysis-hint");
export const analysisHintClose  = document.getElementById("analysis-hint-close");
export const analysisModalLabel = document.getElementById("analysis-modal-label");
export const analysisMarkdown   = document.getElementById("analysis-markdown");
export const previewableImages  = Array.from(document.querySelectorAll(".previewable-image"));

export const state = {
    currentRunId: null,
    currentManifest: null,
    currentData: null,
    activeLoadToken: 0,
    activeStepToken: 0,
    activePrefetchToken: 0,
    currentRunChunkCache: {},
    currentRunChunkInflight: {},
    activeDataset: "circle",   // "circle" | "moons"
    markdownCache: {},
};
