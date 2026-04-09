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
export const mlpAccPill         = document.getElementById("mlp-acc-pill");
export const qnnAccPill         = document.getElementById("qnn-acc-pill");
export const chartEmpty         = document.getElementById("chart-empty");

export const experimentMeta   = document.getElementById("experiment-meta");

// Confusion matrix plot containers
export const cmPlots = {
    mlp: document.getElementById("cm-mlp"),
    qnn: document.getElementById("cm-qnn"),
};

export const lossChart = document.getElementById("loss-chart");

// t-SNE feature space
export const tsneSection   = document.getElementById("tsne-section");
export const tsneSlider    = document.getElementById("tsne-slider");
export const tsneStepLabel = document.getElementById("tsne-step-label");
export const tsnePlayBtn   = document.getElementById("tsne-play");
export const tsnePlots = {
    mlp: document.getElementById("tsne-mlp"),
    qnn: document.getElementById("tsne-qnn"),
};

// Image lightbox
export const imageLightbox        = document.getElementById("image-lightbox");
export const imageLightboxImage   = document.getElementById("image-lightbox-image");
export const imageLightboxCaption = document.getElementById("image-lightbox-caption");
export const imageLightboxClose   = document.getElementById("image-lightbox-close");

// Analysis / answers modal
export const analysisOpenButton  = document.getElementById("analysis-open");
export const answersOpenButton   = document.getElementById("answers-open");
export const analysisModal       = document.getElementById("analysis-modal");
export const analysisCloseButton = document.getElementById("analysis-close");
export const analysisModalLabel  = document.getElementById("analysis-modal-label");
export const analysisMarkdown    = document.getElementById("analysis-markdown");
export const previewableImages   = Array.from(document.querySelectorAll(".previewable-image"));

export const state = {
    currentRunId: null,
    currentManifest: null,
    currentData: null,
    activeLoadToken: 0,
    activeStepToken: 0,
    markdownCache: {},
    activeTsneStep: 0,
};
