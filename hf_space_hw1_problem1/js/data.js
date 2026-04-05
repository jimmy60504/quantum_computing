import {
  experimentMeta, loadingPanel, loadingLabel, loadingPercent, loadingBar,
  exportStatus, manifestUrls, state,
} from "./dom.js";

export function formatMetric(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "—";
  }
  const numeric = Number(value);
  const absValue = Math.abs(numeric);
  if (absValue > 0 && absValue < 1e-3) {
    return numeric.toExponential(2);
  }
  return numeric.toFixed(4);
}

export function formatInteger(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "—";
  }
  return String(Math.round(Number(value)));
}

export function appendMetaRow(label, value) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrapper.append(dt, dd);
  experimentMeta.appendChild(wrapper);
}

export function setLoadingState({ visible, label, percent, status }) {
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

export async function loadManifest() {
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

export async function loadRunData(path, loadToken) {
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
    if (loadToken !== state.activeLoadToken) {
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
