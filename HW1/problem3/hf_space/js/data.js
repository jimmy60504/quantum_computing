import {
    experimentMeta, loadingPanel, loadingLabel, loadingPercent, loadingBar,
    exportStatus, runtimeSourceUrls, state,
} from "./dom.js";

function ensureTrailingSlash(value) {
    if (!value) return "./";
    return value.endsWith("/") ? value : `${value}/`;
}

function isAbsoluteUrl(path) {
    return /^(?:[a-z]+:)?\/\//i.test(path);
}

function resolveAgainstBase(path, baseUrl) {
    if (!path) return path;
    if (isAbsoluteUrl(path)) return path;
    const normalizedBase = isAbsoluteUrl(baseUrl)
        ? baseUrl
        : new URL(baseUrl, window.location.href).toString();
    return new URL(path, normalizedBase).toString();
}

function buildRuntimeRoot(runtimeSource) {
    if (runtimeSource?.runtime_root_url) {
        return ensureTrailingSlash(runtimeSource.runtime_root_url);
    }
    return ensureTrailingSlash("./");
}

function normalizeRuntimeSource(runtimeSource) {
    const normalized = {
        manifest_path: "./runtime/viewer_manifest.json",
        fallback_manifest_urls: ["./data/viewer_manifest.template.json"],
        ...runtimeSource,
    };
    normalized.runtime_root_url = buildRuntimeRoot(normalized);
    return normalized;
}

function buildManifestUrls(runtimeSource) {
    const urls = [];
    if (runtimeSource?.manifest_path) {
        urls.push(resolveAgainstBase(runtimeSource.manifest_path, runtimeSource.runtime_root_url));
    }
    for (const fallback of runtimeSource?.fallback_manifest_urls || []) {
        urls.push(resolveAgainstBase(fallback, "./"));
    }
    return urls;
}

function shouldResolveFromRuntimeRoot(path) {
    if (!path || isAbsoluteUrl(path)) return false;
    return path.startsWith("runtime/") || path.startsWith("./runtime/");
}

function resolveRuntimePath(path) {
    if (!path) return path;
    if (shouldResolveFromRuntimeRoot(path)) {
        return resolveAgainstBase(path, state.runtimeSource?.runtime_root_url || "./");
    }
    return resolveAgainstBase(path, "./");
}

function normalizeManifest(manifest) {
    return {
        ...manifest,
        runs: (manifest.runs || []).map((run) => ({
            ...run,
            path: resolveRuntimePath(run.path),
        })),
    };
}

function normalizeRunPayload(payload) {
    const normalized = { ...payload };
    if (normalized.run?.path) {
        normalized.run = { ...normalized.run, path: resolveRuntimePath(normalized.run.path) };
    }
    return normalized;
}

function describePath(path) {
    if (!path) return "runtime data";
    try {
        const url = new URL(path, window.location.href);
        const parts = url.pathname.split("/").filter(Boolean);
        return parts.slice(-2).join("/");
    } catch {
        return path.replace("./", "");
    }
}

export function withCacheBust(path) {
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}t=${Date.now()}`;
}

export function formatMetric(value) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return "---";
    return Number(value).toFixed(4);
}

export function formatInteger(value) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return "---";
    return String(Math.round(Number(value)));
}

export function formatAcc(v) {
    if (v === null || v === undefined) return "---";
    return (v * 100).toFixed(1) + "%";
}

export function formatParams(v) {
    if (v === null || v === undefined) return "---";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
}

export function formatTime(seconds) {
    if (seconds === null || seconds === undefined) return "---";
    if (seconds < 60) return seconds.toFixed(1) + "s";
    return (seconds / 60).toFixed(1) + "m";
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
    if (loadingPanel) loadingPanel.hidden = !visible;
    if (loadingLabel && label) loadingLabel.textContent = label;
    if (typeof percent === "number") {
        const clamped = Math.max(0, Math.min(100, Math.round(percent)));
        if (loadingPercent) loadingPercent.textContent = `${clamped}%`;
        if (loadingBar) loadingBar.style.width = `${clamped}%`;
    }
    if (status) exportStatus.textContent = status;
}

export async function loadRuntimeSource() {
    for (const url of runtimeSourceUrls) {
        const response = await fetch(withCacheBust(url), { cache: "no-store" });
        if (response.ok) {
            const runtimeSource = normalizeRuntimeSource(await response.json());
            state.runtimeSource = runtimeSource;
            return runtimeSource;
        }
    }
    const fallback = normalizeRuntimeSource({});
    state.runtimeSource = fallback;
    return fallback;
}

export async function loadManifest() {
    const runtimeSource = state.runtimeSource || (await loadRuntimeSource());
    for (const url of buildManifestUrls(runtimeSource)) {
        setLoadingState({ visible: true, label: `Loading manifest from ${describePath(url)}`, percent: 8, status: "loading" });
        const response = await fetch(withCacheBust(url), { cache: "no-store" });
        if (response.ok) {
            const data = normalizeManifest(await response.json());
            setLoadingState({ visible: true, label: "Manifest ready", percent: 18, status: "loading" });
            return data;
        }
    }
    throw new Error("No viewer manifest available.");
}

export async function loadRunData(path, loadToken) {
    const response = await fetch(withCacheBust(path), { cache: "no-store" });
    if (!response.ok) throw new Error(`Failed to load run data: ${path}`);

    setLoadingState({ visible: true, label: `Loading ${describePath(path)}`, percent: 45, status: "loading" });
    return normalizeRunPayload(await response.json());
}
