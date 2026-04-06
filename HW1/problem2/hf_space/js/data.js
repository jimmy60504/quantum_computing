import {
    experimentMeta, loadingPanel, loadingLabel, loadingPercent, loadingBar,
    exportStatus, runtimeSourceUrls, state,
} from "./dom.js";

function ensureTrailingSlash(value) {
    if (!value) {
        return "./";
    }
    return value.endsWith("/") ? value : `${value}/`;
}

function buildDatasetRuntimeRoot(repoId, revision = "main", pathPrefix = "") {
    const safePrefix = pathPrefix.trim().replace(/^\/+|\/+$/g, "");
    const suffix = safePrefix ? `${safePrefix}/` : "";
    return `https://huggingface.co/datasets/${repoId}/resolve/${revision}/${suffix}`;
}

function isAbsoluteUrl(path) {
    return /^(?:[a-z]+:)?\/\//i.test(path);
}

function resolveAgainstBase(path, baseUrl) {
    if (!path) {
        return path;
    }
    if (isAbsoluteUrl(path)) {
        return path;
    }
    const normalizedBase = isAbsoluteUrl(baseUrl)
        ? baseUrl
        : new URL(baseUrl, window.location.href).toString();
    return new URL(path, normalizedBase).toString();
}

function shouldResolveFromRuntimeRoot(path) {
    if (!path || isAbsoluteUrl(path)) {
        return false;
    }
    return path.startsWith("runtime/") || path.startsWith("./runtime/");
}

function buildRuntimeRoot(runtimeSource) {
    if (runtimeSource?.runtime_root_url) {
        return ensureTrailingSlash(runtimeSource.runtime_root_url);
    }
    if (runtimeSource?.hf_dataset_repo) {
        return buildDatasetRuntimeRoot(
            runtimeSource.hf_dataset_repo,
            runtimeSource.hf_dataset_revision || "main",
            runtimeSource.hf_dataset_path_prefix || ""
        );
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

function resolveRuntimePath(path) {
    if (!path) {
        return path;
    }
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
    if (normalized.assets) {
        normalized.assets = {
            ...normalized.assets,
            circuit: resolveRuntimePath(normalized.assets.circuit),
            data_overview: resolveRuntimePath(normalized.assets.data_overview),
        };
    }
    if (normalized.run?.path) {
        normalized.run = {
            ...normalized.run,
            path: resolveRuntimePath(normalized.run.path),
        };
    }
    if (normalized.timeline_chunks) {
        normalized.timeline_chunks = normalized.timeline_chunks.map((chunk) => ({
            ...chunk,
            path: resolveRuntimePath(chunk.path),
        }));
    }
    if (normalized.timeline_steps) {
        normalized.timeline_steps = normalized.timeline_steps.map((step) => {
            if (!step.chunk_path) {
                return step;
            }
            return {
                ...step,
                chunk_path: resolveRuntimePath(step.chunk_path),
            };
        });
    }
    return normalized;
}

function describePath(path) {
    if (!path) {
        return "runtime data";
    }
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
        setLoadingState({
            visible: true,
            label: `Loading manifest from ${describePath(url)}`,
            percent: 8,
            status: "loading",
        });
        const response = await fetch(withCacheBust(url), { cache: "no-store" });
        if (response.ok) {
            const data = normalizeManifest(await response.json());
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

export function resolveRuntimeAssetPath(path) {
    return resolveRuntimePath(path);
}

export async function loadRunData(path, loadToken) {
    const response = await fetch(withCacheBust(path), { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Failed to load run data: ${path}`);
    }
    const contentLength = Number(response.headers.get("content-length") || 0);

    if (!response.body || !contentLength) {
        setLoadingState({
            visible: true,
            label: `Loading ${describePath(path)}`,
            percent: 45,
            status: "loading",
        });
        return normalizeRunPayload(await response.json());
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
            label: `Downloading ${describePath(path)}`,
            percent: progress,
            status: "loading",
        });
    }

    text += decoder.decode();
    return normalizeRunPayload(JSON.parse(text));
}

export async function loadRunChunk(path, loadToken) {
    const response = await fetch(withCacheBust(path), { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Failed to load run chunk: ${path}`);
    }
    if (loadToken !== state.activeLoadToken) {
        throw new Error("Stale chunk load aborted.");
    }
    return normalizeRunPayload(await response.json());
}
