import {
    imageLightbox, imageLightboxImage, imageLightboxCaption, imageLightboxClose,
    analysisOpenButton, answersOpenButton, analysisModal, analysisCloseButton,
    analysisHint, analysisHintClose, analysisModalLabel, analysisMarkdown,
    previewableImages, state,
} from "./dom.js";

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

export function bindPreviewableImages(images) {
    images.forEach((image) => {
        image.addEventListener("click", () => openImageLightbox(image));
    });
}

export function bindImageLightbox() {
    bindPreviewableImages(previewableImages);

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

function dismissAnalysisHint() {
    if (analysisHint) {
        analysisHint.hidden = true;
    }
}

export function maybeShowAnalysisHint() {
    if (!analysisHint) {
        return;
    }
    analysisHint.hidden = false;
}

function escapeHtml(markup) {
    return markup.replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]));
}

function renderMarkdown(markdown) {
    return globalThis.marked?.parse?.(markdown) ?? `<pre>${escapeHtml(markdown)}</pre>`;
}

function enhanceMarkdownImages() {
    const analysisImages = Array.from(analysisMarkdown?.querySelectorAll("img") || []);
    analysisImages.forEach((image) => {
        image.classList.add("previewable-image");
        if (!image.dataset.previewCaption && image.alt) {
            image.dataset.previewCaption = image.alt;
        }
    });
    bindPreviewableImages(analysisImages);
}

async function ensureMarkdownDocument(sourcePath) {
    if (!analysisMarkdown) {
        return;
    }
    if (state.markdownCache[sourcePath]) {
        analysisMarkdown.innerHTML = state.markdownCache[sourcePath];
        enhanceMarkdownImages();
        return;
    }

    try {
        const response = await fetch(sourcePath);
        if (!response.ok) {
            throw new Error(`Failed to load markdown document: ${sourcePath}`);
        }
        const markdown = await response.text();
        const rendered = renderMarkdown(markdown);
        analysisMarkdown.innerHTML = rendered;
        enhanceMarkdownImages();
        state.markdownCache[sourcePath] = rendered;
    } catch (error) {
        console.error(error);
        analysisMarkdown.textContent = "Failed to load notes.";
    }
}

function openAnalysisModal({ sourcePath, label }) {
    if (!analysisModal) {
        return;
    }
    dismissAnalysisHint();
    analysisModal.hidden = false;
    analysisModal.setAttribute("aria-hidden", "false");
    if (analysisModalLabel) {
        analysisModalLabel.textContent = label;
    }
    analysisMarkdown.textContent = "Loading notes...";
    void ensureMarkdownDocument(sourcePath);
}

function closeAnalysisModal() {
    if (!analysisModal) {
        return;
    }
    analysisModal.hidden = true;
    analysisModal.setAttribute("aria-hidden", "true");
}

export function bindAnalysisModal() {
    analysisOpenButton?.addEventListener("click", () => openAnalysisModal({
        sourcePath: "./ANALYSIS.md",
        label: "Analysis notes",
    }));
    answersOpenButton?.addEventListener("click", () => openAnalysisModal({
        sourcePath: "./ANSWERS.md",
        label: "Problem 2 answers",
    }));
    analysisCloseButton?.addEventListener("click", closeAnalysisModal);
    analysisHintClose?.addEventListener("click", dismissAnalysisHint);

    analysisModal?.addEventListener("click", (event) => {
        const closeRequested =
            event.target === analysisModal || event.target?.dataset?.analysisClose === "true";
        if (closeRequested) {
            closeAnalysisModal();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && analysisModal && !analysisModal.hidden) {
            closeAnalysisModal();
        }
    });
}
