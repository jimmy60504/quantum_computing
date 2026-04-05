import {
  imageLightbox, imageLightboxImage, imageLightboxCaption, imageLightboxClose,
  analysisOpenButton, analysisModal, analysisCloseButton,
  analysisHint, analysisHintClose, analysisMarkdown,
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

async function ensureAnalysisMarkdown() {
  if (!analysisMarkdown || state.analysisMarkdownLoaded) {
    return;
  }

  try {
    const response = await fetch("./ANALYSIS.md");
    if (!response.ok) {
      throw new Error("Failed to load analysis markdown.");
    }
    const markdown = await response.text();
    const rendered =
      globalThis.marked?.parse?.(markdown) ??
      `<pre>${markdown.replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]))}</pre>`;
    analysisMarkdown.innerHTML = rendered;
    const analysisImages = Array.from(analysisMarkdown.querySelectorAll("img"));
    analysisImages.forEach((image) => {
      image.classList.add("previewable-image");
      if (!image.dataset.previewCaption && image.alt) {
        image.dataset.previewCaption = image.alt;
      }
    });
    bindPreviewableImages(analysisImages);
    state.analysisMarkdownLoaded = true;
  } catch (error) {
    console.error(error);
    analysisMarkdown.textContent = "Failed to load analysis notes.";
  }
}

function openAnalysisModal() {
  if (!analysisModal) {
    return;
  }
  dismissAnalysisHint();
  analysisModal.hidden = false;
  analysisModal.setAttribute("aria-hidden", "false");
  void ensureAnalysisMarkdown();
}

function closeAnalysisModal() {
  if (!analysisModal) {
    return;
  }
  analysisModal.hidden = true;
  analysisModal.setAttribute("aria-hidden", "true");
}

export function bindAnalysisModal() {
  analysisOpenButton?.addEventListener("click", openAnalysisModal);
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
