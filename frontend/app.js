const inputText = document.getElementById("inputText");
const inputFile = document.getElementById("inputFile");
const fileStatus = document.getElementById("fileStatus");
const analyzeTextBtn = document.getElementById("analyzeTextBtn");
const analyzeTextAndSentencesBtn = document.getElementById(
  "analyzeTextAndSentencesBtn",
);
const outputText = document.getElementById("outputText");
const highlightedText = document.getElementById("highlightedText");
const arthurTrigger = document.getElementById("arthurTrigger");
const fireworksCanvas = document.getElementById("fireworksCanvas");
const leftLetterField = document.getElementById("leftLetterField");
const rightLetterField = document.getElementById("rightLetterField");
const DEFAULT_BAND_TEXT = "No score yet";
const NEUTRAL_GAUGE_COLOR = "#e6e8ee";
const LOADING_STEP_MS = 30;
const LOADING_INCREMENT = 2;
const FIREWORKS_DURATION_MS = 10000;
const LOADING_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// --- Gauge wiring ---
function scoreBand(score, label) {
  if (score >= 80) return `High (likely ${label})`;
  if (score >= 50) return "Medium";
  return "Low";
}

function scoreColor(score) {
  const clamped = Math.max(0, Math.min(100, Number(score)));
  const hue = 120 - (120 * clamped) / 100; // 120=green, 0=red
  return `hsl(${hue} 80% 42%)`;
}

function setupGauge(valueArcEl) {
  if (!valueArcEl) return 0;
  const arcLen = valueArcEl.getTotalLength();
  valueArcEl.style.strokeDasharray = `${arcLen}`;
  valueArcEl.style.strokeDashoffset = `${arcLen}`;
  return arcLen;
}

function createGauge({
  valueArcId,
  needleId,
  scoreValueId,
  scoreBandId,
  label,
}) {
  const valueArc = document.getElementById(valueArcId);
  const needle = document.getElementById(needleId);
  const needleLine = needle ? needle.querySelector(".gauge-needle-line") : null;
  const gaugeCap = valueArc
    ? valueArc.closest("svg")?.querySelector(".gauge-cap")
    : null;
  const meterScore = scoreValueId
    ? document.getElementById(scoreValueId)?.closest(".meter-score")
    : null;
  const scoreValue = document.getElementById(scoreValueId);
  const scoreBandEl = document.getElementById(scoreBandId);
  const arcLen = setupGauge(valueArc);

  return function setGauge(score) {
    if (score == null) {
      if (valueArc) valueArc.style.strokeDashoffset = `${arcLen}`;
      if (needle) needle.style.transform = "rotate(180deg)";
      if (needleLine) needleLine.style.stroke = NEUTRAL_GAUGE_COLOR;
      if (gaugeCap) gaugeCap.style.fill = NEUTRAL_GAUGE_COLOR;
      if (meterScore) meterScore.style.color = NEUTRAL_GAUGE_COLOR;
      if (scoreValue) scoreValue.textContent = "--";
      if (scoreBandEl) scoreBandEl.textContent = DEFAULT_BAND_TEXT;
      return;
    }

    const clamped = Math.max(0, Math.min(100, Number(score)));
    const frac = clamped / 100;

    const offset = arcLen * (1 - frac);
    if (valueArc) valueArc.style.strokeDashoffset = `${offset}`;

    const angle = 180 + 180 * frac;
    if (needle) needle.style.transform = `rotate(${angle}deg)`;

    const color = scoreColor(clamped);
    if (needleLine) needleLine.style.stroke = color;
    if (gaugeCap) gaugeCap.style.fill = color;
    if (meterScore) meterScore.style.color = color;
    if (scoreValue) scoreValue.textContent = `${Math.round(clamped)}`;
    if (scoreBandEl) scoreBandEl.textContent = scoreBand(clamped, label);
  };
}

const setBsGauge = createGauge({
  valueArcId: "bsGaugeValueArc",
  needleId: "bsGaugeNeedle",
  scoreValueId: "bsScoreValue",
  scoreBandId: "bsScoreBand",
  label: "BS",
});

const setAiGauge = createGauge({
  valueArcId: "aiGaugeValueArc",
  needleId: "aiGaugeNeedle",
  scoreValueId: "aiScoreValue",
  scoreBandId: "aiScoreBand",
  label: "AI",
});

function updateGauges(bsScore, aiScore) {
  setBsGauge(bsScore);
  setAiGauge(aiScore);
}

function resetGauges() {
  setBsGauge(null);
  setAiGauge(null);
}

let loadingAnimationId = null;

function startGaugeLoadingAnimation() {
  stopGaugeLoadingAnimation();

  let score = 0;
  loadingAnimationId = window.setInterval(() => {
    updateGauges(score, score);
    score += LOADING_INCREMENT;
    if (score > 100) score = 0;
  }, LOADING_STEP_MS);
}

function stopGaugeLoadingAnimation() {
  if (loadingAnimationId !== null) {
    window.clearInterval(loadingAnimationId);
    loadingAnimationId = null;
  }
}

function setButtonsDisabled(disabled) {
  analyzeTextBtn.disabled = disabled;
  analyzeTextAndSentencesBtn.disabled = disabled;
}

const letterAnimationState = {
  intervalId: null,
  activeLetters: new Set(),
};

function spawnLoadingLetter(field) {
  if (!field) return;

  const letter = document.createElement("span");
  letter.className = "floating-letter";
  letter.textContent =
    LOADING_LETTERS[Math.floor(Math.random() * LOADING_LETTERS.length)];

  const x = randomBetween(12, 88);
  const y = randomBetween(8, 92);
  const size = randomBetween(16, 42);
  const duration = randomBetween(2200, 3400);

  letter.style.left = `${x}%`;
  letter.style.top = `${y}%`;
  letter.style.fontSize = `${size}px`;
  letter.style.animationDuration = `${duration}ms`;

  field.appendChild(letter);
  letterAnimationState.activeLetters.add(letter);

  letter.addEventListener(
    "animationend",
    () => {
      letterAnimationState.activeLetters.delete(letter);
      letter.remove();
    },
    { once: true },
  );
}

function startLoadingLetters() {
  stopLoadingLetters();

  spawnLoadingLetter(leftLetterField);
  spawnLoadingLetter(rightLetterField);

  letterAnimationState.intervalId = window.setInterval(() => {
    spawnLoadingLetter(Math.random() < 0.5 ? leftLetterField : rightLetterField);

    if (Math.random() < 0.7) {
      spawnLoadingLetter(Math.random() < 0.5 ? leftLetterField : rightLetterField);
    }
  }, 220);
}

function stopLoadingLetters() {
  if (letterAnimationState.intervalId !== null) {
    window.clearInterval(letterAnimationState.intervalId);
    letterAnimationState.intervalId = null;
  }

  for (const letter of letterAnimationState.activeLetters) {
    letter.remove();
  }
  letterAnimationState.activeLetters.clear();
}

const fireworksState = {
  ctx: fireworksCanvas ? fireworksCanvas.getContext("2d") : null,
  particles: [],
  running: false,
  frameId: null,
  spawnId: null,
  stopId: null,
  lastFrameTime: 0,
};

function resizeFireworksCanvas() {
  if (!fireworksCanvas) return;
  const ratio = window.devicePixelRatio || 1;
  fireworksCanvas.width = Math.floor(window.innerWidth * ratio);
  fireworksCanvas.height = Math.floor(window.innerHeight * ratio);
  fireworksCanvas.style.width = `${window.innerWidth}px`;
  fireworksCanvas.style.height = `${window.innerHeight}px`;
  if (fireworksState.ctx) {
    fireworksState.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function spawnFireworkBurst() {
  if (!fireworksState.running) return;

  const burstX = randomBetween(window.innerWidth * 0.1, window.innerWidth * 0.9);
  const burstY = randomBetween(window.innerHeight * 0.1, window.innerHeight * 0.55);
  const colors = ["#ff6b6b", "#ffd93d", "#6bcBef", "#c77dff", "#7ae582"];
  const particleCount = Math.floor(randomBetween(18, 32));

  for (let i = 0; i < particleCount; i += 1) {
    const angle = (Math.PI * 2 * i) / particleCount + randomBetween(-0.15, 0.15);
    const speed = randomBetween(90, 220);
    fireworksState.particles.push({
      x: burstX,
      y: burstY,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: randomBetween(0.9, 1.4),
      age: 0,
      size: randomBetween(2, 4),
      color: colors[Math.floor(Math.random() * colors.length)],
    });
  }
}

function stopFireworks() {
  fireworksState.running = false;

  if (fireworksState.spawnId !== null) {
    window.clearInterval(fireworksState.spawnId);
    fireworksState.spawnId = null;
  }

  if (fireworksState.stopId !== null) {
    window.clearTimeout(fireworksState.stopId);
    fireworksState.stopId = null;
  }
}

function renderFireworksFrame(timestamp) {
  if (!fireworksState.ctx || !fireworksCanvas) return;

  if (!fireworksState.lastFrameTime) {
    fireworksState.lastFrameTime = timestamp;
  }

  const delta = Math.min((timestamp - fireworksState.lastFrameTime) / 1000, 0.05);
  fireworksState.lastFrameTime = timestamp;

  fireworksState.ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  fireworksState.particles = fireworksState.particles.filter((particle) => {
    particle.age += delta;
    if (particle.age >= particle.life) return false;

    particle.x += particle.vx * delta;
    particle.y += particle.vy * delta;
    particle.vy += 180 * delta;

    const alpha = 1 - particle.age / particle.life;
    fireworksState.ctx.beginPath();
    fireworksState.ctx.fillStyle = particle.color;
    fireworksState.ctx.globalAlpha = alpha;
    fireworksState.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
    fireworksState.ctx.fill();
    return true;
  });

  fireworksState.ctx.globalAlpha = 1;

  if (fireworksState.running || fireworksState.particles.length > 0) {
    fireworksState.frameId = window.requestAnimationFrame(renderFireworksFrame);
    return;
  }

  fireworksState.frameId = null;
  fireworksState.lastFrameTime = 0;
  fireworksState.ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
}

function startFireworks() {
  if (!fireworksState.ctx) return;

  resizeFireworksCanvas();
  stopFireworks();
  fireworksState.particles = [];
  fireworksState.running = true;
  fireworksState.lastFrameTime = 0;

  spawnFireworkBurst();
  fireworksState.spawnId = window.setInterval(spawnFireworkBurst, 320);
  fireworksState.stopId = window.setTimeout(stopFireworks, FIREWORKS_DURATION_MS);

  if (fireworksState.frameId === null) {
    fireworksState.frameId = window.requestAnimationFrame(renderFireworksFrame);
  }
}

let uploadedFileText = "";
let selectedFile = null;

inputFile.addEventListener("change", async (event) => {
  const [file] = event.target.files || [];

  if (!file) {
    uploadedFileText = "";
    selectedFile = null;
    fileStatus.textContent = "";
    return;
  }

  if (!isSupportedFile(file)) {
    uploadedFileText = "";
    selectedFile = null;
    inputFile.value = "";
    fileStatus.textContent = "Please upload a .txt, .md, .csv, or .pdf file.";
    return;
  }

  selectedFile = file;

  // ✅ PDF: send multipart/form-data to Flask
  if (isPdfFile(file)) {
    fileStatus.textContent = "Uploading PDF for text extraction...";

    try {
      const formData = new FormData();
      formData.append("file", file); // MUST be "file" to match Flask

      const res = await fetch("/api/pdf", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        fileStatus.textContent =
          "Error processing PDF. Please try another file.";
        uploadedFileText = "";
        return;
      }

      const data = await res.json();
      uploadedFileText = data.text || "";
      fileStatus.textContent = `Loaded (PDF): ${file.name}`;

      // Always replace textarea content with the latest uploaded file.
      inputText.value = uploadedFileText;

      return; // IMPORTANT: don't fall through and readAsText(pdf)
    } catch (err) {
      uploadedFileText = "";
      fileStatus.textContent = "Error processing PDF. Please try another file.";
      return;
    }
  }

  // ✅ Non-PDF text-like files: read locally
  try {
    uploadedFileText = await readFileAsText(file);
    fileStatus.textContent = `Loaded: ${file.name}`;
    // Always replace textarea content with the latest uploaded file.
    inputText.value = uploadedFileText;
  } catch (error) {
    uploadedFileText = "";
    fileStatus.textContent =
      "Could not read that file. Please try another one.";
  }
});

async function runAnalysis(mode) {
  const text = inputText.value.trim() || uploadedFileText.trim();

  if (!text) {
    outputText.textContent =
      "Please enter text or upload a text/PDF file first.";
    if (highlightedText) highlightedText.innerHTML = "";
    return;
  }

  resetGauges();
  outputText.textContent = "Analyzing...";
  if (highlightedText) highlightedText.innerHTML = "";
  setButtonsDisabled(true);
  startGaugeLoadingAnimation();
  startLoadingLetters();

  try {
    const result =
      mode === "text-only"
        ? await analyzeWholeTextWithApi(text)
        : await analyzeTextWithApi(text);
    const overallBs = Number(result.overall_bs ?? result.score_bs ?? 0);
    const overallAi = Number(result.overall_ai ?? result.score_ai ?? 0);
    outputText.textContent = `BS: ${Math.round(overallBs)} | AI: ${Math.round(overallAi)}`;

    if (mode === "text-and-sentences") {
      renderSentenceHighlights(text, result.bs_highlight, result.ai_highlight);
    }
  } catch (e) {
    console.error("API error:", e);
    outputText.textContent = "Error calling API. Check server logs.";
    if (highlightedText) highlightedText.innerHTML = "";
  } finally {
    stopGaugeLoadingAnimation();
    stopLoadingLetters();
    setButtonsDisabled(false);
  }
}

analyzeTextBtn.addEventListener("click", () => runAnalysis("text-only"));
analyzeTextAndSentencesBtn.addEventListener("click", () =>
  runAnalysis("text-and-sentences"),
);
arthurTrigger?.addEventListener("click", startFireworks);
window.addEventListener("resize", resizeFireworksCanvas);

async function analyzeWholeTextWithApi(text) {
  const res = await fetch("/api/plaintext", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    // non-JSON response (e.g. HTML error page)
  }

  if (!res.ok) {
    const msg = data?.error || `API error (${res.status})`;
    throw new Error(msg);
  }

  const overallBs = Number(data.overall_bs ?? data.score_bs ?? 0);
  const overallAi = Number(data.overall_ai ?? data.score_ai ?? 0);
  updateGauges(overallBs, overallAi);
  return data;
}

async function analyzeTextWithApi(text) {
  const res = await fetch("/api/analyse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  // Try to parse JSON even on errors (so you can show message)
  let data = null;
  try {
    data = await res.json();
  } catch {
    // non-JSON response (e.g. HTML error page)
  }

  if (!res.ok) {
    const msg = data?.error || `API error (${res.status})`;
    throw new Error(msg);
  }

  // Expect: {"overall_bs": ..., "bySentence_bs": ..., "overall_ai": ..., "bySentence_ai": ...}
  const overallBs = Number(data.overall_bs ?? data.score_bs ?? 0);
  const overallAi = Number(data.overall_ai ?? data.score_ai ?? 0);
  updateGauges(overallBs, overallAi);
  return data;
}

function extractSentenceScores(bySentence) {
  if (!Array.isArray(bySentence)) return [];
  return bySentence.map((entry) => {
    if (Array.isArray(entry)) return Number(entry[0]) || 0;
    if (entry && typeof entry === "object" && "score" in entry) {
      return Number(entry.score) || 0;
    }
    return Number(entry) || 0;
  });
}

function splitTextForHighlighting(text) {
  const chunks = [];
  let start = 0;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "." || ch === "!" || ch === "?" || ch === "\n") {
      chunks.push(text.slice(start, i + 1));
      start = i + 1;
    }
  }

  if (start < text.length) chunks.push(text.slice(start));
  return chunks;
}

function normalizeScoreLength(scores, sentenceCount) {
  if (scores.length === sentenceCount + 1) {
    return scores.slice(0, sentenceCount);
  }
  if (scores.length > sentenceCount) {
    return scores.slice(0, sentenceCount);
  }
  return scores;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 *
 * @param {string} text
 * @param {Array} bsHighlights
 * @param {Array} aiHighlights
 * @returns
 */
function renderSentenceHighlights(text, bsHighlights, aiHighlights) {
  if (!highlightedText) return;

  let sortedBsHighlights = [...(bsHighlights || [])];
  sortedBsHighlights.sort(([s1, e1], [s2, e2]) => s2 - s1);
  let sortedAiHighlights = [...(aiHighlights || [])];
  sortedAiHighlights.sort(([s1, e1], [s2, e2]) => s2 - s1);
  // sorted are reverse sorted by start

  let idx = 0;
  let parsedArr = [];
  let nextBsSentenceIdx = sortedBsHighlights.pop() ?? [Infinity, Infinity];
  let nextAiSentenceIdx = sortedAiHighlights.pop() ?? [Infinity, Infinity];
  while (idx < text.length) {
    let badMarker;
    let badType;

    if (nextAiSentenceIdx[0] < nextBsSentenceIdx[0]) {
      badMarker = nextAiSentenceIdx;
      badType = "ai";
      nextAiSentenceIdx = sortedAiHighlights.pop() ?? [Infinity, Infinity];
    } else if (nextAiSentenceIdx[0] > nextBsSentenceIdx[0]) {
      badMarker = nextBsSentenceIdx;
      badType = "bs";
      nextBsSentenceIdx = sortedBsHighlights.pop() ?? [Infinity, Infinity];
    } else {
      if (
        nextAiSentenceIdx[0] == Infinity &&
        nextBsSentenceIdx[0] == Infinity
      ) {
        parsedArr.push([text.substring(idx), "good"]);
        break;
      }
      badMarker = nextAiSentenceIdx;
      badType = "both";
      nextBsSentenceIdx = sortedBsHighlights.pop() ?? [Infinity, Infinity];
      nextAiSentenceIdx = sortedAiHighlights.pop() ?? [Infinity, Infinity];
    }

    if (badMarker[0] > idx) {
      parsedArr.push([text.substring(idx, badMarker[0]), "good"]);
    }
    // end indexes are of the punctuiation mark, which needs to be included
    // start indexes are of the letter
    parsedArr.push([text.substring(badMarker[0], badMarker[1] + 1), badType]);

    idx = badMarker[1] + 1;
  }
  console.log(parsedArr);
  // now have a parsed array
  const asHtml = parsedArr
    .map(([subtext, tag]) => {
      let safeSubtext = escapeHtml(subtext);
      const classParam = tag == "good" ? "" : `class="hl-${escapeHtml(tag)}"`;
      return `<span ${classParam}>${safeSubtext}</span>`;
    })
    .join("");
  console.log(asHtml);
  highlightedText.innerHTML = asHtml;
}

resetGauges();
resizeFireworksCanvas();

function isTextLikeFile(file) {
  return file.type.startsWith("text/") || /\.(txt|md|csv)$/i.test(file.name);
}

function isPdfFile(file) {
  return file.type === "application/pdf" || /\.pdf$/i.test(file.name);
}

function isSupportedFile(file) {
  return isTextLikeFile(file) || isPdfFile(file);
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}
