const scenes = [...document.querySelectorAll(".scene")];
const replayButton = document.querySelector("#replay-button");
const soundButton = document.querySelector("#sound-button");
const sceneCounter = document.querySelector("#scene-counter");
const voiceover = document.querySelector("#voiceover");
const recordingVideos = [...document.querySelectorAll("[data-recording-video]")].map((video) => ({
  video,
  frame: video.closest(".clip-frame, .recording-frame"),
  scene: video.closest(".scene"),
  rate: Number(video.dataset.rate) || 1,
  reveal: Number(video.closest("[data-reveal]")?.dataset.reveal) || 0,
}));
const query = new URLSearchParams(window.location.search);

if (query.get("record") === "1") {
  document.documentElement.classList.add("recording-mode");
}

const TOTAL_TIME = 51;
let elapsed = 0;
let isPlaying = false;
let frameRequest = null;
let lastFrame = 0;
let activeScene = -1;
let hasVoiceover = false;

function sceneForTime(seconds) {
  const index = scenes.findIndex((scene) => {
    const start = Number(scene.dataset.start);
    const end = Number(scene.dataset.end);
    return seconds >= start && seconds < end;
  });
  return index === -1 ? scenes.length - 1 : index;
}

function activateScene(index) {
  if (activeScene === index) return;
  activeScene = index;
  scenes.forEach((scene, sceneIndex) => {
    scene.classList.toggle("is-active", sceneIndex === index);
  });
  sceneCounter.textContent = `${String(index + 1).padStart(2, "0")} / ${String(scenes.length).padStart(2, "0")}`;
}

function syncRecordings(forceSeek = false) {
  recordingVideos.forEach(({ video, frame, scene, rate, reveal }) => {
    const sceneStart = Number(scene.dataset.start);
    const isActive = scene === scenes[activeScene];
    const localTime = Math.max(0, elapsed - sceneStart);
    const shouldShow = isActive && frame.classList.contains("has-video") && localTime >= reveal;

    scene.classList.toggle("show-recording", shouldShow);
    video.playbackRate = rate;

    if (!isActive) {
      video.pause();
      return;
    }

    if (forceSeek && video.readyState >= 1) {
      const duration = Number.isFinite(video.duration) ? video.duration : localTime * rate;
      try {
        video.currentTime = Math.min(localTime * rate, duration);
      } catch {
        // 元数据尚未就绪时，canplay 事件会再次同步进度。
      }
    }

    if (isPlaying && video.readyState >= 2 && video.paused && !video.ended) {
      video.play().catch(() => {});
    }
  });
}

function resetRecordings() {
  recordingVideos.forEach(({ video }) => {
    video.pause();
    try {
      video.currentTime = 0;
    } catch {
      // 元数据尚未就绪时无需处理。
    }
  });
}

function render(forceSeek = false) {
  activateScene(sceneForTime(elapsed));
  syncRecordings(forceSeek);
}

function stopFrameLoop() {
  if (frameRequest !== null) {
    cancelAnimationFrame(frameRequest);
    frameRequest = null;
  }
}

function tick(timestamp) {
  if (!isPlaying) return;
  if (!lastFrame) lastFrame = timestamp;
  elapsed += (timestamp - lastFrame) / 1000;
  lastFrame = timestamp;

  if (elapsed >= TOTAL_TIME) {
    elapsed = TOTAL_TIME;
    isPlaying = false;
    stopFrameLoop();
    voiceover.pause();
    recordingVideos.forEach(({ video }) => video.pause());
  }

  render();
  if (isPlaying) frameRequest = requestAnimationFrame(tick);
}

function play() {
  if (elapsed >= TOTAL_TIME) {
    elapsed = 0;
    voiceover.currentTime = 0;
    resetRecordings();
  }
  isPlaying = true;
  lastFrame = 0;
  if (hasVoiceover) voiceover.play().catch(() => {});
  render(true);
  frameRequest = requestAnimationFrame(tick);
}

function pause() {
  isPlaying = false;
  stopFrameLoop();
  voiceover.pause();
  recordingVideos.forEach(({ video }) => video.pause());
}

function restart() {
  pause();
  elapsed = 0;
  voiceover.currentTime = 0;
  resetRecordings();
  render(true);
  play();
}

replayButton.addEventListener("click", restart);

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, button, a")) return;
  if (event.code === "Space") {
    event.preventDefault();
    restart();
  }
});

soundButton.addEventListener("click", () => {
  if (!hasVoiceover) return;
  voiceover.muted = !voiceover.muted;
  soundButton.classList.toggle("is-muted", voiceover.muted);
  soundButton.textContent = voiceover.muted ? "旁白已静音" : "旁白";
});

recordingVideos.forEach(({ video, frame }) => {
  video.addEventListener("canplay", () => {
    frame.classList.add("has-video");
    syncRecordings(true);
  });

  video.addEventListener("error", () => {
    frame.classList.remove("has-video");
    syncRecordings();
  });
});

voiceover.addEventListener("canplay", () => {
  hasVoiceover = true;
  soundButton.classList.remove("is-muted");
});

voiceover.addEventListener("error", () => {
  hasVoiceover = false;
  soundButton.classList.add("is-muted");
});

render(true);

if (query.get("autoplay") === "1") {
  window.setTimeout(play, 350);
}
