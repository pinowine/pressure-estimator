let scale = 1; // scale of the canvas

function setupIntroUI() {
  const startBtn = document.getElementById("btn-start");
  const introOverlay = document.getElementById("intro-overlay");
  const bgImg = document.getElementById("bg-img-wrapper");
  const loadingStatus = document.getElementById("loading-status");

  if (!startBtn || !introOverlay) return;

  startBtn.addEventListener("click", () => {
    if (!assetsLoaded) {
      if (loadingStatus) {
        loadingStatus.textContent = "Still loading assets…";
      }
      return;
    }

    // hide intro
    introOverlay.classList.add("hidden");
    bgImg.classList.add("hidden");

    // start game
    startGame();
  });
}

function startGame() {
  // choose what your FIRST scene is
  loadScene(0);
  gameState = "playing";
}

function fitCanvasWrapperHeight() {
  const ui = document.getElementById("ui-wrapper");
  const wrapper = document.getElementById("canvas-wrapper");
  if (!ui || !wrapper) return;

  const uiHeight = ui.getBoundingClientRect().height;
  const availHeight = window.innerHeight - uiHeight;

  wrapper.style.height = `${availHeight}px`;
}

function windowResized() {
  updateCanvasScale();
}

function updateCanvasScale() {
  const wrapper = document.getElementById("canvas-wrapper");
  if (!wrapper) return;

  const wrapperWidth = wrapper.clientWidth;
  const wrapperHeight = wrapper.clientHeight;

  const worldRatio = WORLD_WIDTH / WORLD_HEIGHT;
  const wrapperRatio = wrapperWidth / wrapperHeight;

  let drawWidth, drawHeight;

  if (wrapperRatio > worldRatio) {
    drawHeight = wrapperHeight;
    drawWidth = drawHeight * worldRatio;
  } else {
    drawWidth = wrapperWidth;
    drawHeight = drawWidth / worldRatio;
  }

  scaleFactor = drawWidth / WORLD_WIDTH;

  const c = canvas.elt;
  c.style.width = drawWidth + "px";
  c.style.height = drawHeight + "px";
}
