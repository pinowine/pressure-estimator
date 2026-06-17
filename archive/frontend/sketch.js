let canvas;
let gameState = "intro";

function setup() {
  canvas = createCanvas(WORLD_WIDTH, WORLD_HEIGHT); // Create a canvas that fills the window
  canvas.parent("canvas-wrapper");
  fitCanvasWrapperHeight();
  updateCanvasScale();

  setupIntroUI();
  loadAssets();
}

function draw() {
  background(0);

  if (!assetsLoaded || gameState === "intro" || !scene) return;

  // logic
  player.update(scene);
  snakes.forEach((snake) => snake.update(player));

  // rendering
  scene.draw();
  player.draw(currentSceneType);
  snakes.forEach((snake) => snake.draw(currentSceneType));

  // checks
  checkSnakeEatPlayer();
  checkPLayerReachedEnd();
}
