let sceneConfig;
let layoutConfig;
let tilesetConfig;
let layerConfig;

let scene;
let player;
let snakes = [];

let tilesetImages = [];
let bgImages = [];
let length;

let currentSceneType;

// loading state
let assetsLoaded = false;

function loadImageAsync(src) {
  return new Promise((resolve, reject) => {
    loadImage(
      src,
      (img) => resolve(img),
      (error) => reject(error)
    );
  });
}

async function loadAssets() {
  try {
    const [layoutResponse, tilesetResponse, layerResponse, sceneResponse] =
      await Promise.all([
        fetch("assets/data/layouts.json"),
        fetch("assets/data/tilesets.json"),
        fetch("assets/data/layers.json"),
        fetch("assets/data/scenes.json"),
      ]);
    layoutConfig = await layoutResponse.json();
    tilesetConfig = await tilesetResponse.json();
    layerConfig = await layerResponse.json();
    sceneConfig = await sceneResponse.json();

    const imagePromises = [];
    sceneConfig.scenes.forEach((def) => {
      imagePromises.push(
        loadImageAsync(def.tilesprite).then((img) => {
          tilesetImages[def.id] = img;
        })
      );
      imagePromises.push(
        loadImageAsync(def.bg).then((img) => {
          bgImages[def.id] = img;
        })
      );
    });
    length = sceneConfig.scenes.length;
    await Promise.all(imagePromises);
    assetsLoaded = true;
    console.log("All assets loaded.");
    const loadingStatus = document.getElementById("loading-status");
    if (loadingStatus) {
      loadingStatus.textContent = "Assets ready. Click “Start Game”.";
    }
  } catch (error) {
    console.error("Error loading assets:", error);
    const loadingStatus = document.getElementById("loading-status");
    if (loadingStatus) {
      loadingStatus.textContent = "Failed to load assets (see console).";
    }
  }
}

function loadScene(index) {
  // console.log(sceneConfig);
  const scenes = sceneConfig.scenes;

  let def;
  def = scenes[index];
  currentSceneType = def.type;
  const tilesetImage = tilesetImages[def.id];
  const bgImage = bgImages[def.id];
  // console.log(bgImage);

  scene = new Scene(
    layerConfig,
    layoutConfig,
    tilesetConfig,
    tilesetImage,
    bgImage
  );

  resetEntities();
}

function resetEntities() {
  snakes = []; // clear snakes

  // reset player to the spawen tile
  const spawnPos = tileToWorld(SPAWN_TILE.col, SPAWN_TILE.row);
  player = new Player(spawnPos.x, spawnPos.y);

  // reset snakes
  const snakeNum = floor(random(4, 8));
  for (let i = 0; i < snakeNum; i++) {
    snakes.push(spawnSnake(scene));
  }
}

function spawnSnake(scene) {
  // get the wall and collision grid
  const wallGrid = scene.layout.wall;
  const collisionGrid = scene.layout.collision;

  // try to spawn snake 200 times
  const maxTry = 200;

  for (let i = 0; i < maxTry; i++) {
    // get a random tile
    const row = floor(random(0, WORLD_ROWS));
    const col = floor(random(0, WORLD_COLS));

    const wallId = wallGrid[row][col];
    const collisionId = collisionGrid[row][col];

    // check if the tile is a wall and not a collision tile
    if (wallId > 0 && collisionId === 0) {
      // spawn snake on the center of the tile
      const pos = {
        x: col * TILE_SIZE + TILE_SIZE * 0.5,
        y: row * TILE_SIZE + TILE_SIZE * 0.5,
      };

      const snake = new Snake(scene, pos.x, pos.y);
      return snake;
    }
  }

  return null;
}

function checkSnakeEatPlayer() {
  if (!player || !snakes) return;

  snakes.forEach((snake) => {
    const head = snake.body.head;
    const pPos = player.body.pos;

    const snakeHeadRadius = snake.body.thickness;
    const playerRadius = player.body.radius;

    const d = dist(head.x, head.y, pPos.x, pPos.y);

    // the max distance that the snake can eat the player
    const eatThreshold = snakeHeadRadius + playerRadius * 0.5;

    if (d < eatThreshold) {
      console.log("Snake ate the player!");
      resetEntities(); // same scene but reset entities
    }
  });
}

function checkPLayerReachedEnd() {
  const endPos = tileToWorld(END_TILE.col, END_TILE.row);
  const pPos = player.body.pos;

  const d = dist(endPos.x, endPos.y, pPos.x, pPos.y);

  if (d < CHECK_RADIUS) {
    console.log("Player reached the end!");
    // get a random scene type
    loadScene(floor(random(0, length)));
  }
}
