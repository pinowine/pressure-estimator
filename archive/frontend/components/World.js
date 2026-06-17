const WORLD_WIDTH = 1600;
const WORLD_HEIGHT = 880;

const TILE_SIZE = 40;
const WORLD_COLS = WORLD_WIDTH / TILE_SIZE;
const WORLD_ROWS = WORLD_HEIGHT / TILE_SIZE;

const SPAWN_TILE = { col: 0, row: 11 };
const END_TILE = { col: WORLD_COLS - 1, row: 11 };
const CHECK_RADIUS = TILE_SIZE * 0.8;

let debugTiles = false;

class Scene {
  constructor(layerData, layoutData, tilesetData, tilesetImage, bgImage) {
    this.tileset = tilesetData;
    this.layout = layoutData;
    this.tilesetImage = tilesetImage;
    this.bgImage = bgImage;

    const meta = this.tileset.meta || {};
    const sheetCols = meta.sheetSize ? meta.sheetSize[0] : 1;
    const sheetRows = meta.sheetSize ? meta.sheetSize[1] : 1;

    if (this.tilesetImage) {
      meta.srcTileWidth = this.tilesetImage.width / sheetCols;
      meta.srcTileHeight = this.tilesetImage.height / sheetRows;
      meta.tileSize = meta.srcTileWidth;
    }

    this.tileset.meta = meta;

    // console.log(layerData);
    this.layers = layerData.layers.map((layerObj) => {
      const sourceName = layerObj.source; // get source name
      const layerData = this.layout[sourceName];
      return new Layer(layerObj, layerData, this.tileset, this.tilesetImage);
    });

    this.collisionLayer = this.layers.find((l) => l.type === "collision");
    this.wallLayer = this.layers.find((l) => l.type === "wall");
    this.bgLayer = this.layers.find((l) => l.type === "bg");
  }

  // buffered  rendering
  render() {
    for (const layer of this.layers) {
      layer.renderToBuffer();
    }
  }
  draw() {
    if (this.bgLayer && this.bgImage) {
      const masked = this.bgImage.get();
      masked.mask(this.bgLayer.maskBuffer);
      image(masked, 0, 0, width, height);
    }
    // decoupling layers to render in a right order
    for (const layer of this.layers) {
      if (layer.type === "wall") layer.draw();
    }
    for (const layer of this.layers) {
      if (layer.type === "collision") layer.draw();
    }
  }

  // global collision check
  isSolidAt(x, y) {
    const col = floor(x / TILE_SIZE);
    const row = floor(y / TILE_SIZE);
    if (!this.collisionLayer) return false;
    const tile = this.collisionLayer.getTile(col, row);
    return tile ? tile.solid : false;
  }

  // sight check
  hasLineOfSight(a, b) {
    const ax = a.x;
    const ay = a.y;
    const bx = b.x;
    const by = b.y;
    const dx = bx - ax;
    const dy = by - ay;
    const d = sqrt(dx * dx + dy * dy);
    if (d === 0) return true;

    const step = TILE_SIZE * 0.25;
    const steps = floor(d / step);

    for (let i = 0; i < steps; i++) {
      const x = ax + (dx * i) / steps;
      const y = ay + (dy * i) / steps;
      if (this.isSolidAt(x, y)) return false;
    }
    return true;
  }
}

// class for a single layer of the world
class Layer {
  constructor(layerObj, layoutData, tileset, tilesetImage) {
    this.layer = layerObj.layer;
    this.type = layerObj.type;
    this.tileset = tileset;
    this.tilesetImage = tilesetImage;
    this.buffer = createGraphics(WORLD_WIDTH, WORLD_HEIGHT);

    if (this.type === "bg") {
      this.maskBuffer = createGraphics(WORLD_WIDTH, WORLD_HEIGHT);
    }

    this.tiles = this.createTiles(layoutData);

    this.renderToBuffer();
  }

  // layout data: 0(invalid)/1(valid)
  createTiles(layoutData) {
    const tiles = [];
    for (let row = 0; row < WORLD_ROWS; row++) {
      for (let col = 0; col < WORLD_COLS; col++) {
        const tileId = layoutData[row][col];
        // console.log(tileId);
        tiles.push(
          new Tile(col, row, tileId, this.tileset, this.type, this.tilesetImage)
        );
      }
    }
    return tiles;
  }

  // offscreen rendering
  renderToBuffer() {
    const buf = this.buffer;
    buf.clear();
    buf.noStroke();

    // bg layer is a single image with mask
    if (this.type === "bg" && this.maskBuffer) {
      this.maskBuffer.clear();
      this.maskBuffer.background(0);
      this.maskBuffer.noStroke();
    }

    for (const tile of this.tiles) {
      tile.draw(buf);
      if (this.type === "bg" && this.maskBuffer) {
        tile.drawMask(this.maskBuffer);
      }
    }

    // add a blur effect to the wall layer
    if (this.type === "wall") {
      this.buffer.filter(BLUR, 5);
    }
  }

  // draw the layer buffer to the main canvas
  draw() {
    if (this.type === "bg") return;
    image(this.buffer, 0, 0, width, height);
  }

  // get specific tile at (col, row)
  getTile(col, row) {
    if (row < 0 || row >= WORLD_ROWS || col < 0 || col >= WORLD_COLS)
      return null;
    return this.tiles[col + row * WORLD_COLS];
  }
}

class Tile {
  constructor(x, y, tileId, tileset, layerType, tilesetImage) {
    this.x = x;
    this.y = y;
    this.id = tileId;
    this.tileset = tileset;
    this.layerType = layerType;
    this.tilesetImage = tilesetImage;

    const meta = tilesetConfig.meta || {};
    const tilesDef = tilesetConfig.tiles || tilesetConfig;

    this.def = tilesDef[String(tileId)] || null;
    this.columns = meta.sheetSize ? meta.sheetSize[0] : 1;
    this.rows = meta.sheetSize ? meta.sheetSize[1] : 1;
    this.srcSize = meta.tileSize || TILE_SIZE;

    this.solid = !!(this.def && this.def.solid);
  }
  draw(buf) {
    if (!this.def) return; // skip if tile not found

    const useSprite =
      this.tilesetImage &&
      this.def.spriteVariants != null &&
      this.layerType === "collision" &&
      this.id !== 0;

    // for default scene, only draw wall tiles
    const useRectFill =
      (this.layerType === "wall" &&
        currentSceneType === "default" &&
        this.id !== 0) ||
      // for abstract scene, draw all tiles
      (this.layerType === "wall" && currentSceneType !== "default");

    if (useSprite) {
      const sSize = this.srcSize;
      let sx, sy;

      if (Array.isArray(this.def.spriteVariants)) {
        const [iy, ix] =
          this.def.spriteVariants[
            floor(random(this.def.spriteVariants.length))
          ];
        sx = ix * sSize;
        sy = iy * sSize;
      } else {
        const idx = this.def.spriteIndex;
        sx = (idx % this.columns) * sSize;
        sy = Math.floor(idx / this.columns) * sSize;
      }

      const dx = this.x * TILE_SIZE;
      const dy = this.y * TILE_SIZE;

      buf.image(
        this.tilesetImage,
        dx,
        dy,
        TILE_SIZE,
        TILE_SIZE,
        sx,
        sy,
        sSize,
        sSize
      );
    } else if (useRectFill) {
      let colr = [0, 0, 0, 150];
      if (currentSceneType === "devmode" || currentSceneType === "abstract") {
        colr = [210, 210, 210, 255];
        if (this.id === 0) colr = [225, 225, 225, 255];
      }

      if (colr) {
        const [r, g, b, a = 255] = colr;
        buf.push();
        buf.noStroke();
        buf.fill(r, g, b, a);
        buf.rect(this.x * TILE_SIZE, this.y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
        buf.pop();
      }
    }
  }

  drawMask(buf) {
    if (this.id === 0) {
      return;
    }
    buf.push();
    buf.noStroke();
    buf.fill(255);
    buf.rect(this.x * TILE_SIZE, this.y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
    buf.pop();
  }
}
