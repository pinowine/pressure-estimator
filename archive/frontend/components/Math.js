function angleNormalize(a) {
  a = (a + PI) % (TWO_PI);
  if (a <= 0) a += TWO_PI;
  return a - PI;
}

// function to generate a personality object with random trait values
function generatePersonality() {
  return {
    Se: floor(random(1, 100)),
    Si: floor(random(1, 100)),
    Ne: floor(random(1, 100)),
    Ni: floor(random(1, 100)),
    Te: floor(random(1, 100)),
    Ti: floor(random(1, 100)),
    Fe: floor(random(1, 100)),
    Fi: floor(random(1, 100))
  }
}

// function to map personality traits to configuration parameters
function mapPersonalityToConfig(p) {
  return {
    // movement
    moveSpeed: map((p.Se + p.Te) / 2, 1, 100, 1, 5),
    turnSpeed: map(p.Se, 1, 100, 0.05, 0.55),

    // sensation
    hearingRange: map(p.Fe, 1, 100, 10 * TILE_SIZE, 20 * TILE_SIZE),
    visionRange: map(p.Ni, 1, 100, 5 * TILE_SIZE, 10 * TILE_SIZE),
    visionFov: map(p.Ti, 1, 100, radians(60), radians(120)),

    // cognition
    hearingConfidenceBias: map((p.Ti + p.Ni) / 2, 1, 100, 0.2, 0.8),
    maxChaseTime: map(p.Si, 1, 100, 1000, 6000),
    loseInterestTime: map(p.Ne, 1, 100, 1000, 4000),
    distractionChance: map((p.Ne + p.Fi) / 2, 1, 100, 0.01, 0.12),

    // behavior
    attackCooldown: map((p.Te + p.Ti) / 2, 1, 100, 1500, 400),
    attackRange: map(p.Se, 1, 100, 18, 40),

    // appearance
    bodySegments: floor(map(p.Si, 1, 100, 30, 60)),
    bodyThickness: map(p.Fi, 1, 100, 10, 20),

    // physics
    mass: map((p.Si + p.Fi) / 2, 1, 100, 1.0, 3.0),
    maxCeilingTime: map((p.Ni - p.Fi), -99, 99, 1000, 5000),
    canCeilingCrawl: p.Se > 40,
    adhesion: map((p.Ni + p.Fe) / 2, 1, 100, 0.8, 1.5)
  }
}

// convert world coordinates to tile coordinates
function worldToTile(x, y) {
  return {
    col: floor(x / TILE_SIZE),
    row: floor(y / TILE_SIZE)
  };
}

// convert tile coordinates to world coordinates
function tileToWorld(col, row) {
  return {
    x: (col + 0.5) * TILE_SIZE,
    y: (row + 0.5) * TILE_SIZE
  };
}