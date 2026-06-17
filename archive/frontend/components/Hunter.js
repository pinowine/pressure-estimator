class Snake {
  constructor(scene, x, y, personality = generatePersonality()) {
    this.scene = scene;
    this.personality = personality;
    this.config = mapPersonalityToConfig(personality);

    this.nav = new SnakeNav(scene);
    this.pathfinder = new AStarPathfinder(this.nav);

    this.body = new SnakeBody(this.nav, x, y, this.config);
    this.sense = new SnakeSense(scene, this.body, this.config);
    this.mind = new SnakeMind(this.body, this.sense, this.config);

    this.lastTargetTile = null;
  }

  update(player) {
    // update sense and mind
    this.sense.update(player);
    this.mind.update(player);

    const target = this.mind.getTarget();

    if (target) {
      // first find the tile of snake head and player
      const headTile = worldToTile(this.body.head.x, this.body.head.y);
      const targetTile = worldToTile(target.x, target.y);

      // only when: no path / path finished / player tile changed, recalculate A*
      const keyNow = `${targetTile.col},${targetTile.row}`;
      const keyLast = this.lastTargetTile;

      if (
        !this.body.currentPath ||
        this.body.pathIndex >= this.body.currentPath.length ||
        keyNow !== keyLast
      ) {
        const path = this.pathfinder.findPath(
          headTile.col,
          headTile.row,
          targetTile.col,
          targetTile.row
        );

        if (path) {
          this.body.setPath(path);
          this.lastTargetTile = keyNow;
        }
      }
    }

    this.body.update(deltaTime);
  }

  draw(mode = "default") {
    let alertState = "idle";

    if (this.sense.lastSeenPos && this.sense.seeTimer < 200) {
      alertState = "seen";
    } else if (this.sense.lastHeardPos && this.sense.lastHeardStrength > 0.3) {
      alertState = "heard";
    }

    this.body.draw(alertState, mode);

    if (mode === "devmode") {
      this.drawPath();
      this.drawSensors();
    }
  }

  // helping functions
  drawPath() {
    const path = this.body.currentPath;
    if (!path) return;

    push();
    stroke(0, 255, 255, 150);
    strokeWeight(10);
    noFill();
    beginShape();
    // draw path
    path.forEach((tile) => {
      const pos = tileToWorld(tile.col, tile.row);
      vertex(pos.x, pos.y);
    });
    endShape();
    pop();
  }

  drawSensors() {
    const head = this.body.head;
    const config = this.config;

    push();
    noFill();

    // hearing range
    stroke(255, 255, 0, 80);
    strokeWeight(4);
    ellipse(head.x, head.y, config.hearingRange * 2);

    // vision range
    stroke(0, 255, 0, 80);
    strokeWeight(4);
    ellipse(head.x, head.y, config.visionRange * 2);

    // vision fov
    const faceDir = this.body.getFacingDir();
    const base = atan2(faceDir.y, faceDir.x);
    const half = config.visionFov * 0.5;
    stroke(0, 255, 0, 150);
    line(
      head.x,
      head.y,
      head.x + cos(base - half) * config.visionRange,
      head.y + sin(base - half) * config.visionRange
    );
    line(
      head.x,
      head.y,
      head.x + cos(base + half) * config.visionRange,
      head.y + sin(base + half) * config.visionRange
    );

    // last seen
    if (this.sense.lastSeenPos) {
      stroke(255, 0, 0);
      strokeWeight(4);
      point(this.sense.lastSeenPos.x, this.sense.lastSeenPos.y);
    }

    // last heard
    if (this.sense.lastHeardPos) {
      stroke(255, 255, 0);
      strokeWeight(5);
      point(this.sense.lastHeardPos.x, this.sense.lastHeardPos.y);
    }

    pop();
  }
}

// navigation
class SnakeNav {
  constructor(scene) {
    this.scene = scene;
    this.cols = WORLD_COLS;
    this.rows = WORLD_ROWS;

    this.walkable = this.createWalkableMap();
    // console.log("Sample walkable rows:", this.walkable[10], this.walkable[11]);
  }

  createWalkableMap() {
    const map = [];
    for (let row = 0; row < this.rows; row++) {
      map[row] = [];
      for (let col = 0; col < this.cols; col++) {
        map[row][col] = this.computeWalkable(col, row);
      }
    }
    return map;
  }

  computeWalkable(col, row) {
    const wallId = this.scene.layout.wall[row][col];
    const bgId = this.scene.layout.bg[row][col];
    const collisionId = this.scene.layout.collision[row][col];

    const hasWall = wallId > 0;
    const hasBg = bgId > 0;
    const hasCollision = collisionId > 0;

    // snake can only walk on walls
    if (hasWall && !hasCollision) {
      return true;
    }

    return false;
  }

  // API
  isWalkable(col, row) {
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) {
      return false;
    }
    return this.walkable[row][col];
  }
}

// physics and movement of the snake
class SnakeBody {
  constructor(nav, x, y, config) {
    this.nav = nav;
    this.config = config;

    this.head = createVector(x, y);
    this.facingDir = createVector(1, 0);
    this.speed = config.moveSpeed;

    this.currentPath = null;
    this.pathIndex = 0;

    this.segmentNum = config.bodySegments;
    this.thickness = config.bodyThickness;
    this.segmentSpacing = this.thickness * 0.3;
    this.initSegments();
  }

  setPath(path) {
    this.currentPath = path;
    this.pathIndex = 0;
  }

  update(dt = 16) {
    if (!this.currentPath || this.pathIndex >= this.currentPath.length) {
      this.updateSegments();
      return;
    }

    const tile = this.currentPath[this.pathIndex];
    const targetWorld = tileToWorld(tile.col, tile.row);

    const target = createVector(targetWorld.x, targetWorld.y);
    const dir = p5.Vector.sub(target, this.head);
    const d = dir.mag();

    const step = this.speed * TILE_SIZE * dt * 0.001;

    if (d < step) {
      this.head.set(target.x, target.y);
      this.pathIndex++;
    } else {
      dir.normalize();
      this.facingDir.set(dir.x, dir.y);
      this.head.add(dir.mult(step));
    }

    if (this.facingDir.magSq() === 0) {
      this.facingDir.set(1, 0);
    }

    this.updateSegments();
  }

  initSegments() {
    this.segments = [];

    for (let i = 0; i < this.segmentNum; i++) {
      this.segments.push(this.head.copy());
    }
  }

  // update following body segments positions
  updateSegments() {
    const spacing = this.segmentSpacing;

    this.segments[0].set(this.head.x, this.head.y);

    for (let i = 1; i < this.segmentNum; i++) {
      const prev = this.segments[i - 1];
      const curr = this.segments[i];

      const dir = p5.Vector.sub(prev, curr);
      let d = dir.mag();
      if (d < 0.0001) continue;

      const extra = d - spacing;
      if (extra > 0) {
        dir.setMag(extra);
        curr.add(dir);
      }
    }
  }

  // api
  getFacingDir() {
    return this.facingDir;
  }

  setSpeedMultiplier(m) {
    this.speed = this.config.moveSpeed * m;
  }

  draw(alertState = "idle", mode = "default") {
    push();

    // color settings
    let baseColor;
    switch (alertState) {
      case "seen":
        baseColor = color(196, 51, 2);
        break;
      case "heard":
        baseColor = color(237, 170, 37);
        break;
      default:
        baseColor = color(10, 155, 155);
        break;
    }
    let pupilColor;
    switch (alertState) {
      case "seen":
        pupilColor = color(255, 40, 40);
        break;
      case "heard":
        pupilColor = color(255, 220, 80);
        break;
      default:
        pupilColor = color(40, 80, 40);
        break;
    }

    // calculations
    const headRadius = this.thickness;
    const tailRadius = this.thickness * 0.4;
    const n = this.segments.length;
    const head = this.segments[0] || this.head;
    const headR = headRadius;
    const dir = this.facingDir.copy();
    if (dir.magSq() === 0) {
      dir.set(1, 0);
    }
    dir.setMag(headR * 0.6);
    // const side = createVector(-dir.y, dir.x);
    // side.setMag(headR * 0.45);
    const eyeBase = p5.Vector.add(head, dir * 0.8);
    // const leftEye = p5.Vector.add(eyeBase, side);
    // const rightEye = p5.Vector.sub(eyeBase, side);
    const pupilOffset = dir.copy().setMag(headR * 0.25);
    // const leftPupil = p5.Vector.add(leftEye, pupilOffset);
    // const rightPupil = p5.Vector.add(rightEye, pupilOffset);
    const pupil = p5.Vector.add(eyeBase, pupilOffset);

    // main drawing
    switch (mode) {
      case "default":
        // draw segments
        noStroke();
        for (let i = 0; i < n; i++) {
          const seg = this.segments[i];
          const t = n > 1 ? i / (n - 1) : 0; // percentage of segment length
          // calculate segment properties
          const r = lerp(headRadius, tailRadius, t); // smooth transition between head and tail radius
          const alpha = lerp(255, 120, t);
          // calculate color
          const c = color(
            red(baseColor),
            green(baseColor),
            blue(baseColor),
            alpha
          );
          fill(c);
          circle(seg.x, seg.y, r * 2);
        }
        // draw eyes
        fill(255);
        stroke(0, 80);
        strokeWeight(1);
        circle(eyeBase.x, eyeBase.y, headR * 1.5);
        // draw pupils
        fill(pupilColor);
        noStroke();
        circle(pupil.x, pupil.y, headR * 0.8);
        break;
      case "abstract":
        noFill();
        stroke(baseColor);
        strokeWeight(2);
        // use a line to connect segments
        beginShape();
        this.segments.forEach((seg) => vertex(seg.x, seg.y));
        endShape();
        // draw head
        circle(this.segments[0].x, this.segments[0].y, headRadius * 2);
        break;
      case "devmode": // no eyes and pupils
        for (let i = 0; i < n; i++) {
          const seg = this.segments[i];
          const t = n > 1 ? i / (n - 1) : 0; // percentage of segment length
          // calculate segment properties
          const r = lerp(headRadius, tailRadius, t); // smooth transition between head and tail radius
          const alpha = lerp(255, 120, t);
          // calculate color
          const c = color(
            red(baseColor),
            green(baseColor),
            blue(baseColor),
            alpha
          );
          fill(c);
          circle(seg.x, seg.y, r * 2);
        }
        break;
    }

    pop();
  }
}

// sensation and perception of the snake
class SnakeSense {
  constructor(scene, body, config) {
    this.scene = scene;
    this.body = body;
    this.config = config;

    this.lastHeardPos = null;
    this.lastHeardStrength = 0; // how loud the last sound was
    this.lastSeenPos = null;
    this.seeTimer = 0;
  }

  update(player) {
    const head = this.body.head;
    const pos = player.body.pos;
    this.updateHearing(head, pos);
    this.updateVision(head, pos);
  }

  updateHearing(head, player) {
    const dx = player.x - head.x;
    const dy = player.y - head.y;
    const d = dist(head.x, head.y, player.x, player.y);

    if (d > this.config.hearingRange) return; // out of hearing range

    let hearingStrength = 1 - constrain(d / this.config.hearingRange, 0, 1);
    // trust in sound based on confidence bias
    hearingStrength = pow(
      hearingStrength,
      1.0 / this.config.hearingConfidenceBias
    );

    const baseAngle = atan2(dy, dx);
    const maxAngleNoise = map(
      1 - hearingStrength,
      0,
      1,
      radians(5),
      radians(60)
    );
    const angleNoise = random(-maxAngleNoise, maxAngleNoise);

    const estAngle = baseAngle + angleNoise;
    const estDist = d * random(0.8, 1.2); // add some distance noise

    this.lastHeardPos = {
      x: head.x + estDist * cos(estAngle),
      y: head.y + estDist * sin(estAngle),
    };
    this.lastHeardStrength = hearingStrength;
  }

  updateVision(head, player) {
    const dx = player.x - head.x;
    const dy = player.y - head.y;
    const d = dist(head.x, head.y, player.x, player.y);

    if (d > this.config.visionRange) {
      this.seeTimer += deltaTime;
      return;
    } // out of vision range

    const dir = this.body.getFacingDir()
      ? this.body.getFacingDir()
      : createVector(1, 0);
    const angleToPlayer = atan2(dy, dx);
    const facingAngle = atan2(dir.y, dir.x);
    const angleDiff = angleNormalize(angleToPlayer - facingAngle);

    if (abs(angleDiff) > this.config.visionFov / 2) {
      this.seeTimer += deltaTime;
      return; // out of FOV
    }

    // line of sight check
    if (!this.scene.hasLineOfSight(head, player)) {
      this.seeTimer += deltaTime;
      return; // blocked view
    }

    this.lastSeenPos = { x: player.x, y: player.y };
    this.seeTimer = 0; // reset see timer
  }
}

// decision making and cognition of the snake: AI
class SnakeMind {
  constructor(body, sense, config) {
    this.body = body;
    this.sense = sense;
    this.config = config;

    this.state = "IDLE"; // idle, alert, chasing, attacking
    this.stateTime = 0;
    this.chaseTime = 0;
    this.attackCooldownTimer = 0;

    this.currentTarget = null;
  }

  getTarget() {
    return this.currentTarget;
  }

  update(player) {
    const dt = deltaTime;
    this.stateTime += dt;
    this.attackCooldownTimer -= dt;

    switch (this.state) {
      // idle
      case "IDLE":
        this.updateIdle(player);
        break;
      // randomly patrol on the ground
      case "PATROL":
        this.updatePatrol(player);
        break;
      // alerted by sound or sight
      case "SEARCH":
        this.updateSearch(player);
        break;
      // chasing the player
      case "CHASE":
        this.updateChase(player);
        break;
      // missing target
      case "LOST":
        this.updateLost(player);
        break;
    }
  }

  updateIdle() {
    this.body.setSpeedMultiplier(0.2);

    if (!this.currentTarget || this._nearTarget(this.currentTarget)) {
      this.currentTarget = this._pickPatrolPoint();
    }

    // Transition to PATROL after some time
    if (this.stateTime > 2000) {
      this.transitionTo("PATROL");
    }

    this.checkForPlayer();
  }

  updatePatrol() {
    this.body.setSpeedMultiplier(0.5);

    // Change direction randomly
    if (!this.currentTarget || this._nearTarget(this.currentTarget)) {
      this.currentTarget = this._pickPatrolPoint();
    }

    this.checkForPlayer();
  }

  updateSearch(player) {
    this.body.setSpeedMultiplier(0.8);

    if (this.sense.lastSeenPos) {
      this.currentTarget = this.sense.lastSeenPos;
    } else if (this.sense.lastHeardPos) {
      this.currentTarget = this.sense.lastHeardPos;
    } else {
      // If no target, transition to PATROL
      this.transitionTo("PATROL");
      return;
    }

    // If lost interest, transition to LOST
    if (
      this._nearTarget(this.currentTarget) ||
      this.sense.seeTimer > this.config.loseInterestTime
    ) {
      this.transitionTo("LOST");
    }

    // If sees player clearly, chase
    if (this.sense.lastSeenPos && this.sense.seeTimer < 100) {
      this.transitionTo("CHASE");
    }
  }

  updateChase(player) {
    this.body.setSpeedMultiplier(1.2);
    this.chaseTime += deltaTime;

    if (this.sense.lastSeenPos) {
      this.currentTarget = this.sense.lastSeenPos;
    }

    // Attack logic
    const head = this.body.head;
    const pos = player.body.pos;
    const d = dist(head.x, head.y, pos.x, pos.y);
    if (d < this.config.attackRange && this.attackCooldownTimer <= 0) {
      console.log("ATTACK!");
      this.attackCooldownTimer = this.config.attackCooldown;
      // Implement attack damage here later
    }

    // Give up if lost sight for too long
    if (this.sense.seeTimer > this.config.loseInterestTime) {
      this.transitionTo("LOST");
    }

    // Give up if chased too long
    if (this.chaseTime > this.config.maxChaseTime) {
      this.transitionTo("IDLE");
    }
  }

  updateLost(player) {
    this.body.setSpeedMultiplier(0);

    // Look around (wait)
    if (this.stateTime > 2000) {
      this.transitionTo("PATROL");
    }

    this.checkForPlayer();
  }

  // --- Helpers ---
  transitionTo(newState) {
    this.state = newState;
    this.stateTime = 0;
    if (newState === "CHASE") this.chaseTime = 0;
  }

  checkForPlayer() {
    // Priority: Vision > Hearing
    if (this.sense.lastSeenPos && this.sense.seeTimer < 500) {
      this.transitionTo("CHASE");
      this.currentTarget = this.sense.lastSeenPos;
      return;
    } else if (this.sense.lastHeardPos && this.sense.lastHeardStrength > 0.5) {
      this.transitionTo("SEARCH");
      this.currentTarget = this.sense.lastHeardPos;
    }
  }

  _nearTarget(target) {
    if (!target) return true;
    const head = this.body.head;
    const d = dist(head.x, head.y, target.x, target.y);
    return d < TILE_SIZE * 0.8;
  }

  _pickPatrolPoint() {
    const head = this.body.head;
    const rangeTiles = 6;
    const baseTile = worldToTile(head.x, head.y);

    for (let i = 0; i < 16; i++) {
      const col = baseTile.col + floor(random(-rangeTiles, rangeTiles));
      const row = baseTile.row + floor(random(-rangeTiles, rangeTiles));

      if (col < 0 || col >= WORLD_COLS || row < 0 || row >= WORLD_ROWS)
        continue;
      if (!this.body.nav.isWalkable(col, row)) continue;

      const w = tileToWorld(col, row);
      return { x: w.x, y: w.y };
    }
    return null;
  }
}
