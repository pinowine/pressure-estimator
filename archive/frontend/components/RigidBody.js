const EPS = 0.1;

class RigidBody {
  constructor({
    x, y, radius = 10,
    gravity = 0.4, // base gravity
    riseGravityScale = 1.2, // rise gravity multiplier
    fallGravityScale = 2.0, // fall gravity multiplier
    lowJumpMultiplier = 2.0, // if jump key released early
    maxFallSpeed = 20, // terminal velocity
    groundAccel = 0.5, // horizontal acceleration
    airAccel = 0.35, // horizontal acceleration in air
    maxHorizontalSpeed = 5, // max horizontal speed
    friction = 0.8, // ground friction
    coyoteTimeMs = 200 // coyote time duration in ms
  }) {
    // position and velocity vectors
    this.pos = createVector(x, y);
    this.vel = createVector(0, 0);
    this.radius = radius;

    // gravity and physics parameters
    this.gravity = gravity;
    this.riseGravityScale = riseGravityScale;
    this.fallGravityScale = fallGravityScale;
    this.lowJumpMultiplier = lowJumpMultiplier;
    this.maxFallSpeed = maxFallSpeed;
    this.groundAccel = groundAccel;
    this.airAccel = airAccel;
    this.maxHorizontalSpeed = maxHorizontalSpeed;
    this.friction = friction;

    // state flags
    this.onGround = false;
    this.onWall = false;
    this.isRising = false;

    // timer
    this.lastOnGroundTime = -Infinity;
    this.coyoteTimeMs = coyoteTimeMs;
  }

  update(scene, input) {
    this.applyAcceleration(input);
    this.resolveHorizontalCollision(scene);

    this.applyGravity(input);
    this.resolveVerticalCollision(scene);

    this.resolveEdgeCollision(WORLD_WIDTH, WORLD_HEIGHT);
  }

  applyAcceleration(input) {
    const direction = input ? input.moveDir : 0;
    const accel = this.onGround ? this.groundAccel : this.airAccel;
    // console.log("Applying acceleration:", accel, "Direction:", direction);

    // horizontal movement
    if (direction !== 0) {
      this.vel.x += accel * direction;
    } else {
      this.vel.x *= this.friction;
      if (abs(this.vel.x) < 0.01) this.vel.x = 0; // stop completely if very slow
    }

    // cap horizontal speed
    this.vel.x = constrain(this.vel.x, -this.maxHorizontalSpeed, this.maxHorizontalSpeed);

    this.pos.x += this.vel.x;
  }

  applyGravity(input) {
    // stand still on ground
    if (this.onGround && this.vel.y >= 0) {
      this.vel.y = 0;
      return;
    }
    // is falling or rising
    this.isRising = this.vel.y < 0;
    let g = this.gravity * (this.isRising ? this.riseGravityScale : this.fallGravityScale);

    // low jump adjustment: additional gravity when jump released early
    if (this.isRising && input && !input.jumpHeld) {
      g *= this.lowJumpMultiplier;
    }

    this.vel.y += g;

    // cap fall speed
    if (this.vel.y > this.maxFallSpeed) {
      this.vel.y = this.maxFallSpeed;
    }

    this.pos.y += this.vel.y;
  }

  // collision with the platform tiles horizontally
  resolveHorizontalCollision(scene) {

    const r = this.radius;
    this.onWall = false;

    if (this.vel.x === 0) return;

    const movingRight = this.vel.x > 0;
    const sideX = this.pos.x + (movingRight ? r : -r); // right or left side
    const checkX = sideX + (movingRight ? EPS : -EPS);

    const topY = this.pos.y - r * 0.3;
    const bottomY = this.pos.y + r * 0.3;

    const hitTop = scene.isSolidAt(checkX, topY);
    const hitBottom = scene.isSolidAt(checkX, bottomY);

    if (hitTop || hitBottom) {
      const col = floor(checkX / TILE_SIZE);
      const tileX = movingRight
        ? col * TILE_SIZE  // left side of the right tile
        : (col + 1) * TILE_SIZE;  // right side of the left tile
      this.vel.x = 0; // stop horizontal velocity
      this.pos.x = tileX + (movingRight ? -r : r); // align to side of tile
      this.onWall = true; // mark as on wall
    }
  }

  // collision with tiles above and below
  resolveVerticalCollision(scene) {
    const r = this.radius;
    this.onGround = false;

    // check ground when v.y > 0
    if (this.vel.y >= 0) {
      const footY = this.pos.y + r;
      const checkY = footY + EPS;

      const leftX = this.pos.x - r * 0.4;
      const rightX = this.pos.x + r * 0.4;

      const groundedLeft = scene.isSolidAt(leftX, checkY);
      const groundedRight = scene.isSolidAt(rightX, checkY);

      if (groundedLeft || groundedRight) {
        this.vel.y = 0; // stop downward velocity
        this.pos.y = floor(footY / TILE_SIZE) * TILE_SIZE - r; // align to top of tile
        this.onGround = true; // mark as on ground
      }
    }

    // check ceiling when v.y < 0
    if (this.vel.y < 0) {
      const headY = this.pos.y - r;
      const checkY = headY - EPS;

      const leftX = this.pos.x - r * 0.4;
      const rightX = this.pos.x + r * 0.4;

      const hitTopLeft = scene.isSolidAt(leftX, checkY);
      const hitTopRight = scene.isSolidAt(rightX, checkY);

      if (hitTopLeft || hitTopRight) {
        this.vel.y = 0; // stop upward velocity
        // this.pos.y = (floor(headY / TILE_SIZE) + 1) * TILE_SIZE + r; // align to bottom of tile
      }
    }
  }

  // prevent going out of canvas bounds
  resolveEdgeCollision(canvasWidth, canvasHeight) {
    const r = this.radius;
    // left
    if (this.pos.x - r < 0) {
      this.pos.x = r;
      if (this.vel.x < 0) this.vel.x = 0;
    }
    // right
    if (this.pos.x + r > canvasWidth) {
      this.pos.x = canvasWidth - r;
      if (this.vel.x > 0) this.vel.x = 0;
    }
    // top
    if (this.pos.y - r < 0) {
      this.pos.y = r;
      if (this.vel.y < 0) this.vel.y = 0;
    }
    // bottom
    if (this.pos.y + r > canvasHeight) {
      this.pos.y = canvasHeight - r;
      if (this.vel.y > 0) this.vel.y = 0;
    }
  }

  jump(power = 10) {
    const now = millis();
    const canUseCoyote = (now - this.lastOnGroundTime) <= this.coyoteTimeMs;

    if (this.onGround || canUseCoyote) {
      this.vel.y = -power;
      this.onGround = false;
      this.lastOnGroundTime = -Infinity; // reset coyote timer
    }
  }

  getHitbox() {
    return {
      x: this.pos.x,
      y: this.pos.y,
      r: this.radius
    };
  }
}
