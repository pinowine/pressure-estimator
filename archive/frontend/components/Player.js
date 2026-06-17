class Player {
  constructor(x, y) {
    this.body = new RigidBody({ x, y, radius: 13 });
    this.jumpStrength = 10;
    this.jumpWasHeld = false;
    this.eyePos = createVector(0, 0);
  }

  jump() {
    this.body.jump(this.jumpStrength);
  }

  update(scene) {
    const left = keyIsDown(65) || keyIsDown(LEFT_ARROW); // A key
    const right = keyIsDown(68) || keyIsDown(RIGHT_ARROW); // D key
    const moveDir = (left ? -1 : 0) + (right ? 1 : 0); // -1 = left, 0 = no move, 1 = right

    const jumpHeld = keyIsDown(32); // space
    const input = { moveDir, jumpHeld };

    this.body.update(scene, input);

    this.jumpWasHeld = jumpHeld;

    const v = this.body.vel.copy();
    const speed = v.mag();
    if (speed < 0.01) {
      this.eyePos.set(0, 0);
      return;
    }

    v.normalize();
    const maxOffset = this.body.radius * 0.7;
    const t = constrain(speed / this.body.maxHorizontalSpeed, 0, 1);
    const dEye = t * maxOffset;

    this.eyePos = v.mult(dEye);
  }

  draw(mode = "default") {
    push();
    translate(this.body.pos.x, this.body.pos.y);

    switch (mode) {
      case "default":
        fill(255);
        noStroke();
        ellipse(0, 0, this.body.radius * 2);
        stroke(0);
        ellipse(this.eyePos.x, this.eyePos.y, this.body.radius * 0.9);
        fill(0);
        ellipse(this.eyePos.x, this.eyePos.y, this.body.radius * 0.45);
        break;
      case "abstract":
        stroke(0);
        noFill();
        strokeWeight(2);
        ellipse(0, 0, this.body.radius * 2);
        break;
      case "devmode":
        // default drawing
        fill(0);
        noStroke();
        ellipse(0, 0, this.body.radius * 2);
        stroke(255);
        ellipse(this.eyePos.x, this.eyePos.y, this.body.radius * 0.9);
        fill(255);
        ellipse(this.eyePos.x, this.eyePos.y, this.body.radius * 0.45);
        // debug velocity line
        stroke(0, 255, 255);
        const debugVel = this.body.vel.copy().mult(5);
        line(0, 0, debugVel.x, debugVel.y);
        break;
    }

    pop();
  }
}

function keyPressed() {
  if (key === " " && player) {
    player.jump();
  }
}
