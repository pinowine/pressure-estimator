function keyReleased() {
  if (key === "P" || key === "p") {
    loadScene(floor(random(0, length)));
    console.log("Scene changed.");
  }
}
