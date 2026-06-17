class AStarPathfinder {
  constructor(nav) {
    this.nav = nav; // SnakeNav
  }

  findPath(startCol, startRow, goalCol, goalRow) {
    // if start or goal is not walkable, return null
    if (!this.nav.isWalkable(goalCol, goalRow)) return null;
    if (!this.nav.isWalkable(startCol, startRow)) return null;

    const startKey = `${startCol},${startRow}`;
    const goalKey = `${goalCol},${goalRow}`;

    const open = new Map(); // key -> node
    const closed = new Set(); // key

    function heuristic(c1, r1, c2, r2) {
      // manhattan distance
      return abs(c1 - c2) + abs(r1 - r2);
    }

    const startNode = {
      col: startCol,
      row: startRow,
      g: 0,
      h: heuristic(startCol, startRow, goalCol, goalRow),
      f: 0,
      parent: null,
    };
    startNode.f = startNode.g + startNode.h;

    open.set(startKey, startNode);

    const neighborOffsets = [
      { dc: 1, dr: 0 },
      { dc: -1, dr: 0 },
      { dc: 0, dr: 1 },
      { dc: 0, dr: -1 },
    ];

    while (open.size > 0) {
      // 1. find the node with the smallest f
      let currentKey = null;
      let currentNode = null;
      for (let [k, node] of open.entries()) {
        if (!currentNode || node.f < currentNode.f) {
          currentNode = node;
          currentKey = k;
        }
      }

      if (!currentNode) break;

      // end condition: reached goal
      if (currentNode.col === goalCol && currentNode.row === goalRow) {
        return this.reconstructPath(currentNode);
      }

      open.delete(currentKey);
      closed.add(currentKey);

      // 2. expand neighbors
      for (let { dc, dr } of neighborOffsets) {
        const nc = currentNode.col + dc;
        const nr = currentNode.row + dr;
        const nKey = `${nc},${nr}`;

        if (closed.has(nKey)) continue;
        if (!this.nav.isWalkable(nc, nr)) continue;

        const tentativeG = currentNode.g + 1; // 4-directional movement cost = 1

        let neighbor = open.get(nKey);
        if (!neighbor) {
          neighbor = {
            col: nc,
            row: nr,
            g: tentativeG,
            h: heuristic(nc, nr, goalCol, goalRow),
            f: 0,
            parent: currentNode,
          };
          neighbor.f = neighbor.g + neighbor.h;
          open.set(nKey, neighbor);
        } else if (tentativeG < neighbor.g) {
          neighbor.g = tentativeG;
          neighbor.parent = currentNode;
          neighbor.f = neighbor.g + neighbor.h;
        }
      }
    }

    // no path found
    return null;
  }

  reconstructPath(node) {
    const path = [];
    let cur = node;
    while (cur) {
      path.push({ col: cur.col, row: cur.row });
      cur = cur.parent;
    }
    path.reverse();
    return path;
  }
}
