const state = {
  size: 3,
  algorithm: "linear_conflict",
  boards: {
    start: [1, 2, 3, 4, 5, 6, 7, 8, 0],
    goal: [1, 2, 3, 4, 5, 6, 7, 8, 0]
  },
  validation: {
    start: [],
    goal: [],
    startValid: true,
    goalValid: true
  },
  dragSession: {
    boardKey: null,
    sourceIndex: -1,
    hoverIndex: -1,
    pending: false,
    dragging: false,
    startX: 0,
    startY: 0,
    pointerId: null,
    ghostEl: null,
    trailEl: null,
    trailPoints: []
  },
  solving: false,
  solution: null,
  playback: {
    step: 0,
    playing: false,
    timer: null,
    intervalMs: 650
  }
}

const ui = {
  sizeSelect: document.getElementById("sizeSelect"),
  algorithmSelect: document.getElementById("algorithmSelect"),
  algorithmRow: document.getElementById("algorithmRow"),
  randomBothBtn: document.getElementById("randomBothBtn"),
  randomStartBtn: document.getElementById("randomStartBtn"),
  randomGoalBtn: document.getElementById("randomGoalBtn"),
  startBoard: document.getElementById("startBoard"),
  goalBoard: document.getElementById("goalBoard"),
  goalTip: document.getElementById("goalTip"),
  solveBtn: document.getElementById("solveBtn"),
  solveStatus: document.getElementById("solveStatus"),
  globalValidation: document.getElementById("globalValidation"),
  solveHint: document.getElementById("solveHint"),
  playbackSection: document.getElementById("playbackSection"),
  playbackBoard: document.getElementById("playbackBoard"),
  stepSummary: document.getElementById("stepSummary"),
  moveInfo: document.getElementById("moveInfo"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  stepRange: document.getElementById("stepRange")
}

const solveApi = {
  endpoint: "http://localhost:5000/api/puzzle/solve",
  async solve(payload) {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    })
    if (!response.ok) {
      throw new Error(`求解接口返回异常：HTTP ${response.status}`)
    }
    return response.json()
  }
}

function buildSolvedBoard(size) {
  const total = size * size
  const board = []
  for (let i = 1; i < total; i += 1) {
    board.push(i)
  }
  board.push(0)
  return board
}

function resizeBoards(size) {
  const solved = buildSolvedBoard(size)
  state.boards.start = solved.slice()
  state.boards.goal = solved.slice()
  state.solution = null
  stopPlayback()
  state.playback.step = 0
}

function getRangeMax() {
  return state.size * state.size - 1
}

function normalizeCellInput(raw) {
  if (raw === "") {
    return null
  }
  const value = Number(raw)
  if (!Number.isInteger(value)) {
    return NaN
  }
  return value
}

function validateBoard(board) {
  const max = getRangeMax()
  const flags = Array(board.length).fill(false)
  const seen = new Map()
  let valid = true
  for (let i = 0; i < board.length; i += 1) {
    const value = board[i]
    const outOfRange = !Number.isInteger(value) || value < 0 || value > max
    if (outOfRange) {
      flags[i] = true
      valid = false
      continue
    }
    const list = seen.get(value) || []
    list.push(i)
    seen.set(value, list)
  }
  for (const entry of seen.entries()) {
    const indexes = entry[1]
    if (indexes.length > 1) {
      valid = false
      for (const idx of indexes) {
        flags[idx] = true
      }
    }
  }
  if (!seen.has(0)) {
    valid = false
  }
  return { flags, valid }
}

function inversionCount(board) {
  // 可行性判定基础：统计逆序数（忽略空白格 0）
  const values = board.filter((item) => item !== 0)
  let count = 0
  for (let i = 0; i < values.length; i += 1) {
    for (let j = i + 1; j < values.length; j += 1) {
      if (values[i] > values[j]) {
        count += 1
      }
    }
  }
  return count
}

function boardParity(board, size) {
  // 可行性判定核心：计算状态奇偶性
  const inversions = inversionCount(board)
  if (size % 2 === 1) {
    // 3x3 等奇数阶：逆序数奇偶性即为状态奇偶性
    return inversions % 2
  }
  // 4x4 等偶数阶：逆序数 + 空白格自底向上的行号 共同决定奇偶性
  const blankIndex = board.indexOf(0)
  const blankRowFromBottom = size - Math.floor(blankIndex / size)
  return (inversions + blankRowFromBottom) % 2
}

function isSolvable(start, goal, size) {
  // 可行性判定结论：起点与终点奇偶性一致，则可达
  return boardParity(start, size) === boardParity(goal, size)
}

function randomPermutation(size) {
  const values = []
  const total = size * size
  for (let i = 0; i < total; i += 1) {
    values.push(i)
  }
  for (let i = values.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1))
    const tmp = values[i]
    values[i] = values[j]
    values[j] = tmp
  }
  return values
}

function generateRandomSolvableBoard(relativeBoard) {
  let candidate = randomPermutation(state.size)
  let guard = 0
  while (!isSolvable(candidate, relativeBoard, state.size) && guard < 2000) {
    candidate = randomPermutation(state.size)
    guard += 1
  }
  return candidate
}

function setCellValue(boardKey, index, rawValue) {
  const value = normalizeCellInput(rawValue)
  state.boards[boardKey][index] = value === null ? NaN : value
  state.solution = null
  stopPlayback()
  renderAll()
}

function renderBoard(boardKey, container, readonly) {
  const board = state.boards[boardKey]
  const validation = state.validation[boardKey]
  container.innerHTML = ""
  container.style.setProperty("--board-size", String(state.size))
  for (let i = 0; i < board.length; i += 1) {
    const value = board[i]
    const wrap = document.createElement("div")
    wrap.className = "cell-wrap"
    const cell = document.createElement("div")
    const blank = value === 0
    cell.className = `cell ${blank ? "blank" : "draggable"}`
    cell.dataset.index = String(i)

    // 始终使用input元素以保持样式一致
    const input = document.createElement("input")
    input.type = "text"
    input.inputMode = "numeric"
    input.pattern = "[0-9]*"
    input.maxLength = 2
    input.value = Number.isInteger(value) ? String(value) : ""
    input.setAttribute("aria-label", `${boardKey === "start" ? "初始" : "目标"}状态 第${i + 1}格`)

    if (readonly) {
      // 只读模式：禁用输入但不改变样式
      input.disabled = true
      input.classList.add("readonly")
    } else {
      // 可编辑模式：添加输入事件监听
      input.addEventListener("input", (event) => {
        const raw = event.target.value.replace(/[^\d]/g, "")
        event.target.value = raw
        setCellValue(boardKey, i, raw)
      })
    }

    cell.appendChild(input)
    const error = document.createElement("div")
    error.className = "cell-error"
    error.textContent = validation[i] ? "数字重复或越界" : ""
    wrap.appendChild(cell)
    wrap.appendChild(error)
    container.appendChild(wrap)
  }
}

function getMoveInfo(prev, curr, size) {
  const prevBlank = prev.indexOf(0)
  const movedTileValue = curr[prevBlank]
  if (!Number.isInteger(movedTileValue) || movedTileValue === 0) {
    return null
  }
  const fromIndex = prev.indexOf(movedTileValue)
  const toIndex = curr.indexOf(movedTileValue)
  const fromRow = Math.floor(fromIndex / size)
  const fromCol = fromIndex % size
  const toRow = Math.floor(toIndex / size)
  const toCol = toIndex % size
  let arrow = ""
  let direction = ""
  if (toCol > fromCol) {
    arrow = "→"
    direction = "右"
  } else if (toCol < fromCol) {
    arrow = "←"
    direction = "左"
  } else if (toRow > fromRow) {
    arrow = "↓"
    direction = "下"
  } else if (toRow < fromRow) {
    arrow = "↑"
    direction = "上"
  }
  return {
    tile: movedTileValue,
    toIndex,
    arrow,
    direction
  }
}

function renderPlaybackBoard() {
  if (!state.solution || !Array.isArray(state.solution.states) || state.solution.states.length === 0) {
    ui.playbackSection.classList.add("hidden")
    return
  }
  ui.playbackSection.classList.remove("hidden")
  ui.playbackBoard.style.setProperty("--board-size", String(state.size))
  ui.playbackBoard.innerHTML = ""
  const current = state.solution.states[state.playback.step]
  const prev = state.playback.step > 0 ? state.solution.states[state.playback.step - 1] : null
  const move = prev ? getMoveInfo(prev, current, state.size) : null
  for (let i = 0; i < current.length; i += 1) {
    const wrap = document.createElement("div")
    wrap.className = "cell-wrap"
    const cell = document.createElement("div")
    const value = current[i]
    const blank = value === 0
    let cls = `cell static ${blank ? "blank" : ""}`
    if (move && i === move.toIndex) {
      cls += " moved"
    }
    cell.className = cls
    cell.textContent = blank ? "" : String(value)
    if (move && i === move.toIndex) {
      const arrow = document.createElement("span")
      arrow.className = "arrow"
      arrow.textContent = move.arrow
      cell.appendChild(arrow)
    }
    const error = document.createElement("div")
    error.className = "cell-error"
    wrap.appendChild(cell)
    wrap.appendChild(error)
    ui.playbackBoard.appendChild(wrap)
  }
  const totalSteps = state.solution.states.length - 1
  ui.stepSummary.textContent = `总步数 ${totalSteps} ｜ 当前 ${state.playback.step}`
  ui.stepRange.max = String(totalSteps)
  ui.stepRange.value = String(state.playback.step)
  if (move) {
    ui.moveInfo.textContent = `第 ${state.playback.step} 步：移动数字 ${move.tile} 向${move.direction} ${move.arrow}`
    ui.moveInfo.className = "status ok"
  } else {
    ui.moveInfo.textContent = "初始位置"
    ui.moveInfo.className = "status ok"
  }
}

function normalizeSolution(raw) {
  const states = Array.isArray(raw?.states) ? raw.states : Array.isArray(raw?.path) ? raw.path : null
  if (!states || states.length === 0) {
    throw new Error("接口响应缺少 states/path 轨迹数据")
  }
  const total = state.size * state.size
  const normalizedStates = states.map((item, idx) => {
    if (!Array.isArray(item) || item.length !== total) {
      throw new Error(`第 ${idx + 1} 个状态长度不合法`)
    }
    return item.map((cell) => Number(cell))
  })
  return {
    states: normalizedStates
  }
}

function updateValidation() {
  const startCheck = validateBoard(state.boards.start)
  const goalCheck = validateBoard(state.boards.goal)
  state.validation.start = startCheck.flags
  state.validation.goal = goalCheck.flags
  state.validation.startValid = startCheck.valid
  // 目标状态校验
  state.validation.goalValid = goalCheck.valid
  const validInput = startCheck.valid && goalCheck.valid
  // “初始状态能否到达目标状态”的可行性校验入口
  const parityOK = validInput && isSolvable(state.boards.start, state.boards.goal, state.size)
  let globalText = ""
  let globalClass = "status ok"
  if (!validInput) {
    globalText = "存在非法输入：请确保每个网格均为 0~最大值且无重复，且包含 0。"
    globalClass = "status error"
  } else if (!parityOK) {
    globalText = "当前初始状态与目标状态在该网格规模下不可达。"
    globalClass = "status error"
  } else {
    globalText = "输入合法且可解，可以开始求解。"
    globalClass = "status ok"
  }
  ui.globalValidation.textContent = globalText
  ui.globalValidation.className = globalClass
  ui.solveBtn.disabled = !parityOK || state.solving
  ui.solveHint.textContent = `数值范围：0 ~ ${getRangeMax()}，其中 0 表示空白格。`
}

function renderAll() {
  updateValidation()
  renderBoard("start", ui.startBoard, false)

  // 模式数据库算法要求目标必须是标准目标，设为只读并更新提示
  const goalReadonly = state.algorithm === "pattern_db"
  if (goalReadonly) {
    const maxNum = state.size * state.size - 1
    ui.goalTip.textContent = `模式数据库算法仅支持标准目标状态（1到${maxNum}顺序排列，0在最后），目标已锁定。`
  } else {
    ui.goalTip.textContent = `请正确输入数字 0–${state.size * state.size - 1} 不重复、空格用 0 表示。`
  }

  renderBoard("goal", ui.goalBoard, goalReadonly)
  renderPlaybackBoard()
}

function swapAt(boardKey, indexA, indexB) {
  const next = state.boards[boardKey].slice()
  const temp = next[indexA]
  next[indexA] = next[indexB]
  next[indexB] = temp
  state.boards[boardKey] = next
  state.solution = null
  stopPlayback()
  renderAll()
}

function clearDropTarget() {
  const prev = document.querySelector(".cell.drop-target")
  if (prev) {
    prev.classList.remove("drop-target")
  }
}

function setDropTarget(cell) {
  clearDropTarget()
  if (cell) {
    cell.classList.add("drop-target")
  }
}

function clearGhost() {
  if (state.dragSession.ghostEl) {
    state.dragSession.ghostEl.remove()
    state.dragSession.ghostEl = null
  }
}

function clearTrail() {
  if (state.dragSession.trailEl) {
    state.dragSession.trailEl.remove()
    state.dragSession.trailEl = null
  }
  state.dragSession.trailPoints = []
}

function renderTrailPath() {
  if (!state.dragSession.trailEl) {
    return
  }
  const polyline = state.dragSession.trailEl.querySelector("polyline")
  if (!polyline) {
    return
  }
  const pointsText = state.dragSession.trailPoints.map((point) => `${point.x},${point.y}`).join(" ")
  polyline.setAttribute("points", pointsText)
}

function appendTrailPoint(clientX, clientY) {
  const points = state.dragSession.trailPoints
  const last = points.length > 0 ? points[points.length - 1] : null
  if (last) {
    const dx = clientX - last.x
    const dy = clientY - last.y
    if (Math.abs(dx) + Math.abs(dy) < 10) {
      return
    }
  }
  points.push({ x: clientX, y: clientY })
  renderTrailPath()
}

function createTrail(clientX, clientY) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
  svg.setAttribute("class", "drag-trail")
  svg.setAttribute("viewBox", `0 0 ${window.innerWidth} ${window.innerHeight}`)
  svg.setAttribute("width", String(window.innerWidth))
  svg.setAttribute("height", String(window.innerHeight))
  const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline")
  polyline.setAttribute("class", "drag-trail-line")
  svg.appendChild(polyline)
  document.body.appendChild(svg)
  state.dragSession.trailEl = svg
  state.dragSession.trailPoints = []
  appendTrailPoint(clientX, clientY)
}

function updateGhostPosition(clientX, clientY) {
  if (!state.dragSession.ghostEl) {
    return
  }
  state.dragSession.ghostEl.style.left = `${clientX}px`
  state.dragSession.ghostEl.style.top = `${clientY}px`
}

function createGhost(boardKey, index, clientX, clientY) {
  const value = state.boards[boardKey][index]
  const ghost = document.createElement("div")
  ghost.className = "drag-ghost"
  ghost.textContent = String(value)
  document.body.appendChild(ghost)
  state.dragSession.ghostEl = ghost
  updateGhostPosition(clientX, clientY)
}

function sourceCellElement() {
  if (!state.dragSession.boardKey || state.dragSession.sourceIndex < 0) {
    return null
  }
  const boardContainer = state.dragSession.boardKey === "start" ? ui.startBoard : ui.goalBoard
  return boardContainer.querySelector(`.cell[data-index="${state.dragSession.sourceIndex}"]`)
}

function dragTargetAtPoint(clientX, clientY) {
  const hit = document.elementFromPoint(clientX, clientY)
  if (!hit) {
    return null
  }
  const cell = hit.closest(".cell")
  if (!cell || !state.dragSession.boardKey) {
    return null
  }
  const boardContainer = state.dragSession.boardKey === "start" ? ui.startBoard : ui.goalBoard
  if (!boardContainer.contains(cell)) {
    return null
  }
  return cell
}

function beginDrag(boardKey, sourceIndex, clientX, clientY) {
  const sourceCell = sourceCellElement()
  if (!sourceCell) {
    return
  }
  state.dragSession.dragging = true
  sourceCell.classList.add("dragging-source")
  document.body.classList.add("dragging-active")
  createGhost(boardKey, sourceIndex, clientX, clientY)
  createTrail(clientX, clientY)
}

function startDragSession(boardKey, event) {
  if (event.button !== 0) {
    return
  }
  const cell = event.target.closest(".cell")
  if (!cell) {
    return
  }
  const sourceIndex = Number(cell.dataset.index)
  if (!Number.isInteger(sourceIndex)) {
    return
  }
  const value = state.boards[boardKey][sourceIndex]
  if (!Number.isInteger(value) || value === 0) {
    return
  }
  state.dragSession.boardKey = boardKey
  state.dragSession.sourceIndex = sourceIndex
  state.dragSession.hoverIndex = -1
  state.dragSession.pending = true
  state.dragSession.dragging = false
  state.dragSession.startX = event.clientX
  state.dragSession.startY = event.clientY
  state.dragSession.pointerId = event.pointerId
}

function endDragSession() {
  const sourceCell = sourceCellElement()
  if (sourceCell) {
    sourceCell.classList.remove("dragging-source")
  }
  clearDropTarget()
  clearGhost()
  clearTrail()
  document.body.classList.remove("dragging-active")
  state.dragSession.boardKey = null
  state.dragSession.sourceIndex = -1
  state.dragSession.hoverIndex = -1
  state.dragSession.pending = false
  state.dragSession.dragging = false
  state.dragSession.startX = 0
  state.dragSession.startY = 0
  state.dragSession.pointerId = null
}

function moveDragSession(event) {
  if (!state.dragSession.pending || !state.dragSession.boardKey) {
    return
  }
  if (state.dragSession.pointerId !== null && event.pointerId !== state.dragSession.pointerId) {
    return
  }
  const dx = event.clientX - state.dragSession.startX
  const dy = event.clientY - state.dragSession.startY
  if (!state.dragSession.dragging) {
    if (Math.abs(dx) < 6 && Math.abs(dy) < 6) {
      return
    }
    beginDrag(state.dragSession.boardKey, state.dragSession.sourceIndex, event.clientX, event.clientY)
  }
  updateGhostPosition(event.clientX, event.clientY)
  appendTrailPoint(event.clientX, event.clientY)
  const targetCell = dragTargetAtPoint(event.clientX, event.clientY)
  if (targetCell) {
    const targetIndex = Number(targetCell.dataset.index)
    state.dragSession.hoverIndex = Number.isInteger(targetIndex) ? targetIndex : -1
    if (state.dragSession.hoverIndex >= 0 && state.dragSession.hoverIndex !== state.dragSession.sourceIndex) {
      setDropTarget(targetCell)
    } else {
      clearDropTarget()
    }
  } else {
    state.dragSession.hoverIndex = -1
    clearDropTarget()
  }
}

function finishDragSession() {
  if (!state.dragSession.pending || !state.dragSession.boardKey) {
    return
  }
  const shouldSwap = state.dragSession.dragging && state.dragSession.hoverIndex >= 0 && state.dragSession.hoverIndex !== state.dragSession.sourceIndex
  const boardKey = state.dragSession.boardKey
  const sourceIndex = state.dragSession.sourceIndex
  const targetIndex = state.dragSession.hoverIndex
  endDragSession()
  if (shouldSwap) {
    swapAt(boardKey, sourceIndex, targetIndex)
  }
}

function stopPlayback() {
  if (state.playback.timer) {
    clearInterval(state.playback.timer)
    state.playback.timer = null
  }
  state.playback.playing = false
  ui.playPauseBtn.textContent = "播放"
}

function stepTo(index) {
  if (!state.solution) {
    return
  }
  const max = state.solution.states.length - 1
  state.playback.step = Math.max(0, Math.min(index, max))
  renderPlaybackBoard()
}

function nextStep() {
  if (!state.solution) {
    return
  }
  const max = state.solution.states.length - 1
  if (state.playback.step >= max) {
    stopPlayback()
    return
  }
  state.playback.step += 1
  renderPlaybackBoard()
}

function prevStep() {
  if (!state.solution) {
    return
  }
  state.playback.step = Math.max(0, state.playback.step - 1)
  renderPlaybackBoard()
}

function togglePlayPause() {
  if (!state.solution) {
    return
  }
  if (state.playback.playing) {
    stopPlayback()
    return
  }
  state.playback.playing = true
  ui.playPauseBtn.textContent = "暂停"
  state.playback.timer = setInterval(() => {
    const max = state.solution.states.length - 1
    if (state.playback.step >= max) {
      stopPlayback()
      return
    }
    nextStep()
  }, state.playback.intervalMs)
}

async function requestSolve() {
  if (ui.solveBtn.disabled) {
    return
  }
  state.solving = true
  ui.solveBtn.disabled = true
  ui.solveStatus.textContent = "正在请求后端求解..."
  ui.solveStatus.className = "status ok"
  try {
    const payload = {
      size: state.size,
      algorithm: state.algorithm,
      start: state.boards.start,
      goal: state.boards.goal
    }
    const raw = await solveApi.solve(payload)
    const normalized = normalizeSolution(raw)
    state.solution = normalized
    state.playback.step = 0
    stopPlayback()
    ui.solveStatus.textContent = "求解完成，已生成步骤播放。"
    ui.solveStatus.className = "status ok"
    renderPlaybackBoard()
  } catch (error) {
    state.solution = null
    ui.solveStatus.textContent = `求解失败：${error.message}`
    ui.solveStatus.className = "status error"
    renderPlaybackBoard()
  } finally {
    state.solving = false
    updateValidation()
  }
}

function bindEvents() {
  ui.sizeSelect.addEventListener("change", (event) => {
    state.size = Number(event.target.value)
    resizeBoards(state.size)
    if (state.size === 4) {
      ui.algorithmRow.style.display = "flex"
    } else {
      ui.algorithmRow.style.display = "none"
    }
    renderAll()
  })

  ui.algorithmSelect.addEventListener("change", (event) => {
    state.algorithm = event.target.value

    // 模式数据库算法要求目标必须是标准目标（123...0）
    if (state.algorithm === "pattern_db") {
      const standardGoal = buildSolvedBoard(state.size)
      state.boards.goal = standardGoal
      ui.randomGoalBtn.disabled = true
      ui.randomGoalBtn.classList.add("disabled")
    } else {
      ui.randomGoalBtn.disabled = false
      ui.randomGoalBtn.classList.remove("disabled")
    }

    renderAll()
  })

  ui.randomBothBtn.addEventListener("click", () => {
    state.boards.goal = randomPermutation(state.size)
    state.boards.start = generateRandomSolvableBoard(state.boards.goal)
    state.solution = null
    stopPlayback()
    renderAll()
  })

  ui.randomStartBtn.addEventListener("click", () => {
    state.boards.start = generateRandomSolvableBoard(state.boards.goal)
    state.solution = null
    stopPlayback()
    renderAll()
  })

  ui.randomGoalBtn.addEventListener("click", () => {
    state.boards.goal = generateRandomSolvableBoard(state.boards.start)
    state.solution = null
    stopPlayback()
    renderAll()
  })

  ui.startBoard.addEventListener("pointerdown", (event) => {
    startDragSession("start", event)
  })

  ui.goalBoard.addEventListener("pointerdown", (event) => {
    startDragSession("goal", event)
  })

  window.addEventListener("pointermove", (event) => {
    moveDragSession(event)
  })

  window.addEventListener("pointerup", (event) => {
    if (state.dragSession.pointerId !== null && event.pointerId !== state.dragSession.pointerId) {
      return
    }
    finishDragSession()
  })

  window.addEventListener("pointercancel", (event) => {
    if (state.dragSession.pointerId !== null && event.pointerId !== state.dragSession.pointerId) {
      return
    }
    finishDragSession()
  })

  ui.solveBtn.addEventListener("click", () => {
    requestSolve()
  })

  ui.prevBtn.addEventListener("click", () => {
    stopPlayback()
    prevStep()
  })

  ui.nextBtn.addEventListener("click", () => {
    stopPlayback()
    nextStep()
  })

  ui.playPauseBtn.addEventListener("click", () => {
    togglePlayPause()
  })

  ui.stepRange.addEventListener("input", (event) => {
    stopPlayback()
    stepTo(Number(event.target.value))
  })

  window.addEventListener("keydown", (event) => {
    const tag = event.target && event.target.tagName ? event.target.tagName.toLowerCase() : ""
    if (tag === "input" || tag === "textarea" || tag === "select") {
      return
    }
    if (event.code === "Space") {
      event.preventDefault()
      togglePlayPause()
    }
    if (event.code === "ArrowLeft") {
      event.preventDefault()
      stopPlayback()
      prevStep()
    }
    if (event.code === "ArrowRight") {
      event.preventDefault()
      stopPlayback()
      nextStep()
    }
  })
}

function init() {
  resizeBoards(state.size)
  bindEvents()
  renderAll()
}

init()
