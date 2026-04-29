(() => {
  const $ = (id) => document.getElementById(id);

  const FOLD = 0, CHECK_CALL = 1, RAISE_START = 2, ALL_IN = 9;

  const SUIT_GLYPH = { c: "\u2663", d: "\u2666", h: "\u2665", s: "\u2660" };

  let lastState = null;
  let userTouchedRaise = false;
  // Remember the last user-chosen raise bucket so we can highlight the active preset.
  let activeRaisePreset = null;
  // Track the pending auto-new-hand timer so we don't fire twice.
  let autoNextHandTimer = null;

  function cardEl(cardStr, isBoard = false) {
    const el = document.createElement("div");
    el.className = "card" + (isBoard ? " board-card" : "");
    if (!cardStr || cardStr === "??") {
      el.classList.add("back");
      return el;
    }
    const rank = cardStr[0];
    const suitChar = cardStr[1];
    const isRed = suitChar === "h" || suitChar === "d";
    el.classList.add(isRed ? "red" : "black");
    const glyph = SUIT_GLYPH[suitChar] || suitChar;
    el.innerHTML =
      `<span class="rank">${rank === "T" ? "10" : rank}</span>` +
      `<span class="suit">${glyph}</span>` +
      `<span class="center-suit">${glyph}</span>`;
    return el;
  }

  async function api(path, body) {
    const opts = body === undefined
      ? { method: "GET" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(path, opts);
    return await r.json();
  }

  function fmt(n, digits = 2, sign = false) {
    const s = n.toFixed(digits);
    return sign && n > 0 ? `+${s}` : s;
  }

  // Map a seat index to its slot (HERO is always slot 0, others go clockwise).
  function seatToSlot(seat, heroSeat) {
    return ((seat - heroSeat) % 6 + 6) % 6;
  }

  function renderSlot(slotIdx, p) {
    const slotEl = document.querySelector(`.slot-${slotIdx}`);
    const seatEl = slotEl.querySelector(".seat");
    const chipEl = slotEl.querySelector(".bet-chip");

    seatEl.classList.remove("hero", "to-act", "folded", "winner");

    if (!p) {
      seatEl.innerHTML = "";
      chipEl.classList.remove("visible");
      chipEl.textContent = "";
      return;
    }

    if (p.is_hero) seatEl.classList.add("hero");
    if (p.is_to_act) seatEl.classList.add("to-act");
    if (p.folded) seatEl.classList.add("folded");
    if (p.is_winner) seatEl.classList.add("winner");

    const tags = [];
    if (p.is_sb) tags.push(`<span class="tag sb">SB</span>`);
    if (p.is_bb) tags.push(`<span class="tag bb">BB</span>`);
    if (p.is_button) tags.push(`<span class="tag btn">D</span>`);
    if (p.all_in) tags.push(`<span class="tag allin">ALL-IN</span>`);

    const name = p.is_hero ? "YOU" : `Bot ${p.seat}`;
    const nameCls = p.is_hero ? "name hero-name" : "name";

    seatEl.innerHTML =
      `<div class="seat-head">
         <span class="pos-tag">${p.pos}</span>
         <span class="tags">${tags.join("")}</span>
       </div>
       <div class="${nameCls}">${name}</div>
       <div class="stack">${fmt(p.stack)} bb</div>`;

    const holes = document.createElement("div");
    holes.className = "holes";
    for (const c of p.hole) holes.appendChild(cardEl(c, false));
    seatEl.appendChild(holes);

    // External bet chip (outside seat box, toward pot)
    if (p.committed_street > 0) {
      chipEl.textContent = `${fmt(p.committed_street)} bb`;
      chipEl.classList.add("visible");
    } else {
      chipEl.classList.remove("visible");
      chipEl.textContent = "";
    }
  }

  function renderDealerChip(state) {
    document.querySelectorAll(".dealer-chip").forEach((el) => el.remove());
    const btn = state.players.find((p) => p.is_button);
    if (!btn) return;
    const slotIdx = seatToSlot(btn.seat, state.hero_seat);
    const slotEl = document.querySelector(`.slot-${slotIdx}`);
    if (!slotEl) return;
    const chip = document.createElement("div");
    chip.className = "dealer-chip";
    chip.textContent = "D";
    const feltRect = $("table").getBoundingClientRect();
    const slotRect = slotEl.getBoundingClientRect();
    const cx = feltRect.left + feltRect.width / 2;
    const cy = feltRect.top + feltRect.height / 2;
    const sx = slotRect.left + slotRect.width / 2;
    const sy = slotRect.top + slotRect.height / 2;
    // Place the chip toward the pot center, slightly inside the slot edge.
    const dx = cx - sx;
    const dy = cy - sy;
    const mag = Math.max(1, Math.hypot(dx, dy));
    const bias = 40;
    const anchorX = sx - feltRect.left + (dx / mag) * bias - 14;
    const anchorY = sy - feltRect.top + (dy / mag) * bias - 14;
    chip.style.left = anchorX + "px";
    chip.style.top = anchorY + "px";
    $("table").appendChild(chip);
  }

  function renderSession(state) {
    const s = state.session || {};
    $("s-hands").textContent = s.hands ?? 0;
    const pnl = s.pnl ?? 0;
    const pnlEl = $("s-pnl");
    pnlEl.textContent = fmt(pnl, 2, true);
    pnlEl.classList.toggle("pos", pnl > 0);
    pnlEl.classList.toggle("neg", pnl < 0);
    $("s-avg").textContent = fmt(s.avg_bb_per_hand ?? 0, 3, true);
    $("s-wlt").textContent = `${s.wins ?? 0}/${s.ties ?? 0}/${s.losses ?? 0}`;
    const bestEl = $("s-best");
    bestEl.textContent = fmt(s.best ?? 0, 2, true);
    bestEl.classList.toggle("pos", (s.best ?? 0) > 0);
    const worstEl = $("s-worst");
    worstEl.textContent = fmt(s.worst ?? 0, 2, true);
    worstEl.classList.toggle("neg", (s.worst ?? 0) < 0);
  }

  function renderRaiseControls(state) {
    const waiting = state.waiting_for_human && !state.done;
    const anyRaise = state.legal.any_raise;

    // preset bet-size buttons
    const rb = $("raise-buttons");
    rb.innerHTML = "";
    for (const rr of state.raise_buttons) {
      const b = document.createElement("button");
      b.textContent = `${rr.label} → ${rr.raise_to_bb}bb`;
      b.disabled = !waiting || !rr.legal;
      if (activeRaisePreset === rr.bucket) b.classList.add("active");
      b.onclick = () => {
        activeRaisePreset = rr.bucket;
        doAct(rr.bucket, null);
      };
      rb.appendChild(b);
    }

    const slider = $("raise-slider");
    const inp = $("raise-input");
    const hint = $("raise-hint");
    const minR = state.min_raise_to;
    const maxR = state.max_raise_to;
    const canRaiseUI = waiting && anyRaise && maxR > minR + 1e-6;

    slider.disabled = !canRaiseUI;
    inp.disabled = !canRaiseUI;
    $("btn-raise-custom").disabled = !canRaiseUI;
    $("btn-raise-minus").disabled = !canRaiseUI;
    $("btn-raise-plus").disabled = !canRaiseUI;

    slider.min = minR;
    slider.max = maxR;
    inp.min = minR;
    inp.max = maxR;

    if (!userTouchedRaise && canRaiseUI) {
      const defaultTo = state.current_bet > 0
        ? Math.min(maxR, Math.max(minR, state.current_bet * 2.5))
        : Math.min(maxR, Math.max(minR, state.pot > 0 ? state.pot : minR));
      inp.value = defaultTo.toFixed(2);
      slider.value = defaultTo;
    } else if (!canRaiseUI) {
      inp.value = "";
      slider.value = minR;
    }

    if (canRaiseUI) {
      hint.textContent = `min ${fmt(minR)} · max ${fmt(maxR)} · pot ${fmt(state.pot)}`;
    } else {
      hint.textContent = "";
    }
  }

  function renderActionButtons(state) {
    const waiting = state.waiting_for_human && !state.done;
    const legal = state.legal;

    $("btn-fold").disabled = !waiting || !legal.fold;
    $("btn-check-call").disabled = !waiting;
    $("btn-check-call").textContent = state.to_call > 0
      ? `Call ${fmt(state.to_call)} (C)`
      : "Check (C)";
    $("btn-allin").disabled = !waiting || !legal.allin;
  }

  function renderStatus(state) {
    const el = $("action-status");
    el.classList.remove("waiting", "done");
    if (state.done) {
      const pnl = state.settle ? state.settle.pnl : 0;
      const win = state.settle ? state.settle.winners.join(", ") : "";
      const sd = state.settle && state.settle.went_to_showdown ? "(showdown)" : "(folded out)";
      el.textContent = `手牌结束 ${sd} · 获胜: ${win} · HERO PnL ${fmt(pnl, 2, true)} bb — 按 N 开始新的手牌。`;
      el.classList.add("done");

      const banner = $("showdown-banner");
      banner.classList.remove("hidden");
      banner.textContent =
        (state.settle && state.settle.went_to_showdown ? "SHOWDOWN · " : "WINNER · ") +
        (win || "");
    } else if (state.waiting_for_human) {
      el.textContent = `轮到你 (${state.hero_pos}) · to_call ${fmt(state.to_call)} bb · pot ${fmt(state.pot)} bb`;
      el.classList.add("waiting");
      $("showdown-banner").classList.add("hidden");
    } else {
      el.textContent = `等 ${state.to_act_pos} 动作...`;
      $("showdown-banner").classList.add("hidden");
    }
  }

  function render(state) {
    lastState = state;
    if (!state.started) {
      $("action-status").textContent = "点击 New Hand 开始。";
      renderSession(state);
      $("toggle-reveal").checked = !!state.reveal_all;
      return;
    }
    $("hand-idx").textContent = "Hand #" + state.hand_idx;
    $("street").textContent = state.street_name;
    $("pot").textContent = "Pot " + fmt(state.pot) + " bb";
    $("pot-center").textContent = "Pot " + fmt(state.pot);

    const board = $("board");
    board.innerHTML = "";
    for (const c of state.board) board.appendChild(cardEl(c, true));
    // placeholder face-down slots so table doesn't look empty preflop
    for (let i = state.board.length; i < 5; i++) {
      const ph = document.createElement("div");
      ph.className = "card board-card back";
      ph.style.opacity = "0.25";
      board.appendChild(ph);
    }

    // Clear all slots first, then fill by slot index mapped from seat.
    for (let k = 0; k < 6; k++) renderSlot(k, null);
    for (const p of state.players) {
      const slotIdx = seatToSlot(p.seat, state.hero_seat);
      renderSlot(slotIdx, p);
    }
    renderDealerChip(state);

    $("log").textContent = state.log.join("\n");
    $("log").scrollTop = $("log").scrollHeight;

    renderActionButtons(state);
    renderRaiseControls(state);
    renderStatus(state);
    renderSession(state);

    $("toggle-reveal").checked = !!state.reveal_all;

    scheduleAutoNextHand(state);
  }

  // If the hand ended because HERO folded, auto-start the next hand
  // after a short delay so the user can glance at the result.
  function scheduleAutoNextHand(state) {
    if (autoNextHandTimer) { clearTimeout(autoNextHandTimer); autoNextHandTimer = null; }
    if (!state.started || !state.done) return;
    const hero = state.players.find((p) => p.is_hero);
    if (!hero || !hero.folded) return;
    autoNextHandTimer = setTimeout(() => {
      autoNextHandTimer = null;
      newHand();
    }, 1400);
  }

  async function doAct(bucket, raiseTo) {
    const body = { bucket };
    if (raiseTo !== null && raiseTo !== undefined) body.raise_to_bb = raiseTo;
    const state = await api("/api/act", body);
    userTouchedRaise = false;
    activeRaisePreset = null;
    render(state);
  }

  async function newHand() {
    if (autoNextHandTimer) { clearTimeout(autoNextHandTimer); autoNextHandTimer = null; }
    const state = await api("/api/new_hand", {});
    userTouchedRaise = false;
    activeRaisePreset = null;
    render(state);
  }

  async function resetSession() {
    if (!confirm("重置本局 session 统计？")) return;
    const state = await api("/api/reset_session", {});
    render(state);
  }

  async function toggleReveal(on) {
    const state = await api("/api/reveal", { on });
    render(state);
  }

  function syncSliderToInput() {
    const slider = $("raise-slider");
    const inp = $("raise-input");
    const v = parseFloat(slider.value);
    if (!isNaN(v)) {
      inp.value = v.toFixed(2);
      userTouchedRaise = true;
      activeRaisePreset = null;
      highlightPresetsByValue(v);
    }
  }
  function syncInputToSlider() {
    const slider = $("raise-slider");
    const inp = $("raise-input");
    const v = parseFloat(inp.value);
    if (!isNaN(v)) {
      slider.value = v;
      userTouchedRaise = true;
      activeRaisePreset = null;
      highlightPresetsByValue(v);
    }
  }

  function highlightPresetsByValue(v) {
    if (!lastState) return;
    const buttons = $("raise-buttons").querySelectorAll("button");
    let idx = -1;
    for (let i = 0; i < lastState.raise_buttons.length; i++) {
      const rr = lastState.raise_buttons[i];
      if (Math.abs(rr.raise_to_bb - v) < 0.01 && rr.legal) { idx = i; break; }
    }
    buttons.forEach((b, i) => b.classList.toggle("active", i === idx));
  }

  function stepRaise(delta) {
    const inp = $("raise-input");
    const slider = $("raise-slider");
    let v = parseFloat(inp.value);
    if (isNaN(v)) v = parseFloat(slider.min);
    v = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), v + delta));
    inp.value = v.toFixed(2);
    slider.value = v;
    userTouchedRaise = true;
    activeRaisePreset = null;
    highlightPresetsByValue(v);
  }

  function submitCustomRaise() {
    if ($("btn-raise-custom").disabled) return;
    const v = parseFloat($("raise-input").value);
    if (isNaN(v)) return;
    doAct(RAISE_START, v);
  }

  // ---- keyboard shortcuts ----
  function onKeyDown(e) {
    if (e.target.matches("input, textarea")) return;
    const k = e.key.toLowerCase();
    if (k === "n") { e.preventDefault(); $("btn-new-hand").click(); return; }
    if (!lastState || !lastState.waiting_for_human || lastState.done) return;
    if (k === "f") { e.preventDefault(); if (!$("btn-fold").disabled) $("btn-fold").click(); }
    else if (k === "c") { e.preventDefault(); if (!$("btn-check-call").disabled) $("btn-check-call").click(); }
    else if (k === "a") { e.preventDefault(); if (!$("btn-allin").disabled) $("btn-allin").click(); }
    else if (k === "r") { e.preventDefault(); submitCustomRaise(); }
    else if (["1", "2", "3", "4", "5", "6"].includes(k)) {
      const idx = parseInt(k, 10) - 1;
      const buttons = $("raise-buttons").querySelectorAll("button");
      if (buttons[idx] && !buttons[idx].disabled) {
        e.preventDefault();
        buttons[idx].click();
      }
    }
  }

  // ---- wiring ----
  window.addEventListener("DOMContentLoaded", async () => {
    $("btn-new-hand").onclick = newHand;
    $("btn-fold").onclick = () => doAct(FOLD, null);
    $("btn-check-call").onclick = () => doAct(CHECK_CALL, null);
    $("btn-allin").onclick = () => doAct(ALL_IN, null);
    $("btn-raise-custom").onclick = submitCustomRaise;
    $("btn-raise-minus").onclick = () => stepRaise(-0.5);
    $("btn-raise-plus").onclick = () => stepRaise(+0.5);
    $("raise-slider").oninput = syncSliderToInput;
    $("raise-input").oninput = syncInputToSlider;
    $("raise-input").onkeydown = (e) => {
      if (e.key === "Enter") { e.preventDefault(); submitCustomRaise(); }
    };
    $("btn-reset-session").onclick = resetSession;
    $("toggle-reveal").onchange = (e) => toggleReveal(e.target.checked);

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", () => {
      if (lastState && lastState.started) renderDealerChip(lastState);
    });

    const state = await api("/api/state");
    render(state);
  });
})();
