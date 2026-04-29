var POS = ['SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN'];
var SUIT_SYM = { s: '\u2660', h: '\u2665', d: '\u2666', c: '\u2663' };
var SUIT_COLOR = { s: 'black', h: 'red', d: 'red', c: 'black' };

var hands = [];
var curHand = 0;
var curStep = 0;
var autoTimer = null;

function parseCard(s) {
    if (!s || s.length < 2) return null;
    var r = s.slice(0, -1), su = s.slice(-1).toLowerCase();
    if (!SUIT_SYM[su]) return null;
    var sym = SUIT_SYM[su];
    return { rank: r, suit: su, sym: sym, color: SUIT_COLOR[su], display: r + sym };
}

function cardHTML(c, cls) {
    if (!c) return '';
    cls = cls || '';
    return '<span class="mini-card ' + c.color + ' ' + cls + '">' + c.display + '</span>';
}

function boardCardHTML(c, hidden, anim) {
    if (!c) return '';
    if (hidden) return '<div class="board-card hidden"></div>';
    var a = anim ? ' deal-anim' : '';
    return '<div class="board-card ' + c.color + a + '">' + c.display + '</div>';
}

function parseHands(text) {
    var result = [];
    var blocks = text.split(/^==========\s*Hand\s+#(\d+)\s*==========\s*$/m);
    for (var i = 1; i < blocks.length; i += 2) {
        var handNum = parseInt(blocks[i]);
        var body = blocks[i + 1];
        var hand = parseOneHand(handNum, body);
        if (hand) result.push(hand);
    }
    return result;
}

function parseOneHand(num, body) {
    var lines = body.split('\n');
    var hand = { num: num, seats: [], holes: [], events: [], showdown: [], winners: [], finalStacks: [], fullBoard: null };

    for (var li = 0; li < lines.length; li++) {
        var line = lines[li].trim();
        if (!line || line.charAt(0) === '#') continue;
        var m;

        if ((m = line.match(/^Seats:\s*(.*)/))) {
            hand.seats = m[1].split('|').map(function (s, i) {
                var p = s.trim().match(/(\w+)=([\d.]+)bb/);
                return p ? { pos: p[1], stack: parseFloat(p[2]), idx: i } : { pos: '?', stack: 100, idx: i };
            });
            continue;
        }
        if ((m = line.match(/^Holes:\s*(.*)/))) {
            hand.holes = m[1].split('|').map(function (s) {
                var p = s.trim().match(/(\w+)=(\S+)/);
                if (!p) return { pos: '?', cards: [] };
                var cs = p[2].match(/.{2}/g) || [];
                return { pos: p[1], cards: cs.map(parseCard).filter(Boolean) };
            });
            continue;
        }
        if ((m = line.match(/^Blinds:\s*SB\s*([\d.]+)\s*\/\s*BB\s*([\d.]+)/))) {
            hand.sb = parseFloat(m[1]); hand.bb = parseFloat(m[2]);
            continue;
        }
        if ((m = line.match(/^--\s*(Preflop|Flop|Turn|River)\s*--/))) {
            hand.events.push({ type: 'street', street: m[1] });
            continue;
        }
        if ((m = line.match(/^--\s*(Flop|Turn|River):\s*\[([^\]]*)\]\s*\(pot\s*([\d.]+)\)\s*--/))) {
            var cards = m[2].trim().split(/\s+/).filter(Boolean).map(parseCard).filter(Boolean);
            hand.events.push({ type: 'deal', street: m[1], cards: cards, pot: parseFloat(m[3]) });
            continue;
        }
        if ((m = line.match(/^\s*(\w+):\s*folds/))) {
            hand.events.push({ type: 'action', pos: m[1], posIdx: POS.indexOf(m[1]), action: 'fold' });
            continue;
        }
        if ((m = line.match(/^\s*(\w+):\s*checks/))) {
            hand.events.push({ type: 'action', pos: m[1], posIdx: POS.indexOf(m[1]), action: 'check' });
            continue;
        }
        if ((m = line.match(/^\s*(\w+):\s*calls\s*([\d.-]+)/))) {
            hand.events.push({
                type: 'action', pos: m[1], posIdx: POS.indexOf(m[1]),
                action: 'call', amount: parseFloat(m[2])
            });
            continue;
        }
        if ((m = line.match(/^\s*(\w+):\s*(raise\s*[\d.]+x-pot|ALL-IN)\s+to\s*([\d.]+)\s*\(\+([\d.]+),\s*pot\s*([\d.]+)\)\s*(\[ALL-IN\])?/))) {
            hand.events.push({
                type: 'action', pos: m[1], posIdx: POS.indexOf(m[1]),
                action: m[2].trim() === 'ALL-IN' ? 'ALL-IN' : m[2].trim(),
                to: parseFloat(m[3]), added: parseFloat(m[4]),
                pot: parseFloat(m[5]), isAllIn: true
            });
            continue;
        }
        if ((m = line.match(/^Board:\s*\[([^\]]*)\]/))) {
            var boardContent = m[1].trim();
            if (boardContent && boardContent !== '(no board)') {
                hand.fullBoard = boardContent.split(/\s+/).filter(Boolean).map(parseCard).filter(Boolean);
            } else {
                hand.fullBoard = [];
            }
            continue;
        }
        if ((m = line.match(/^Winners:\s*(.*)/))) {
            hand.winners = m[1].split(',').map(function (s) { return s.trim(); });
            continue;
        }
        if ((m = line.match(/^Showdown:\s*(.*)/))) {
            hand.showdown = m[1].trim().split(/\s{2,}/).map(function (s) {
                var p = s.match(/(\w+)=(\S+)/);
                if (!p) return null;
                var cs = p[2].match(/.{2}/g) || [];
                return { pos: p[1], cards: cs.map(parseCard).filter(Boolean) };
            }).filter(Boolean);
            continue;
        }
        if ((m = line.match(/^Final stacks:\s*(.*)/))) {
            hand.finalStacks = m[1].split('|').map(function (s, i) {
                var p = s.trim().match(/(\w+)=([\d.]+)/);
                return p ? { pos: p[1], stack: parseFloat(p[2]) } : { pos: POS[i], stack: 100 };
            });
            continue;
        }
    }
    if (!hand.fullBoard) hand.fullBoard = [];
    return hand;
}

function render() {
    if (!hands.length) return;
    var h = hands[curHand];
    var evts = h.events;
    var step = curStep;
    if (!evts.length) return;

    var pot = 0;
    var seatStates = [];
    for (var si = 0; si < 6; si++) {
        seatStates.push({
            pos: POS[si],
            stack: h.seats[si] ? h.seats[si].stack : 100,
            hole: [],
            holeRevealed: false,
            folded: false,
            active: false,
            winner: false,
            currentBet: 0
        });
    }
    var holeSource = h.holes.length ? h.holes : h.showdown;
    for (var sdi0 = 0; sdi0 < holeSource.length; sdi0++) {
        var sd0 = holeSource[sdi0];
        var ssi0 = POS.indexOf(sd0.pos);
        if (ssi0 >= 0 && ssi0 < 6) {
            seatStates[ssi0].hole = sd0.cards;
            seatStates[ssi0].holeRevealed = true;
        }
    }
    var logLines = [];

    var lastDealStep = -1;
    for (var di = 0; di <= Math.min(step, evts.length - 1); di++) {
        if (evts[di].type === 'deal') lastDealStep = di;
    }

    for (var i = 0; i <= Math.min(step, evts.length - 1); i++) {
        var e = evts[i];
        if (e.type === 'street') {
            for (var j = 0; j < 6; j++) seatStates[j].currentBet = 0;
            logLines.push({ cls: 'log-street', text: e.street, isCurrent: false });
        } else if (e.type === 'deal') {
            pot = e.pot;
            for (var k = 0; k < 6; k++) seatStates[k].currentBet = 0;
            var cardStr = e.cards.map(function (c) { return c.display; }).join(' ');
            logLines.push({ cls: 'log-street', text: e.street + ': [' + cardStr + ']  pot ' + e.pot, isCurrent: false });
        } else if (e.type === 'action') {
            var idx = e.posIdx;
            if (idx >= 0 && idx < 6) {
                seatStates[idx].active = (i === step);
                var actText = '';
                var actClass = '';
                if (e.action === 'fold') {
                    seatStates[idx].folded = true;
                    actText = 'fold';
                    actClass = 'fold-act';
                } else if (e.action === 'check') {
                    actText = 'check';
                    actClass = 'check-act';
                } else if (e.action === 'call') {
                    var displayAmt = Math.abs(e.amount);
                    actText = 'call ' + displayAmt;
                    seatStates[idx].currentBet = Math.abs(e.amount);
                    actClass = 'act';
                } else {
                    var label = e.isAllIn ? 'ALL-IN' : e.action;
                    actText = label + ' \u2192 ' + e.to;
                    seatStates[idx].currentBet = e.added || e.to;
                    actClass = 'act';
                }

                var logText = '<span class="pos">' + e.pos + '</span>: <span class="' + actClass + '">' + actText + '</span>';
                if (e.pot) logText += ' <span class="amt">pot ' + e.pot + '</span>';
                logLines.push({ cls: 'log-action', text: logText, isCurrent: (i === step) });
            }
        }
    }

    if (step >= evts.length - 1) {
        for (var wi = 0; wi < h.winners.length; wi++) {
            var wsi = POS.indexOf(h.winners[wi]);
            if (wsi >= 0 && wsi < 6) seatStates[wsi].winner = true;
        }
        if (h.finalStacks) {
            for (var fi = 0; fi < h.finalStacks.length; fi++) {
                var fs = h.finalStacks[fi];
                var fsi = POS.indexOf(fs.pos);
                if (fsi >= 0 && fsi < 6) seatStates[fsi].stack = fs.stack;
            }
        }
        logLines.push({ cls: 'log-result', text: 'Winner: ' + h.winners.join(', '), isCurrent: false });
        for (var sdi2 = 0; sdi2 < h.showdown.length; sdi2++) {
            var sd2 = h.showdown[sdi2];
            logLines.push({ cls: 'log-showdown', text: sd2.pos + ' = ' + sd2.cards.map(function (c) { return c.display; }).join(' '), isCurrent: false });
        }
    }

    var seatHTML = '';
    for (var si2 = 0; si2 < 6; si2++) {
        var s = seatStates[si2];
        var cls = 'seat seat-' + si2;
        if (s.active) cls += ' active';
        if (s.folded) cls += ' folded';
        if (s.winner) cls += ' winner';
        var cardsHTML = '';
        if (s.holeRevealed && s.hole.length) {
            cardsHTML = s.hole.map(function (c) { return cardHTML(c); }).join('');
        } else {
            var dimCls = s.folded ? ' dimmed' : '';
            cardsHTML = '<span class="mini-card back' + dimCls + '"></span><span class="mini-card back' + dimCls + '"></span>';
        }
        var dealerHTML = (si2 === 5) ? '<div class="seat-dealer">D</div>' : '';
        seatHTML += '<div class="' + cls + '">' +
            '<div class="seat-box">' +
            dealerHTML +
            '<div class="seat-name">' + s.pos + '</div>' +
            '<div class="seat-cards">' + cardsHTML + '</div>' +
            '<div class="seat-stack">' + s.stack.toFixed(1) + ' bb</div>' +
            '</div></div>';
    }
    document.getElementById('seats').innerHTML = seatHTML;

    var betHTML = '';
    for (var bi = 0; bi < 6; bi++) {
        var bs = seatStates[bi];
        if (bs.currentBet > 0 && !bs.folded) {
            betHTML += '<div class="bet-chip bet-' + bi + '">' +
                '<div class="chip-icon"></div>' +
                '<div class="chip-label">' + bs.currentBet.toFixed(1) + ' bb</div>' +
                '</div>';
        }
    }
    document.getElementById('bets').innerHTML = betHTML;

    var fullBoard = h.fullBoard || [];
    var boardHTML = '';
    var streetMap = { Flop: 3, Turn: 4, River: 5 };
    var dealt = 0;
    for (var di2 = 0; di2 <= Math.min(step, evts.length - 1); di2++) {
        if (evts[di2].type === 'deal') {
            var target = streetMap[evts[di2].street] || 0;
            if (target > dealt) dealt = target;
        }
    }
    if (step >= evts.length - 1) dealt = fullBoard.length;
    var showBoard = dealt > 0 || fullBoard.length > 0;
    if (showBoard) {
        for (var bii = 0; bii < 5; bii++) {
            if (bii < dealt && fullBoard[bii]) {
                var isDeal = (lastDealStep === step);
                boardHTML += boardCardHTML(fullBoard[bii], false, isDeal);
            } else {
                boardHTML += boardCardHTML(null, true, false);
            }
        }
    }
    document.getElementById('boardArea').innerHTML = boardHTML;

    var potText = '';
    if (pot > 0) {
        potText = '<div class="pot-icon">$</div> Pot: ' + pot.toFixed(1) + ' bb';
    }
    document.getElementById('potDisplay').innerHTML = potText;

    var logHTML = '';
    for (var li2 = 0; li2 < logLines.length; li2++) {
        var ll = logLines[li2];
        var curCls = ll.isCurrent ? ' current' : '';
        logHTML += '<div class="' + ll.cls + curCls + '">' + ll.text + '</div>';
    }
    document.getElementById('actionLog').innerHTML = logHTML;
    document.getElementById('actionLog').scrollTop = 99999;

    document.getElementById('handCounter').textContent = 'Hand ' + (curHand + 1) + ' / ' + hands.length;

    var totalSteps = evts.length;
    var pct = totalSteps > 1 ? ((step) / (totalSteps - 1)) * 100 : 100;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressLabel').textContent = 'Step ' + (step + 1) + ' / ' + totalSteps;
}

function clickProgress(e) {
    if (!hands.length) return;
    var bar = document.getElementById('progressBar');
    var rect = bar.getBoundingClientRect();
    var pct = (e.clientX - rect.left) / rect.width;
    var totalSteps = hands[curHand].events.length;
    var target = Math.round(pct * (totalSteps - 1));
    target = Math.max(0, Math.min(totalSteps - 1, target));
    curStep = target;
    render();
}

function nextStep() {
    if (!hands.length) return;
    if (curStep < hands[curHand].events.length - 1) { curStep++; render(); }
    else nextHand();
}
function prevStep() {
    if (!hands.length) return;
    if (curStep > 0) { curStep--; render(); }
    else prevHand();
}
function nextHand() {
    if (!hands.length || curHand >= hands.length - 1) return;
    curHand++; curStep = 0; render(); updateSelect();
}
function prevHand() {
    if (!hands.length || curHand <= 0) return;
    curHand--; curStep = 0; render(); updateSelect();
}
function goFirst() { if (!hands.length) return; curHand = 0; curStep = 0; render(); updateSelect(); }
function goLast() { if (!hands.length) return; curHand = hands.length - 1; curStep = hands[curHand].events.length - 1; render(); updateSelect(); }
function jumpHand(idx) { curHand = parseInt(idx); curStep = 0; render(); }
function updateSelect() { document.getElementById('handSelect').value = curHand; }

function toggleAuto() {
    var btn = document.getElementById('btnAuto');
    if (autoTimer) {
        clearInterval(autoTimer); autoTimer = null;
        btn.textContent = '\u25B6 Auto';
        btn.classList.remove('active');
        return;
    }
    btn.textContent = '\u23F8 Stop';
    btn.classList.add('active');
    autoTimer = setInterval(function () {
        if (curStep < hands[curHand].events.length - 1) { curStep++; render(); }
        else if (curHand < hands.length - 1) { curHand++; curStep = 0; render(); updateSelect(); }
        else { clearInterval(autoTimer); autoTimer = null; btn.textContent = '\u25B6 Auto'; btn.classList.remove('active'); }
    }, 550);
}

function loadFile(file) {
    var reader = new FileReader();
    reader.onload = function (e) {
        var text = e.target.result;
        hands = parseHands(text);
        if (!hands.length) { alert('\u672A\u627E\u5230\u6709\u6548\u624B\u724C\u6570\u636E'); return; }
        document.getElementById('dropZone').classList.add('hasData');
        document.getElementById('controls').style.display = 'flex';
        document.getElementById('progressWrap').style.display = 'flex';
        var sel = document.getElementById('handSelect');
        sel.innerHTML = '';
        for (var i = 0; i < hands.length; i++) {
            var o = document.createElement('option');
            o.value = i; o.textContent = '#' + (i + 1);
            sel.appendChild(o);
        }
        curHand = 0; curStep = 0; render(); updateSelect();
    };
    reader.readAsText(file);
}

document.getElementById('fileInput').addEventListener('change', function (e) {
    if (e.target.files.length) loadFile(e.target.files[0]);
});

var dz = document.getElementById('dropZone');
dz.addEventListener('dragover', function (e) { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', function () { dz.classList.remove('dragover'); });
dz.addEventListener('drop', function (e) {
    e.preventDefault(); dz.classList.remove('dragover');
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});

document.addEventListener('keydown', function (e) {
    if (!hands.length) return;
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); nextStep(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); prevStep(); }
    else if (e.key === 'ArrowUp' || e.key === 'PageUp') { e.preventDefault(); prevHand(); }
    else if (e.key === 'ArrowDown' || e.key === 'PageDown') { e.preventDefault(); nextHand(); }
    else if (e.key === 'Home') { e.preventDefault(); goFirst(); }
    else if (e.key === 'End') { e.preventDefault(); goLast(); }
});
