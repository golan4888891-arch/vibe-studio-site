// HUM voice-to-notes — browser port of engine/transcribe (Rust). Same algorithm, same constants,
// f32 rounding where the Rust code works in f32. Parity is checked by tests/parity/check_transcribe.js
// against the Rust output on the same recordings.
(function (root) {
  "use strict";
  const SR = 16000, HOP = 160, WIN = 512, FMIN = 60, FMAX = 1000, MAX_SECONDS = 180;
  const BINS_PER_SEMITONE = 10, N_THRESH = 100, BOLTZMANN = 2, NO_TROUGH_PROB = 0.01, SWITCH_PROB = 0.01, MAX_TRANSITION_RATE = 35.92;
  const f32 = Math.fround;
  // Rust f64::round: half away from zero
  const rnd = (x) => (x < 0 ? -Math.round(-x) : Math.round(x));
  const clamp = (x, a, b) => Math.min(Math.max(x, a), b);

  function toAnalysisRate(samples, channels, sampleRate) {
    const ch = Math.max(1, channels | 0);
    const n = Math.floor(samples.length / ch);
    const mono = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let c = 0; c < ch; c++) s = f32(s + samples[i * ch + c]);
      mono[i] = f32(s / ch);
    }
    if (sampleRate === SR) return mono;
    const ratio = sampleRate / SR;
    const cutoff = ratio > 1 ? 0.45 / ratio : 0.5;
    const taps = 63, m = taps - 1;
    const h = new Float64Array(taps);
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const x = i - m / 2;
      const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
      const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / m) + 0.08 * Math.cos((4 * Math.PI * i) / m);
      h[i] = sinc * w;
      sum += h[i];
    }
    for (let i = 0; i < taps; i++) h[i] /= sum;
    const half = Math.floor(taps / 2);
    const filt = (c) => {
      let acc = 0;
      for (let k = 0; k < taps; k++) {
        const idx = c + k - half;
        if (idx >= 0 && idx < n) acc += h[k] * mono[idx];
      }
      return acc;
    };
    const outLen = Math.floor(n / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      const a = filt(i0), b = filt(Math.min(i0 + 1, n - 1));
      out[i] = f32(a + (b - a) * frac);
    }
    return out;
  }

  const db = (x) => 20 * Math.log10(Math.max(x, 1e-9));

  function frameRmsDb(x) {
    const n = Math.floor(x.length / HOP) + 1;
    const out = new Float64Array(n);
    for (let t = 0; t < n; t++) {
      const c = t * HOP, a = Math.max(0, c - WIN / 2), b = Math.min(c + WIN / 2, x.length);
      if (b <= a) { out[t] = -120; continue; }
      let e = 0;
      for (let i = a; i < b; i++) e += x[i] * x[i];
      out[t] = db(Math.sqrt(e / (b - a)));
    }
    return out;
  }

  function validate(x) {
    const secs = x.length / SR;
    if (secs < 0.5) throw "ההקלטה קצרה מדי (פחות מחצי שנייה). נסה לזמזם קצת יותר";
    if (secs > MAX_SECONDS) throw `ההקלטה ארוכה מדי לניתוח (יותר מ-${MAX_SECONDS / 60} דקות). אפשר לחלק אותה לחלקים`;
    let peak = 0;
    for (const v of x) peak = Math.max(peak, Math.abs(v));
    if (db(peak) < -45) throw "כמעט לא נקלט קול. בדוק שהמיקרופון פועל ושהוא קרוב אליך";
    const w = [];
    if (db(peak) < -30) w.push("ההקלטה שקטה מאוד — כדאי להתקרב למיקרופון");
    let clipped = 0;
    for (const v of x) if (Math.abs(v) >= f32(0.99)) clipped++;
    if (clipped > x.length * 0.005) w.push("ההקלטה חזקה מדי ונחתכה בחלקים — כדאי להתרחק מעט מהמיקרופון");
    return w;
  }

  // ---------------- pYIN ----------------
  const nBins = () => Math.floor(12 * BINS_PER_SEMITONE * Math.log2(FMAX / FMIN)) + 1;
  const binFreq = (b) => FMIN * Math.pow(2, b / (12 * BINS_PER_SEMITONE));
  function betaProbs() {
    const cdf = (x) => 1 - Math.pow(1 - x, 19) - 19 * x * Math.pow(1 - x, 18);
    const p = new Float64Array(N_THRESH);
    for (let k = 0; k < N_THRESH; k++) p[k] = cdf((k + 1) / N_THRESH) - cdf(k / N_THRESH);
    return p;
  }
  const boltzmann = (k, n) => ((1 - Math.exp(-BOLTZMANN)) * Math.exp(-BOLTZMANN * k)) / (1 - Math.exp(-BOLTZMANN * n));

  function frameCandidates(p, off, tauMin, tauMax, beta, d, cm) {
    for (let tau = 1; tau <= tauMax; tau++) {
      let s = 0;
      for (let j = 0; j < WIN; j++) {
        const e = f32(p[off + j] - p[off + j + tau]);
        s += e * e;
      }
      d[tau] = s;
    }
    cm[0] = 1;
    let run = 0;
    for (let tau = 1; tau <= tauMax; tau++) {
      run += d[tau];
      cm[tau] = run > 0 ? (d[tau] * tau) / run : 1;
    }
    const troughs = [];
    for (let tau = tauMin; tau <= tauMax; tau++) {
      const leftOk = tau === tauMin || cm[tau] < cm[tau - 1];
      const rightOk = tau === tauMax || cm[tau] <= cm[tau + 1];
      if (leftOk && rightOk) troughs.push(tau);
    }
    if (!troughs.length) return [];
    const probs = new Float64Array(troughs.length);
    for (let k = 0; k < N_THRESH; k++) {
      const thr = (k + 1) / N_THRESH;
      const below = [];
      for (let i = 0; i < troughs.length; i++) if (cm[troughs[i]] < thr) below.push(i);
      for (let r = 0; r < below.length; r++) probs[below[r]] += boltzmann(r, below.length) * beta[k];
    }
    let gmin = 0;
    for (let i = 1; i < troughs.length; i++) if (cm[troughs[i]] < cm[troughs[gmin]]) gmin = i;
    const hmin = cm[troughs[gmin]];
    let none = 0;
    for (let k = 0; k < N_THRESH; k++) if ((k + 1) / N_THRESH <= hmin) none += beta[k];
    probs[gmin] += NO_TROUGH_PROB * none;
    const out = [];
    for (let i = 0; i < troughs.length; i++) {
      if (!(probs[i] > 0)) continue;
      const tau = troughs[i];
      let shift = 0;
      if (tau > 1 && tau < tauMax) {
        const a = cm[tau - 1], b = cm[tau], c = cm[tau + 1];
        const den = a - 2 * b + c;
        if (Math.abs(den) > 1e-12) shift = clamp((0.5 * (a - c)) / den, -1, 1);
      }
      out.push([SR / (tau + shift), probs[i]]);
    }
    return out;
  }

  function pyin(x) {
    const tauMin = Math.floor(SR / FMAX), tauMax = Math.ceil(SR / FMIN);
    const nb = nBins();
    const frames = Math.floor(x.length / HOP) + 1;
    const padded = new Float32Array(WIN / 2 + x.length + WIN + tauMax + 1);
    padded.set(x, WIN / 2);
    const beta = betaProbs();
    const d = new Float64Array(tauMax + 2), cm = new Float64Array(tauMax + 2);
    const obs = [], vprob = new Float64Array(frames);
    for (let t = 0; t < frames; t++) {
      const off = t * HOP;
      let energy = 0;
      for (let j = 0; j < WIN; j++) energy += padded[off + j] * padded[off + j];
      const cands = energy < 1e-10 ? [] : frameCandidates(padded, off, tauMin, tauMax, beta, d, cm);
      const o = [];
      let total = 0;
      for (const [f, p] of cands) {
        const b = rnd(12 * BINS_PER_SEMITONE * Math.log2(f / FMIN));
        if (b >= 0 && b < nb) {
          const e = o.find((q) => q[0] === b);
          if (e) e[1] += p; else o.push([b, p]);
          total += p;
        }
      }
      vprob[t] = Math.min(total, 1);
      obs.push(o);
    }
    const maxSt = rnd((MAX_TRANSITION_RATE * 12 * HOP) / SR);
    const half = maxSt * BINS_PER_SEMITONE, w = 2 * half + 1;
    const tri = (dl) => 1 - dl / (half + 1);
    const rowNorm = new Float64Array(nb);
    for (let i = 0; i < nb; i++) {
      let s = 0;
      for (let j = Math.max(0, i - half); j <= Math.min(i + half, nb - 1); j++) s += tri(Math.abs(i - j));
      rowNorm[i] = s;
    }
    const ltab = new Float64Array(nb * w).fill(-Infinity);
    for (let ib = 0; ib < nb; ib++)
      for (let jb = Math.max(0, ib - half); jb <= Math.min(ib + half, nb - 1); jb++) ltab[ib * w + (jb + half - ib)] = Math.log(tri(Math.abs(ib - jb)) / rowNorm[ib]);
    const ln = (p) => Math.log(p + 1e-300);
    const ls = ln(SWITCH_PROB), lns = ln(1 - SWITCH_PROB);
    const ns = 2 * nb;
    let delta = new Float64Array(ns).fill(ln(1 / ns));
    let next = new Float64Array(ns);
    const back = new Uint16Array(frames * ns);
    const obsFull = new Float64Array(ns);
    const fillObs = (t) => {
      const unv = ln((1 - vprob[t]) / nb), z = ln(0);
      for (let j = 0; j < nb; j++) obsFull[j] = z;
      for (let j = nb; j < ns; j++) obsFull[j] = unv;
      for (const [b, p] of obs[t]) obsFull[b] = ln(p);
    };
    fillObs(0);
    for (let j = 0; j < ns; j++) delta[j] += obsFull[j];
    for (let t = 1; t < frames; t++) {
      fillObs(t);
      for (let j = 0; j < ns; j++) {
        const jb = j % nb, jv = j < nb;
        const lo = Math.max(0, jb - half), hi = Math.min(jb + half, nb - 1);
        let best = -Infinity, arg = 0;
        for (let ib = lo; ib <= hi; ib++) {
          const lt = ltab[ib * w + (jb + half - ib)];
          // voiced source first, then unvoiced (same order as the Rust loop)
          let v = delta[ib] + lt + (jv ? lns : ls);
          if (v > best) { best = v; arg = ib; }
          v = delta[nb + ib] + lt + (!jv ? lns : ls);
          if (v > best) { best = v; arg = nb + ib; }
        }
        next[j] = best + obsFull[j];
        back[t * ns + j] = arg;
      }
      const tmp = delta; delta = next; next = tmp;
    }
    let state = 0;
    for (let j = 1; j < ns; j++) if (delta[j] > delta[state]) state = j;
    const f0 = new Float64Array(frames), voiced = new Array(frames).fill(false);
    for (let t = frames - 1; t >= 0; t--) {
      voiced[t] = state < nb;
      f0[t] = voiced[t] ? binFreq(state % nb) : 0;
      if (t > 0) state = back[t * ns + state];
    }
    return { f0, voiced, voiced_prob: vprob };
  }
  const midiOf = (tr, i) => (tr.voiced[i] && tr.f0[i] > 0 ? 69 + 12 * Math.log2(tr.f0[i] / 440) : null);

  // ---------------- notes ----------------
  const FS = HOP / SR, MIN_NOTE = 6, SPLIT_SEMITONES = 0.6, SPLIT_FRAMES = 5, DIP_DB = 6, RISE_DB = 4, GAP_FRAMES = 3;
  function median(v) {
    v.sort((a, b) => a - b);
    const n = v.length;
    if (!n) return 0;
    return n % 2 === 1 ? v[(n - 1) / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
  function tuningOffset(tr) {
    let c = 0, s = 0;
    for (let i = 1; i < tr.f0.length - 1; i++) {
      const m = midiOf(tr, i), a = midiOf(tr, i - 1), b = midiOf(tr, i + 1);
      if (m === null || a === null || b === null) continue;
      if (Math.abs(b - a) > 0.15) continue;
      const ang = 2 * Math.PI * (m - rnd(m));
      const wt = Math.max(tr.voiced_prob[i], 0.05);
      c += wt * Math.cos(ang);
      s += wt * Math.sin(ang);
    }
    return c === 0 && s === 0 ? 0 : Math.atan2(s, c) / (2 * Math.PI);
  }
  function segment(tr, rms) {
    const n = Math.min(tr.f0.length, rms.length);
    const mc = Array.from({ length: n }, (_, i) => midiOf(tr, i));
    const regions = [];
    let i = 0;
    while (i < n) {
      if (mc[i] === null) { i++; continue; }
      const a = i;
      let b = i + 1, gap = 0;
      for (let j = i + 1; j < n; j++) {
        if (mc[j] !== null) { b = j + 1; gap = 0; } else if (++gap > GAP_FRAMES) break;
      }
      regions.push([a, b]);
      i = b;
    }
    const segs = [];
    for (const [a, b] of regions) {
      const sm = [];
      for (let t = a; t < b; t++) {
        if (mc[t] === null) { sm.push(null); continue; }
        const w = [];
        for (let k = Math.max(0, t - 2); k < Math.min(t + 3, b); k++) if (mc[k] !== null) w.push(mc[k]);
        sm.push(median(w));
      }
      let start = a, cut = "phrase", devRun = 0, peak = -Infinity;
      for (let t = a; t < b; t++) {
        const k = t - a;
        peak = Math.max(peak, rms[t]);
        let split = null;
        if (t >= start + 3) {
          const ref = [];
          for (let q = start - a; q < k; q++) if (sm[q] !== null) ref.push(sm[q]);
          if (sm[k] !== null && ref.length) {
            const r = median(ref);
            if (Math.abs(sm[k] - r) > SPLIT_SEMITONES) {
              devRun++;
              if (devRun >= SPLIT_FRAMES) split = [t + 1 - SPLIT_FRAMES, "pitch"];
            } else devRun = 0;
          }
        }
        if (!split && t >= start + 8) {
          const lo = Math.max(Math.max(0, t - 6), start + 2);
          if (lo < t) {
            let mi = lo, mv = rms[lo];
            for (let q = lo; q < t; q++) if (rms[q] < mv) { mv = rms[q]; mi = q; }
            if (peak - mv >= DIP_DB && rms[t] - mv >= RISE_DB && mi > start + 3) split = [mi + 1, "energy"];
          }
        }
        if (split) {
          segs.push([start, split[0], cut]);
          start = split[0];
          cut = split[1];
          devRun = 0;
          peak = -Infinity;
          for (let q = start; q <= t; q++) peak = Math.max(peak, rms[q]);
        }
      }
      segs.push([start, b, cut]);
    }
    const raws = [];
    for (const [a, b, cut] of segs) {
      const voiced = [];
      for (let t = a; t < b; t++) if (mc[t] !== null) voiced.push(t);
      if (voiced.length < MIN_NOTE) continue;
      const len = voiced.length;
      const [s0, s1] = len >= 10 ? [Math.floor(len / 4), len - Math.floor((len * 15) / 100)] : [0, len];
      const core = voiced.slice(s0, s1).map((t) => mc[t]);
      let mean = 0;
      for (const v of core) mean += v;
      mean /= core.length;
      let sq = 0;
      for (const v of core) sq += (v - mean) * (v - mean);
      const std = Math.sqrt(sq / core.length);
      const p = median(core);
      let vp = 0, level = 0;
      for (const t of voiced) { vp += tr.voiced_prob[t]; level += rms[t]; }
      raws.push({ a, b, cut, p, std, len, vp: vp / len, level: level / len });
    }
    let c = 0, sn = 0;
    for (const r of raws) {
      const ang = 2 * Math.PI * (r.p - rnd(r.p));
      c += r.len * Math.cos(ang);
      sn += r.len * Math.sin(ang);
    }
    const off = raws.length ? Math.atan2(sn, c) / (2 * Math.PI) : tuningOffset(tr);
    let peakDb = -Infinity;
    for (const v of rms) peakDb = Math.max(peakDb, v);
    const notes = raws.map((r) => {
      const p = r.p - off;
      const pitch = clamp(rnd(p), 0, 127);
      const conf = clamp(r.vp * Math.exp(-r.std / 0.35) * Math.min(r.len / 10, 1) * (1 - 1.6 * Math.abs(p - rnd(p))), 0, 1);
      const velocity = clamp(rnd(50 + (70 * (r.level - (peakDb - 30))) / 30), 30, 120);
      return { ev: { pitch, start_seconds: r.a * FS, duration_seconds: (r.b - r.a) * FS, velocity, confidence: rnd(conf * 100) / 100 }, cut: r.cut, a: r.a, b: r.b };
    });
    const out = [];
    for (const it of notes) {
      const last = out[out.length - 1];
      if (last && it.cut === "pitch" && last.ev.pitch === it.ev.pitch && it.a <= last.b + 2) {
        last.b = it.b;
        last.ev.duration_seconds = (last.b - last.a) * FS;
        last.ev.confidence = Math.min(last.ev.confidence, it.ev.confidence);
        continue;
      }
      out.push(it);
    }
    return { notes: out.map((x) => x.ev), tuning_cents: rnd(off * 100) };
  }

  // ---------------- tempo, grid, key ----------------
  function gridOffset(notes, unit) {
    let best = [Infinity, 0];
    for (let k = 0; k < 50; k++) {
      const off = (unit * k) / 50;
      let err = 0;
      for (const n of notes) { const r = (n.start_seconds - off) / unit; err += (r - rnd(r)) ** 2; }
      if (err < best[0] - 1e-12) best = [err, off];
    }
    return best[1];
  }
  function estimateTempo(notes) {
    const iois = [];
    for (let i = 1; i < notes.length; i++) { const d = notes[i].start_seconds - notes[i - 1].start_seconds; if (d > 0.08) iois.push(d); }
    if (iois.length < 3) return { bpm: null, confidence: 0, offset_seconds: notes.length ? notes[0].start_seconds : 0 };
    let best = [0, 0, 0];
    for (let bpm = 50; bpm <= 200 + 1e-9; bpm += 0.5) {
      const unit = 60 / bpm / 2;
      let fit = 0;
      for (const d of iois) {
        const r = d / unit, k = clamp(rnd(r), 1, 16), dev = (r - k) / Math.sqrt(k);
        fit += Math.exp(-(dev * dev) / (2 * 0.12 * 0.12));
      }
      fit /= iois.length;
      const prior = Math.exp(-(Math.log2(bpm / 105) ** 2) / (2 * 0.45 * 0.45));
      const score = fit * (0.6 + 0.4 * prior);
      if (score > best[0] + 1e-12) best = [score, bpm, fit];
    }
    const bpm = rnd(best[1]);
    return { bpm, confidence: rnd(best[2] * 100) / 100, offset_seconds: gridOffset(notes, 60 / bpm / 4) };
  }
  function quantize(notes, bpm, offset, division) {
    const unit = 60 / bpm / Math.max(1, division);
    const snap = (t) => rnd((t - offset) / unit) * unit + offset;
    const out = notes.map((n) => {
      const s = Math.max(snap(n.start_seconds), 0);
      const e = Math.max(snap(n.start_seconds + n.duration_seconds), s + unit);
      return { ...n, start_seconds: rnd(s * 1e6) / 1e6, duration_seconds: rnd((e - s) * 1e6) / 1e6 };
    });
    for (let i = 0; i + 1 < out.length; i++) {
      const next = out[i + 1].start_seconds;
      if (out[i].start_seconds + out[i].duration_seconds > next && next > out[i].start_seconds) out[i].duration_seconds = rnd((next - out[i].start_seconds) * 1e6) / 1e6;
    }
    return out;
  }
  const MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  function estimateKey(notes) {
    const h = new Array(12).fill(0);
    for (const n of notes) h[n.pitch % 12] += n.duration_seconds;
    if (h.filter((v) => v > 0).length < 3) return null;
    const corr = (a, b) => {
      const ma = a.reduce((x, y) => x + y, 0) / 12, mb = b.reduce((x, y) => x + y, 0) / 12;
      let num = 0, da = 0, dbb = 0;
      for (let i = 0; i < 12; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; dbb += (b[i] - mb) ** 2; }
      return da === 0 || dbb === 0 ? 0 : num / Math.sqrt(da * dbb);
    };
    const all = [];
    for (let t = 0; t < 12; t++)
      for (const [major, prof] of [[true, MAJOR], [false, MINOR]]) {
        const rot = Array.from({ length: 12 }, (_, i) => prof[(i + 12 - t) % 12]);
        all.push([t, major, rnd(corr(h, rot) * 1000) / 1000]);
      }
    all.sort((a, b) => b[2] - a[2] || a[0] - b[0] || (b[1] ? 1 : 0) - (a[1] ? 1 : 0));
    const [t, m, r] = all[0];
    const margin = r - all[1][2];
    const confidence = rnd((Math.max(r, 0) * 0.6 + Math.min(margin * 4, 0.4)) * 100) / 100;
    return { tonic: t, major: m, confidence, alternatives: all.slice(1, 4) };
  }

  function transcribe16k(x) {
    const warnings = validate(x);
    const track = pyin(x);
    const rms = frameRmsDb(x);
    const n = Math.min(track.voiced.length, rms.length);
    const quiet = [], loud = [];
    for (let i = 0; i < n; i++) (track.voiced[i] ? loud : quiet).push(rms[i]);
    if (quiet.length >= 30 && loud.length) {
      quiet.sort((a, b) => a - b);
      loud.sort((a, b) => a - b);
      if (loud[Math.floor(loud.length / 2)] - quiet[Math.floor(quiet.length / 2)] < 15) warnings.push("יש רעש רקע חזק יחסית לקול — התווים עלולים להיות פחות מדויקים");
    }
    const { notes, tuning_cents } = segment(track, rms);
    if (!notes.length) throw "לא זוהו תווים בהקלטה. נסה לזמזם ברור יותר, צליל אחרי צליל";
    const low = notes.filter((n) => n.confidence < 0.5).length;
    if (low * 3 > notes.length) warnings.push(`${low} מתוך ${notes.length} התווים זוהו בביטחון נמוך — כדאי לבדוק אותם (מסומנים בצבע חלש)`);
    if (Math.abs(tuning_cents) >= 40) warnings.push("הזמזום היה בדיוק בין שני צלילים — אם המנגינה נשמעת גבוהה או נמוכה בחצי טון, אפשר להזיז אותה בכפתורי הסולם");
    return { notes, track, tuning_cents, tempo: estimateTempo(notes), key: estimateKey(notes), warnings };
  }
  function transcribe(samples, channels, sampleRate) {
    return transcribe16k(toAnalysisRate(samples, channels, sampleRate));
  }

  // ---- vocal range (port of vocal.rs) ----
  const FSV = HOP / SR;
  function analyzeTrack(track) {
    const st = [];
    for (let i = 1; i < track.f0.length - 1; i++) {
      const a = midiOf(track, i - 1), m = midiOf(track, i), b = midiOf(track, i + 1);
      if (a !== null && m !== null && b !== null && Math.abs(a - m) <= 0.3 && Math.abs(b - m) <= 0.3) st.push([i, m]);
    }
    if (st.length < 30) return null;
    const held = [];
    let i = 0;
    while (i < st.length) {
      let j = i + 1;
      while (j < st.length && st[j][0] === st[j - 1][0] + 1 && Math.abs(st[j][1] - st[i][1]) <= 0.5) j++;
      if (j - i >= 15) { const seg = st.slice(i, j).map((x) => x[1]).sort((x, y) => x - y); held.push(seg[Math.floor(seg.length / 2)]); }
      i = j;
    }
    const all = st.map((x) => x[1]).sort((x, y) => x - y);
    const pct = (p) => all[rnd((all.length - 1) * p)];
    const lo = held.length ? Math.min(...held) : pct(0.02), hi = held.length ? Math.max(...held) : pct(0.98);
    const wob = [];
    for (let k = 1; k < st.length; k++) if (st[k][0] === st[k - 1][0] + 1) wob.push(Math.abs(st[k][1] - st[k - 1][1]) * 100);
    const stability = wob.length ? wob.reduce((x, y) => x + y, 0) / wob.length : 0;
    const voiced = track.voiced.filter(Boolean).length * FSV;
    const spanOk = Math.min((hi - lo) / 12, 1);
    const confidence = (Math.min(voiced / 20, 1) * 0.6 + spanOk * 0.4) * (1 - Math.min(stability / 60, 0.5));
    const r = (x) => Math.min(127, Math.max(0, rnd(x)));
    return { lowest: r(lo), highest: r(hi), comfort_low: r(pct(0.1)), comfort_high: r(pct(0.9)), center: r(pct(0.5)), stability_cents: rnd(stability * 10) / 10, voiced_seconds: rnd(voiced * 10) / 10, confidence: rnd(confidence * 100) / 100 };
  }
  function analyzeRange(samples, channels, sampleRate) {
    const x = toAnalysisRate(samples, channels, sampleRate);
    validate(x);
    const r = analyzeTrack(pyin(x));
    if (!r) throw "לא נקלט מספיק שירה יציבה כדי למדוד את הטווח. נסה לשיר כמה שורות בקול ברור";
    return r;
  }
  function recommendTransposition(melody, range) {
    const total = melody.reduce((x, m) => x + m[1], 0);
    if (!melody.length || total <= 0) return null;
    const fit = (s) => melody.filter(([p]) => p + s >= range.comfort_low && p + s <= range.comfort_high).reduce((x, m) => x + m[1], 0) / total;
    const outside = (s) => melody.filter(([p]) => !(p + s >= range.lowest && p + s <= range.highest)).reduce((x, m) => x + m[1], 0) / total;
    const centre = melody.reduce((x, [p, d]) => x + p * d, 0) / total;
    const scored = [];
    for (let s = -12; s <= 12; s++) scored.push([s, fit(s) * 3 - outside(s) * 4 - Math.abs(centre + s - range.center) * 0.08 - Math.abs(s) * 0.03, fit(s)]);
    scored.sort((a, b) => b[1] - a[1] || Math.abs(a[0]) - Math.abs(b[0]));
    return { semitones: scored[0][0], fit_before: rnd(fit(0) * 100) / 100 + 0, fit_after: rnd(scored[0][2] * 100) / 100, alternatives: scored.slice(1, 4).map((x) => [x[0], rnd(x[2] * 100) / 100]) };
  }

  // One imitation attempt of the guided range check (spec addendum 25.10 part 8): what was sung, how close to
  // the demonstrated tone (in any octave), how steady, and how long it was held.
  function analyzeAttempt(samples, channels, sampleRate, target) {
    const x = toAnalysisRate(samples, channels, sampleRate);
    if (x.length < SR * 0.3) return { voiced_seconds: 0, sung: null, cents_off: null, stability_cents: null, held_seconds: 0 };
    const tr = pyin(x), m = [];
    for (let i = 0; i < tr.f0.length; i++) { const v = midiOf(tr, i); if (v !== null) m.push([i, v]); }
    const voiced = m.length * FSV;
    if (m.length < 8) return { voiced_seconds: rnd(voiced * 10) / 10, sung: null, cents_off: null, stability_cents: null, held_seconds: 0 };
    const sorted = m.map((q) => q[1]).sort((a, b) => a - b), med = sorted[Math.floor(sorted.length / 2)];
    // the held part: frames within half a semitone of the median, longest consecutive run
    let best = 0, run = 0, prevI = -2;
    for (const [i, v] of m) { if (Math.abs(v - med) <= 0.5) { run = i === prevI + 1 ? run + 1 : 1; prevI = i; best = Math.max(best, run); } }
    const near = m.filter((q) => Math.abs(q[1] - med) <= 0.5).map((q) => q[1]);
    const mean = near.reduce((a, b) => a + b, 0) / near.length;
    const sd = Math.sqrt(near.reduce((a, b) => a + (b - mean) * (b - mean), 0) / near.length) * 100;
    let off = ((med - target) % 12 + 12) % 12; if (off > 6) off -= 12; // octave does not matter for "did you match it"
    return { voiced_seconds: rnd(voiced * 10) / 10, sung: Math.round(med * 100) / 100, cents_off: rnd(off * 100), stability_cents: rnd(sd), held_seconds: rnd(best * FSV * 10) / 10 };
  }
  // frame-by-frame pitch (MIDI, null when unvoiced) for the singing coach
  function pitchTrack(samples, channels, sampleRate) {
    const x = toAnalysisRate(samples, channels, sampleRate), tr = pyin(x), midi = [];
    for (let i = 0; i < tr.f0.length; i++) midi.push(midiOf(tr, i));
    return { hop_sec: FSV, midi };
  }
  const api = { pitchTrack, analyzeAttempt, analyzeRange, recommendTransposition, transcribe, transcribe16k, toAnalysisRate, quantize, estimateTempo, estimateKey, SR };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HUMTranscribe = api;
})(typeof window !== "undefined" ? window : globalThis);

;
// HUM built-in instrument sounds — browser port of engine/audio-core/src/synth.rs (same patches,
// same envelopes), used by the test lab to play notes through Web Audio.
(function (root) {
  "use strict";
  const RELEASE = { epiano: 0.35, soft_lead: 0.12, flute: 0.08, strings: 0.35, piano: 0.25, bass: 0.08, guitar: 0.1, drums: 1.6 };
  function noise(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s / 4294967295) * 2 - 1; }; }
  function drumHit(L, R, n, sr, seed) {
    const vel = Math.pow(n.velocity / 127, 1.2), rnd = noise(seed), TAU = 2 * Math.PI, p = n.pitch;
    const len = { 36: 0.45, 38: 0.3, 37: 0.08, 42: 0.09, 70: 0.09, 46: 0.45, 49: 1.5, 51: 0.9, 45: 0.45, 47: 0.45, 50: 0.45, 60: 0.5, 61: 0.12, 62: 0.12 }[p] || 0.2;
    const total = Math.floor(len * sr);
    let lp = 0, lp2 = 0, ph = 0;
    for (let i = 0; i < total; i++) {
      const idx = n.start + i;
      if (idx >= L.length) break;
      const t = i / sr, w = rnd();
      let v;
      if (p === 36) { const f = 45 + 105 * Math.exp(-t / 0.035); ph += f / sr; v = Math.sin(TAU * ph) * Math.exp(-t / 0.22) * 1.1 + w * 0.15 * Math.exp(-t / 0.004); }
      else if (p === 38) { lp += 0.35 * (w - lp); ph += 190 / sr; v = (w - lp) * 0.7 * Math.exp(-t / 0.11) + Math.sin(TAU * ph) * 0.45 * Math.exp(-t / 0.06); }
      else if (p === 37) { ph += 820 / sr; v = (Math.sin(TAU * ph) * 0.6 + w * 0.4) * Math.exp(-t / 0.018); }
      else if (p === 42 || p === 46 || p === 70) { lp += 0.6 * (w - lp); const d = p === 46 ? 0.18 : p === 70 ? 0.035 : 0.03; const a = p === 70 ? Math.min(t / 0.01, 1) : 1; v = (w - lp) * 0.45 * a * Math.exp(-t / d); }
      else if (p === 49 || p === 51) { lp += 0.5 * (w - lp); ph += 3150 / sr; const bell = p === 51 ? Math.sin(TAU * ph) * 0.12 * Math.exp(-t / 0.3) : 0; v = (w - lp) * 0.4 * Math.exp(-t / (p === 49 ? 0.55 : 0.35)) + bell; }
      else if (p === 45 || p === 47 || p === 50) { const base = p === 45 ? 98 : p === 47 ? 140 : 196; ph += (base * (1 + 0.5 * Math.exp(-t / 0.03))) / sr; v = Math.sin(TAU * ph) * 0.8 * Math.exp(-t / 0.2); }
      else if (p === 60) { ph += (92 * (1 + 0.4 * Math.exp(-t / 0.02))) / sr; lp2 += 0.05 * (w - lp2); v = Math.sin(TAU * ph) * 0.9 * Math.exp(-t / 0.25) + lp2 * 0.3 * Math.exp(-t / 0.03); }
      else if (p === 61 || p === 62) { ph += (p === 61 ? 930 : 700) / sr; lp += 0.5 * (w - lp); v = (Math.sin(TAU * ph) * 0.5 + (w - lp) * 0.5) * Math.exp(-t / (p === 61 ? 0.035 : 0.025)) * (p === 62 ? 0.6 : 1); }
      else v = w * 0.3 * Math.exp(-t / 0.05);
      const sv = v * 0.32 * vel;
      L[idx] += sv; R[idx] += sv;
    }
  }
  function pluck(L, R, n, sr, seed) {
    const f = hz(n.pitch), vel = Math.pow(n.velocity / 127, 1.3), period = Math.max(sr / f, 2);
    const len = Math.max(Math.floor(period - 0.6), 2), d = period - 0.5 - len, c = (1 - d) / (1 + d);
    const buf = new Float64Array(len + 2), rnd = noise(seed);
    let lp = 0;
    const bright = 0.3 + 0.6 * vel;
    for (let i = 0; i < buf.length; i++) { lp += bright * (rnd() - lp); buf[i] = lp; }
    const rel = Math.floor(0.1 * sr), total = n.len + rel;
    let i0 = 0, ax = 0, ay = 0;
    for (let i = 0; i < total; i++) {
      const idx = n.start + i;
      if (idx >= L.length) break;
      const a = buf[i0], b = buf[(i0 + 1) % len];
      const held = i < n.len;
      const nv = (a + b) * (held ? 0.4985 : 0.47);
      const y = c * nv + ax - c * ay; ax = nv; ay = y;
      buf[i0] = y; i0 = (i0 + 1) % len;
      const fade = held ? 1 : Math.pow(Math.max(1 - (i - n.len) / Math.max(rel, 1), 0), 2);
      const sv = a * 0.5 * vel * fade;
      L[idx] += sv; R[idx] += sv;
    }
  }
  const hz = (p) => 440 * Math.pow(2, (p - 69) / 12);
  function blep(t, dt) {
    if (t < dt) { const x = t / dt; return x + x - x * x - 1; }
    if (t > 1 - dt) { const x = (t - 1) / dt; return x * x + x + x + 1; }
    return 0;
  }
  function saw(ph) { return { ph }; }
  function sawNext(s, f, sr) {
    const dt = f / sr;
    const v = 2 * s.ph - 1 - blep(s.ph, dt);
    s.ph += dt; if (s.ph >= 1) s.ph -= 1;
    return v;
  }
  function renderNote(L, R, n, patch, sr, seed) {
    if (patch === "drums") return drumHit(L, R, n, sr, seed);
    if (patch === "guitar") return pluck(L, R, n, sr, seed);
    const f = hz(n.pitch), vel = Math.pow(n.velocity / 127, 1.4);
    const rel = Math.floor(RELEASE[patch] * sr), total = n.len + rel, frames = L.length;
    if (n.start >= frames) return;
    const saws = [saw(0), saw(0.33), saw(0.71)];
    let lp = 0, lp2 = 0, cph = 0, mph = 0, noise = seed >>> 0 || 1, nlp = 0;
    const gain = 0.22 * vel, TAU = 2 * Math.PI;
    for (let i = 0; i < total; i++) {
      const idx = n.start + i;
      if (idx >= frames) break;
      const t = i / sr, held = i < n.len;
      const relEnv = held ? 1 : Math.pow(Math.max(0, 1 - (i - n.len) / Math.max(rel, 1)), 2);
      let v;
      if (patch === "epiano") {
        const a = Math.min(t / 0.004, 1);
        const d = Math.exp(-t / (0.9 + 200 / f)) * 0.75 + 0.25 * Math.exp(-t / 3);
        const index = 1.6 * Math.sqrt(vel) * Math.exp(-t / 0.25) + 0.15;
        mph += f / sr; cph += f / sr;
        const m = Math.sin(TAU * mph) * index;
        const tine = Math.sin(TAU * cph * 14) * 0.04 * Math.exp(-t / 0.03);
        v = (Math.sin(TAU * cph + m) + tine) * a * d;
      } else if (patch === "soft_lead") {
        const a = Math.min(t / 0.015, 1);
        const s = 0.5 * (sawNext(saws[0], f * 1.003, sr) + sawNext(saws[1], f * 0.997, sr));
        const cutoff = Math.min(f * 4 + 1200 * Math.exp(-t / 0.4), 9000), k = 1 - Math.exp((-TAU * cutoff) / sr);
        lp += k * (s - lp); lp2 += k * (lp - lp2);
        v = lp2 * a * (0.85 + 0.15 * Math.exp(-t / 0.5));
      } else if (patch === "piano") {
        const a = Math.min(t / 0.002, 1), tau0 = Math.min(2.8 * Math.sqrt(261.6 / f), 6), bright = 0.5 + 0.5 * vel;
        let acc = 0;
        for (let k = 1; k <= 8; k++) {
          const fk = k * f * Math.sqrt(1 + 0.0004 * k * k);
          if (fk > sr * 0.45) break;
          acc += (Math.pow(bright, k - 1) / Math.pow(k, 1.2)) * Math.sin(TAU * fk * t) * Math.exp((-t * (1 + 0.7 * k)) / tau0);
        }
        v = acc * a * 0.8;
      } else if (patch === "bass") {
        const a = Math.min(t / 0.005, 1);
        cph += f / sr;
        const ph = TAU * cph, sw = sawNext(saws[0], f, sr);
        const k = 1 - Math.exp((-TAU * (f * 3 + 600 * Math.exp(-t / 0.08))) / sr);
        lp += k * (sw - lp);
        v = (Math.sin(ph) * 0.9 + 0.3 * Math.sin(2 * ph) + lp * 0.5) * a * (0.6 + 0.4 * Math.exp(-t / 0.9));
      } else if (patch === "flute") {
        const a = Math.min(t / 0.06, 1);
        const vib = 1 + 0.004 * Math.sin(TAU * 5 * t) * Math.min(t / 0.4, 1);
        cph += (f * vib) / sr;
        const ph = TAU * cph;
        noise ^= noise << 13; noise >>>= 0; noise ^= noise >>> 17; noise ^= noise << 5; noise >>>= 0;
        nlp += 0.08 * (noise / 4294967295 - 0.5 - nlp);
        v = (Math.sin(ph) + 0.18 * Math.sin(2 * ph) + 0.06 * Math.sin(3 * ph) + nlp * (0.6 + 1.2 * Math.exp(-t / 0.08))) * a;
      } else {
        const a = Math.min(t / 0.18, 1);
        const s = (sawNext(saws[0], f * 1.004, sr) + sawNext(saws[1], f * 0.996, sr) + sawNext(saws[2], f, sr)) / 3;
        const cutoff = Math.min(f * 3 + 1500, 7000), k = 1 - Math.exp((-TAU * cutoff) / sr);
        lp += k * (s - lp); lp2 += k * (lp - lp2);
        v = lp2 * a;
      }
      const s = v * gain * relEnv;
      L[idx] += s; R[idx] += s;
    }
  }
  /** notes: [{start, len, pitch, velocity}] in samples → {left, right} Float32Arrays incl. release tails. */
  function render(notes, patch, sr) {
    const rel = Math.floor(RELEASE[patch] * sr);
    let end = 0;
    for (const n of notes) end = Math.max(end, n.start + n.len + rel);
    const L = new Float32Array(end), R = new Float32Array(end);
    notes.forEach((n, i) => renderNote(L, R, n, patch, sr, Math.imul(0x9e3779b9, i + 1)));
    return { left: L, right: R };
  }
  const api = { render };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HUMSynth = api;
})(typeof window !== "undefined" ? window : globalThis);

;
// HUM composition engine — browser port of engine/compose (Rust): harmony, arrangement and
// original songs. Same rules, same random sequence; parity is checked by
// tests/parity/compose/check.sh against the Rust output.
(function (root) {
  "use strict";
  const PPQ = 960, BAR = PPQ * 4, STEP = PPQ / 4;
  const idiv = (a, b) => Math.trunc(a / b);
  const remE = (a, n) => ((a % n) + n) % n;
  const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
  const DRUM = { KICK: 36, SIDESTICK: 37, SNARE: 38, HAT: 42, HAT_OPEN: 46, TOM_LOW: 45, TOM_MID: 47, TOM_HIGH: 50, CRASH: 49, RIDE: 51, SHAKER: 70, DOUM: 60, TEK: 61, KA: 62 };
  const STYLES = ["pop", "rock", "mediterranean", "acoustic", "ballad"];
  const STYLE_HE = { pop: "פופ", rock: "רוק", mediterranean: "ים תיכוני", acoustic: "אקוסטי", ballad: "בלדה" };
  const DEFAULT_BPM = { pop: 110, rock: 124, mediterranean: 100, acoustic: 96, ballad: 72 };
  const ENERGY = { intro: 1, outro: 1, verse: 2, bridge: 2, chorus: 3, pre_chorus: 2, solo: 3, break: 1 };
  const SECTION_HE = { intro: "פתיחה", verse: "בית", chorus: "פזמון", bridge: "גשר", outro: "סיום", pre_chorus: "קדם־פזמון", solo: "סולו", break: "הפסקה" };
  const SALT = { drums: 0x1000, bass: 0x2000, guitar: 0x3000, piano: 0x4000, strings: 0x5000, winds: 0x6000 };

  class Rng {
    constructor(seed) { this.s = (seed >>> 0) === 0 ? 0x9e3779b9 : seed >>> 0; }
    nextU32() { let s = this.s; s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; this.s = s; return s; }
    nextF64() { return this.nextU32() / 4294967296; }
    below(n) { return Math.trunc(this.nextF64() * n); }
  }
  const mul = (a, b) => Math.imul(a >>> 0, b >>> 0) >>> 0;

  function pitchClasses(c) {
    const [a, b] = c.quality === "maj" ? [4, 7] : c.quality === "min" ? [3, 7] : [3, 6];
    return [c.root, (c.root + a) % 12, (c.root + b) % 12];
  }
  const contains = (c, pc) => pitchClasses(c).includes(pc % 12);
  const NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const chordName = (c) => NAMES[c.root % 12] + (c.quality === "min" ? "m" : c.quality === "dim" ? "dim" : "");
  function scale(key) {
    const steps = key.major ? [0, 2, 4, 5, 7, 9, 11] : [0, 2, 3, 5, 7, 8, 10];
    return steps.map((s) => (key.tonic + s) % 12);
  }
  const inScale = (key, pc) => scale(key).includes(pc % 12);

  function barSpan(start, end) {
    const a = idiv(Math.max(start, 0), BAR) * BAR;
    const b = idiv(Math.max(end, a + 1) + BAR - 1, BAR) * BAR;
    return [a, b];
  }
  function sectionsAround(ms, me, introBars, outroBars) {
    const [a, b] = barSpan(ms, me);
    const out = [];
    const is = Math.max(a - introBars * BAR, 0);
    if (a > is) out.push({ kind: "intro", start: is, dur: a - is });
    const bars = (b - a) / BAR;
    if (bars >= 8) {
      const half = idiv(bars, 2) * BAR;
      out.push({ kind: "verse", start: a, dur: half });
      out.push({ kind: "chorus", start: a + half, dur: b - a - half });
    } else out.push({ kind: "verse", start: a, dur: b - a });
    out.push({ kind: "outro", start: b, dur: outroBars * BAR });
    return out;
  }

  // ---------------- harmony ----------------
  function candidates(key, style) {
    const s = scale(key), v = [];
    if (key.major) {
      for (const [deg, q] of [[0, "maj"], [1, "min"], [2, "min"], [3, "maj"], [4, "maj"], [5, "min"]]) v.push([deg, s[deg], q]);
      if (style === "rock") v.push([10, (key.tonic + 10) % 12, "maj"]);
    } else {
      for (const [deg, q] of [[0, "min"], [2, "maj"], [3, "min"], [5, "maj"], [6, "maj"]]) v.push([deg, s[deg], q]);
      v.push([4, s[4], "maj"]);
      if (style === "mediterranean") v.push([8, (key.tonic + 1) % 12, "maj"]);
      else v.push([9, s[4], "min"]);
    }
    return v;
  }
  const GOOD_MAJ = [[0, 3, 0.6], [0, 4, 0.6], [0, 5, 0.5], [3, 4, 0.7], [3, 0, 0.5], [4, 0, 0.9], [4, 5, 0.5], [5, 3, 0.6], [5, 1, 0.4], [1, 4, 0.7], [2, 5, 0.4], [10, 3, 0.5], [0, 10, 0.4], [10, 0, 0.5]];
  const GOOD_MIN = [[0, 3, 0.6], [0, 5, 0.6], [0, 6, 0.5], [3, 4, 0.7], [4, 0, 0.9], [5, 6, 0.5], [6, 2, 0.5], [2, 5, 0.4], [5, 4, 0.6], [6, 0, 0.5], [8, 0, 0.7], [0, 8, 0.3], [9, 0, 0.4], [3, 0, 0.4]];
  function transition(a, b, major, style) {
    if (a === b) return -0.35;
    const g = (major ? GOOD_MAJ : GOOD_MIN).find(([x, y]) => x === a && y === b);
    const base = g ? g[2] : 0;
    let bonus = 0;
    if (style === "pop" && ((a === 5 && b === 3) || (a === 4 && b === 5))) bonus = 0.2;
    else if (style === "ballad" && ((a === 0 && b === 5) || (a === 5 && b === 3))) bonus = 0.2;
    else if (style === "mediterranean" && ((a === 4 && b === 0) || (a === 8 && b === 0))) bonus = 0.3;
    return base + bonus;
  }
  function harmonize(melody, key, from, to, opt) {
    const span = opt.chords_per_bar >= 2 ? BAR / 2 : BAR;
    const [a, b] = barSpan(from, to);
    const n = Math.max(idiv(b - a, span), 1);
    const cands = candidates(key, opt.style);
    const rng = new Rng((0x5eed0000 ^ mul(opt.variation, 0x9e3779b9)) >>> 0);
    const noise = [];
    for (let i = 0; i < n * cands.length; i++) noise.push(opt.variation === 0 ? 0 : (rng.nextF64() - 0.5) * 0.5);
    const emit = [], has = [];
    for (let i = 0; i < n; i++) {
      const s0 = a + i * span, s1 = s0 + span, row = [];
      has.push(false);
      cands.forEach(([, root, q], ci) => {
        const ch = { start: s0, dur: span, root, quality: q };
        let sc = 0;
        for (const nt of melody) {
          const o = Math.min(nt.start + nt.dur, s1) - Math.max(nt.start, s0);
          if (o <= 0) continue;
          has[i] = true;
          const bp = remE(nt.start - s0, BAR);
          const strong = nt.start >= s0 && (bp === 0 || bp === BAR / 2) ? 1.6 : 1.0;
          const w = (o / PPQ) * strong, pc = nt.pitch % 12;
          if (contains(ch, pc)) sc += w * (pc === root ? 1.2 : 1.0);
          else if (inScale(key, pc)) sc -= 0.35 * w;
          else sc -= 0.8 * w;
        }
        row.push(sc + noise[i * cands.length + ci]);
      });
      emit.push(row);
    }
    let tonic = cands.findIndex((c) => c[0] === 0); if (tonic < 0) tonic = 0;
    let dominant = cands.findIndex((c) => c[0] === 4 && c[2] === "maj"); if (dominant < 0) dominant = tonic;
    emit[0][tonic] += 1.0;
    emit[n - 1][tonic] += 1.5;
    if (n >= 3) emit[n - 2][dominant] += 0.6;
    for (let i = 0; i < n; i++) if (!has[i]) emit[i][tonic] += 0.3;
    const k = cands.length;
    let score = emit[0].slice();
    const back = Array.from({ length: n }, () => new Array(k).fill(0));
    for (let i = 1; i < n; i++) {
      const next = new Array(k).fill(-Infinity);
      for (let j = 0; j < k; j++) {
        for (let p = 0; p < k; p++) {
          const v = score[p] + transition(cands[p][0], cands[j][0], key.major, opt.style);
          if (v > next[j] + 1e-12) { next[j] = v; back[i][j] = p; }
        }
        next[j] += emit[i][j];
      }
      score = next;
    }
    let best = 0;
    for (let j = 1; j < k; j++) if (score[j] > score[best] + 1e-12) best = j;
    const path = new Array(n).fill(0);
    path[n - 1] = best;
    for (let i = n - 1; i >= 1; i--) path[i - 1] = back[i][path[i]];
    const out = [];
    path.forEach((ci, i) => {
      const [, root, q] = cands[ci];
      const s0 = a + i * span;
      const last = out[out.length - 1];
      if (last && last.root === root && last.quality === q && last.dur < BAR * 2) { last.dur += span; return; }
      out.push({ start: s0, dur: span, root, quality: q });
    });
    return out;
  }
  function melodicFit(melody, chords) {
    let fit = 0, total = 0;
    for (const n of melody) for (const c of chords) {
      const o = Math.min(n.start + n.dur, c.start + c.dur) - Math.max(n.start, c.start);
      if (o > 0) { total += o; if (contains(c, n.pitch % 12)) fit += o; }
    }
    return total === 0 ? 1 : fit / total;
  }
  function extendOver(chords, sections, key) {
    const out = [];
    const tq = key.major ? "maj" : "min";
    const dom = { start: 0, dur: BAR, root: (key.tonic + 7) % 12, quality: "maj" };
    for (const s of sections) {
      if (chords.some((c) => c.start < s.start + s.dur && c.start + c.dur > s.start)) continue;
      const bars = Math.max(idiv(s.dur, BAR), 1);
      for (let k = 0; k < bars; k++) {
        const st = s.start + k * BAR, last = k === bars - 1;
        let ch;
        if (s.kind === "intro" && last && bars > 1) ch = { ...dom, start: st };
        else if (s.kind === "intro" && bars > 1 && k % 2 === 1) ch = chords.length ? { ...chords[0], start: st, dur: BAR } : { start: st, dur: BAR, root: key.tonic, quality: tq };
        else ch = { start: st, dur: BAR, root: key.tonic, quality: tq };
        out.push(ch);
      }
    }
    out.push(...chords.map((c) => ({ ...c })));
    out.sort((x, y) => x.start - y.start);
    return out;
  }

  function inScaleFraction(melody, key) {
    const total = melody.reduce((a, n) => a + n.dur, 0);
    if (total === 0) return 1;
    return melody.filter((n) => inScale(key, n.pitch % 12)).reduce((a, n) => a + n.dur, 0) / total;
  }
  const KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  function estimateKey(melody) {
    if (melody.length < 3) return null;
    const h = new Array(12).fill(0);
    for (const n of melody) h[n.pitch % 12] += n.dur * (remE(n.start, BAR) === 0 ? 1.5 : 1.0);
    const meanH = h.reduce((a, b) => a + b, 0) / 12;
    const first = melody[0].pitch % 12, last = melody[melody.length - 1].pitch % 12;
    let best = null, bestScore = 0, second = -Infinity;
    for (let t = 0; t < 12; t++) for (const major of [true, false]) {
      const key = { tonic: t, major }, prof = major ? KK_MAJ : KK_MIN;
      const rot = Array.from({ length: 12 }, (_, i) => prof[(i + 12 - t) % 12]);
      const mr = rot.reduce((a, b) => a + b, 0) / 12;
      let num = 0, da = 0, db = 0;
      for (let i = 0; i < 12; i++) { num += (h[i] - meanH) * (rot[i] - mr); da += (h[i] - meanH) ** 2; db += (rot[i] - mr) ** 2; }
      const r = da === 0 ? 0 : num / Math.sqrt(da * db);
      const triad = [t, (t + (major ? 4 : 3)) % 12, (t + 7) % 12];
      const score = inScaleFraction(melody, key) * 2 + r + (last === t ? 0.2 : triad.includes(last) ? 0.1 : 0) + (triad.includes(first) ? 0.1 : 0);
      if (best !== null && score <= bestScore + 1e-12) second = Math.max(second, score);
      else { if (best !== null) second = Math.max(second, bestScore); best = key; bestScore = score; }
    }
    return [best, clamp((bestScore - second) * 2, 0, 1)];
  }

  // ---------------- arrangement ----------------
  const chordAt = (chords, t) => chords.find((c) => t >= c.start && t < c.start + c.dur) || [...chords].reverse().find((c) => c.start <= t) || null;
  const vel = (rng, base) => clamp(base + rng.below(9) - 4, 1, 127);
  function voice(ch, lo, hi, size, prev) {
    const pcs = pitchClasses(ch), options = [];
    for (let r = 0; r < 3; r++) {
      let p = lo;
      while (remE(p, 12) !== pcs[r]) p++;
      const v = [p];
      let k = 1;
      while (v.length < size) {
        const pc = pcs[(r + k) % 3];
        let q = v[v.length - 1] + 1;
        while (remE(q, 12) !== pc) q++;
        v.push(q); k++;
      }
      if (v[v.length - 1] <= hi) options.push(v.slice());
      if (v.every((x) => x + 12 <= hi)) options.push(v.map((x) => x + 12));
    }
    if (!options.length) return pcs.map((pc) => lo + ((pc + 12 - (lo % 12)) % 12));
    const cost = (v) => {
      if (prev.length === v.length) return v.reduce((s, x, i) => s + Math.abs(x - prev[i]), 0);
      const c = Math.trunc(v.reduce((s, x) => s + x, 0) / v.length);
      return Math.abs(c - Math.trunc((lo + hi) / 2));
    };
    let best = null, bk = null;
    for (const v of options) { const key = [cost(v), v[0]]; if (!best || key[0] < bk[0] || (key[0] === bk[0] && key[1] < bk[1])) { best = v; bk = key; } }
    return best;
  }
  const bassPitch = (pc) => { const p = 36 + (pc % 12); return p > 43 ? p - 12 : p; };
  function bars(sections) {
    const v = [];
    for (const s of sections) { const n = Math.max(idiv(s.dur, BAR), 1); for (let k = 0; k < n; k++) v.push([s.start + k * BAR, s, k === n - 1]); }
    return v;
  }
  const sortNotes = (a) => a.sort((x, y) => x.start - y.start || x.pitch - y.pitch);

  function drums(sections, style, rng) {
    const D = DRUM, out = [];
    const hit = (t, p, v) => out.push({ pitch: p, start: t, dur: STEP, velocity: vel(rng, v) });
    for (const [bar, sec, last] of bars(sections)) {
      const e = ENERGY[sec.kind];
      if (bar === sec.start && e === 3 && style !== "mediterranean") hit(bar, D.CRASH, 100);
      const s = (i) => bar + i * STEP;
      if (style === "pop") {
        for (let i = 0; i < 16; i += 2) hit(s(i), e === 3 && i === 14 ? D.HAT_OPEN : D.HAT, (e === 3 ? 78 : 62) + (i % 4 === 0 ? 8 : 0));
        hit(s(0), D.KICK, 96); hit(s(8), D.KICK, 90);
        if (e >= 2) { hit(s(4), D.SNARE, 96); hit(s(12), D.SNARE, 98); if (e === 3 || rng.nextF64() < 0.4) hit(s(10), D.KICK, 80); }
      } else if (style === "rock") {
        for (let i = 0; i < 16; i += 2) hit(s(i), e === 3 ? D.RIDE : D.HAT, e >= 2 ? 88 : 70);
        hit(s(0), D.KICK, 110);
        if (e >= 2) { hit(s(6), D.KICK, 96); hit(s(8), D.KICK, 104); hit(s(4), D.SNARE, 112); hit(s(12), D.SNARE, 115); if (e === 3) hit(s(11), D.KICK, 92); }
        else hit(s(8), D.KICK, 100);
      } else if (style === "mediterranean") {
        const lvl = e === 1 ? 70 : 96;
        hit(s(0), D.DOUM, lvl + 10); hit(s(2), D.TEK, lvl); hit(s(6), D.TEK, lvl); hit(s(8), D.DOUM, lvl + 8); hit(s(12), D.TEK, lvl);
        if (e >= 2) { for (const i of [4, 10, 14]) hit(s(i), D.KA, 52); hit(s(0), D.KICK, 84); hit(s(8), D.KICK, 78); }
        if (e === 3) for (let i = 1; i < 16; i += 2) hit(s(i), D.SHAKER, 46);
      } else if (style === "acoustic") {
        const step = e === 3 ? 1 : 2;
        for (let i = 0; i < 16; i += step) hit(s(i), D.SHAKER, i % 4 === 0 ? 58 : 44);
        if (e >= 2) { hit(s(0), D.KICK, 78); hit(s(8), D.KICK, 72); hit(s(4), e === 3 ? D.SNARE : D.SIDESTICK, 70); hit(s(12), e === 3 ? D.SNARE : D.SIDESTICK, 72); }
      } else {
        if (e >= 2) {
          const hs = e === 3 ? 2 : 4;
          for (let i = 0; i < 16; i += hs) hit(s(i), D.HAT, 54);
          hit(s(0), D.KICK, 82);
          hit(s(8), e === 3 ? D.SNARE : D.SIDESTICK, e === 3 ? 88 : 66);
          if (e === 3) hit(s(10), D.KICK, 70);
        }
      }
      if (last && e >= 2 && (style === "pop" || style === "rock") && sec.kind !== "outro") {
        for (let i = out.length - 1; i >= 0; i--) { const n = out[i]; if (n.start >= s(12) && n.start < bar + BAR && n.pitch !== D.KICK) out.splice(i, 1); }
        for (const [i, p] of [[12, D.TOM_HIGH], [13, D.TOM_HIGH], [14, D.TOM_MID], [15, D.TOM_LOW]]) hit(s(i), p, 96);
      }
    }
    const lastS = sections[sections.length - 1];
    if (lastS && style !== "mediterranean") out.push({ pitch: D.CRASH, start: lastS.start + lastS.dur - BAR, dur: STEP, velocity: 90 });
    return sortNotes(out);
  }
  function bassPart(chords, sections, style, rng) {
    const out = [];
    for (const [bar, sec] of bars(sections)) {
      const e = ENERGY[sec.kind];
      let pat;
      if (e === 1) pat = [[0, 0, 16]];
      else if (style === "pop") pat = e === 2 ? [[0, 0, 3], [4, 0, 3], [8, 0, 3], [12, 0, 3]] : [[0, 0, 2], [2, 0, 1], [4, 0, 2], [6, 0, 1], [8, 0, 2], [10, 0, 1], [12, 0, 2], [14, 7, 2]];
      else if (style === "rock") pat = [[0, 0, 1], [2, 0, 1], [4, 0, 1], [6, 0, 1], [8, 0, 1], [10, 0, 1], [12, 0, 1], [14, 0, 1]];
      else if (style === "mediterranean") pat = [[0, 0, 3], [6, 7, 2], [8, 0, 3], [12, 12, 2]];
      else if (style === "acoustic") pat = e === 2 ? [[0, 0, 8], [8, 7, 8]] : [[0, 0, 6], [6, 0, 2], [8, 7, 6], [14, 0, 2]];
      else pat = e === 2 ? [[0, 0, 16]] : [[0, 0, 12], [12, 7, 4]];
      for (const [st, iv, len] of pat) {
        const t = bar + st * STEP, c = chordAt(chords, t);
        if (!c) continue;
        const r = bassPitch(c.root);
        const pitch = iv === 7 ? r + 7 : iv === 12 ? r + 12 : r;
        const base = style === "rock" ? 100 : style === "ballad" ? 80 : 92;
        out.push({ pitch, start: t, dur: len * STEP - STEP / 4, velocity: vel(rng, base + (st === 0 ? 6 : 0)) });
      }
    }
    return out;
  }
  function strum(out, v, t, dur, down, velocity) {
    const order = down ? v.slice() : v.slice().reverse().slice(0, 3);
    order.forEach((p, k) => { const off = k * 14; out.push({ pitch: p, start: t + off, dur: Math.max(dur - off, STEP / 2), velocity: Math.max(Math.max(velocity - k * 3, 0), 1) }); });
  }
  function guitar(chords, sections, style, rng) {
    const out = [];
    let prev = [];
    for (const [bar, sec] of bars(sections)) {
      const e = ENERGY[sec.kind];
      const c0 = chordAt(chords, bar);
      if (!c0) continue;
      if (style === "rock") {
        const r = 40 + ((c0.root + 12 - 4) % 12);
        if (e === 1) strum(out, [r, r + 7, r + 12], bar, BAR, true, vel(rng, 80));
        else for (let i = 0; i < 16; i += 2) {
          const t = bar + i * STEP, c = chordAt(chords, t) || c0, rr = 40 + ((c.root + 12 - 4) % 12);
          strum(out, [rr, rr + 7, rr + 12], t, e === 3 ? 2 * STEP : STEP, true, vel(rng, e === 3 ? 100 : 88));
        }
      } else if ((style === "pop" || style === "ballad") && e === 1) {
        const v = voice(c0, 52, 71, 4, prev);
        strum(out, v, bar, BAR, true, vel(rng, 70));
        prev = v;
      } else if (style === "pop") {
        const pat = e === 3 ? [[0, true], [2, false], [4, true], [6, false], [8, true], [10, false], [12, true], [14, false]] : [[0, true], [6, false], [8, true], [10, false], [12, true], [14, false]];
        pat.forEach(([st, down], k) => {
          const t = bar + st * STEP, c = chordAt(chords, t) || c0, v = voice(c, 52, 71, 4, prev);
          const next = k + 1 < pat.length ? pat[k + 1][0] : 16;
          strum(out, v, t, (next - st) * STEP, down, vel(rng, down ? 84 : 70));
          prev = v;
        });
      } else if (style === "mediterranean" || style === "ballad") {
        const step = style === "mediterranean" ? 1 : 2;
        let k = 0;
        for (let i = 0; i < 16; i += step) {
          const t = bar + i * STEP, c = chordAt(chords, t) || c0, v = voice(c, 52, 76, 4, prev);
          const idx = [0, 1, 2, 3, 2, 1][k % 6];
          out.push({ pitch: v[idx], start: t, dur: step * STEP * 2, velocity: vel(rng, e === 3 ? 80 : 68) });
          k++; prev = v;
        }
      } else {
        const pick = e === 1 ? [0, 2, 1, 2] : [0, 2, 3, 2];
        const step = e === 1 ? 4 : 2;
        let k = 0;
        for (let i = 0; i < 16; i += step) {
          const t = bar + i * STEP, c = chordAt(chords, t) || c0, v = voice(c, 48, 72, 4, prev);
          const bassNote = i === 8 ? 40 + ((c.root + 7 + 12 - 4) % 12) : 40 + ((c.root + 12 - 4) % 12);
          const p = k % 4 === 0 ? bassNote : v[Math.min(pick[k % 4], v.length - 1)];
          out.push({ pitch: p, start: t, dur: step * STEP * 2, velocity: vel(rng, k % 4 === 0 ? 80 : 66) });
          prev = v; k++;
        }
      }
    }
    return out;
  }
  function piano(chords, sections, style, melodyLow, rng) {
    const out = [];
    const hi = melodyLow === null ? 76 : clamp(Math.max(melodyLow - 1, 0), 64, 76);
    const lo = hi - 17;
    let prev = [];
    for (const [bar, sec] of bars(sections)) {
      const e = ENERGY[sec.kind];
      const c0 = chordAt(chords, bar);
      if (!c0) continue;
      const block = (t, len, base) => {
        const c = chordAt(chords, t) || c0, v = voice(c, lo, hi, 3, prev);
        for (const p of v) out.push({ pitch: p, start: t, dur: len - STEP / 4, velocity: vel(rng, base) });
        prev = v;
      };
      if ((style === "rock" || style === "acoustic") && e === 1) { /* rests */ }
      else if (style === "rock" || style === "acoustic") block(bar, BAR, style === "rock" ? 66 : 58);
      else if (style === "pop" && e === 1) block(bar, BAR, 62);
      else if (style === "pop" && e === 2) for (const i of [0, 8]) block(bar + i * STEP, 8 * STEP, 70);
      else if (style === "pop") for (const i of [0, 6, 10]) block(bar + i * STEP, i === 0 ? 6 * STEP : 4 * STEP, 78);
      else if (style === "mediterranean" && e === 1) block(bar, BAR, 60);
      else if (style === "mediterranean") for (const i of [4, 12]) block(bar + i * STEP, 2 * STEP, 72);
      else if (e === 1) block(bar, BAR, 60);
      else {
        out.push({ pitch: bassPitch(c0.root) + 12, start: bar, dur: BAR - STEP, velocity: vel(rng, 70) });
        let k = 0;
        for (let i = 0; i < 16; i += 2) {
          const t = bar + i * STEP, ch = chordAt(chords, t) || c0, v = voice(ch, lo, hi, 3, prev);
          const idx = [0, 1, 2, 1][k % 4];
          out.push({ pitch: v[idx], start: t, dur: 3 * STEP, velocity: vel(rng, k === 0 ? 72 : 62) });
          prev = v; k++;
        }
      }
    }
    return out;
  }
  function strings(chords, sections, rng) {
    const out = [];
    let prev = [];
    for (const [bar, sec] of bars(sections)) {
      const e = ENERGY[sec.kind];
      if (e === 2 && sec.kind === "verse" && idiv(bar - sec.start, BAR) % 2 === 1) continue;
      const c = chordAt(chords, bar);
      if (!c) continue;
      const v = voice(c, 55, 79, e === 3 ? 4 : 3, prev);
      for (const p of v) out.push({ pitch: p, start: bar, dur: BAR - 20, velocity: vel(rng, e === 3 ? 72 : 56) });
      prev = v;
    }
    return out;
  }
  function flute(chords, sections, rng) {
    const out = [];
    let last = null;
    for (const [bar, sec] of bars(sections)) {
      if (sec.kind !== "chorus" && sec.kind !== "outro") continue;
      for (const half of [0, 8]) {
        const t = bar + half * STEP, c = chordAt(chords, t);
        if (!c) continue;
        const pcs = pitchClasses(c), target = last === null ? 79 : last;
        let best = [2147483647, 79];
        for (const oct of [72, 84]) for (const pc of [pcs[1], pcs[2], pcs[0]]) {
          const p = oct + pc;
          if (p < 74 || p > 91) continue;
          const d = Math.abs(p - target) + (p === last ? 3 : 0);
          if (d < best[0]) best = [d, p];
        }
        out.push({ pitch: best[1], start: t, dur: 8 * STEP - STEP / 2, velocity: vel(rng, 70) });
        last = best[1];
      }
    }
    return out;
  }
  function arrangePart(role, chords, sections, style, melody, variation) {
    const rng = new Rng((SALT[role] ^ mul(variation, 0x2545f491) ^ 0xa5a5) >>> 0);
    let low = null;
    if (melody.length) { const ps = melody.map((n) => n.pitch).sort((a, b) => a - b); low = ps[idiv(ps.length, 5)]; }
    let notes;
    if (role === "drums") notes = drums(sections, style, rng);
    else if (role === "bass") notes = bassPart(chords, sections, style, rng);
    else if (role === "guitar") notes = guitar(chords, sections, style, rng);
    else if (role === "piano") notes = piano(chords, sections, style, low, rng);
    else if (role === "strings") notes = strings(chords, sections, rng);
    else notes = flute(chords, sections, rng);
    return sortNotes(notes.filter((n) => n.dur > 0 && n.start >= 0));
  }

  // ---------------- composition ----------------
  function degree(key, d) {
    const M = { I: [0, "maj"], ii: [2, "min"], iii: [4, "min"], IV: [5, "maj"], V: [7, "maj"], vi: [9, "min"], bVII: [10, "maj"] };
    const m = { i: [0, "min"], III: [3, "maj"], iv: [5, "min"], V: [7, "maj"], v: [7, "min"], VI: [8, "maj"], VII: [10, "maj"], bII: [1, "maj"] };
    const e = (key.major ? M : m)[d] || [0, key.major ? "maj" : "min"];
    return [(key.tonic + e[0]) % 12, e[1]];
  }
  function progression(style, major, kind) {
    if (kind === "intro" || kind === "outro" || kind === "break") return major ? ["I", "IV", "I", "V"] : ["i", "VI", "iv", "V"];
    if (kind === "pre_chorus") return major ? ["ii", "IV", "V", "V"] : ["iv", "VI", "V", "V"];
    const ch = kind === "chorus";
    const T = {
      pop: major ? (ch ? ["IV", "I", "V", "vi"] : ["I", "V", "vi", "IV"]) : ch ? ["VI", "VII", "i", "i"] : ["i", "VI", "III", "VII"],
      rock: major ? (ch ? ["I", "V", "vi", "IV"] : ["I", "bVII", "IV", "I"]) : ch ? ["i", "VII", "VI", "VII"] : ["i", "VI", "VII", "i"],
      mediterranean: major ? (ch ? ["IV", "V", "I", "I"] : ["I", "ii", "V", "I"]) : ch ? ["i", "VII", "VI", "V"] : ["i", "iv", "V", "i"],
      acoustic: major ? (ch ? ["vi", "IV", "I", "V"] : ["I", "IV", "vi", "V"]) : ch ? ["VI", "III", "VII", "i"] : ["i", "iv", "VI", "V"],
      ballad: major ? (ch ? ["IV", "V", "iii", "vi"] : ["I", "vi", "IV", "V"]) : ch ? ["VI", "VII", "III", "V"] : ["i", "VI", "iv", "V"],
    };
    return T[style];
  }
  const MOTIFS = {
    pop: [[[0, 2], [2, 2], [4, 4], [8, 2], [10, 2], [12, 4]], [[0, 4], [4, 2], [6, 2], [8, 8]]],
    rock: [[[0, 2], [2, 2], [4, 2], [6, 2], [8, 4], [12, 4]], [[0, 4], [4, 4], [8, 8]]],
    mediterranean: [[[0, 3], [3, 1], [4, 2], [6, 2], [8, 4], [12, 2], [14, 2]], [[0, 2], [2, 2], [4, 4], [8, 8]]],
    acoustic: [[[0, 2], [2, 4], [6, 2], [8, 4], [12, 4]], [[0, 4], [4, 4], [8, 8]]],
    ballad: [[[0, 6], [6, 2], [8, 8]], [[0, 4], [4, 4], [8, 8]]],
  };
  const FORM = [["intro", 4], ["verse", 8], ["chorus", 8], ["verse", 8], ["chorus", 8], ["outro", 4]];
  const chordAtStrict = (chords, t) => chords.find((c) => t >= c.start && t < c.start + c.dur) || null;
  function writeMelody(sections, chords, key, style, rng) {
    const sc = scale(key);
    const pool = [];
    for (let p = 55; p <= 84; p++) if (sc.includes(p % 12)) pool.push(p);
    let tonicMid = 60;
    for (let p = 60; p < 72; p++) if (p % 12 === key.tonic) { tonicMid = p; break; }
    const [mA, mB] = MOTIFS[style];
    const out = [];
    let prev = tonicMid;
    const memory = [];
    const minBy = (arr, f) => { let best = null, bk = null; for (const x of arr) { const k = f(x); if (bk === null || k < bk) { best = x; bk = k; } } return best; };
    for (const s of sections) {
      if (s.kind === "intro" || s.kind === "outro" || s.kind === "solo" || s.kind === "break") continue; // instrumental
      const mem = memory.find(([k]) => k === s.kind);
      if (mem) {
        const first = mem[1];
        const base = sections.find((x) => x.kind === s.kind).start;
        const off = s.start - base, n = first.length;
        first.forEach((nt, i) => {
          const c = { ...nt, start: nt.start + off };
          if (i >= Math.max(n - 2, 0) && rng.nextF64() < 0.5) {
            const ch = chordAtStrict(chords, c.start);
            if (ch) {
              const pcs = pitchClasses(ch);
              // Rust: min_by_key((diff == 0, diff)) over chord tones in the pool
              const p = minBy(pool.filter((p) => pcs.includes(p % 12)), (p) => (Math.abs(p - c.pitch) === 0 ? 1e6 : 0) + Math.abs(p - c.pitch));
              if (p !== null) c.pitch = p;
            }
          }
          out.push(c);
        });
        continue;
      }
      const lift = s.kind === "chorus" ? 4 : 0;
      const secNotes = [];
      const nbars = idiv(s.dur, BAR);
      for (let b = 0; b < nbars; b++) {
        const bar = s.start + b * BAR;
        const motif = b % 2 === 0 ? mA : mB;
        const phraseEnd = b % 4 === 3;
        motif.forEach(([st, len], k) => {
          const t = bar + st * (PPQ / 4);
          const lastOfBar = k === motif.length - 1;
          const ch = chordAtStrict(chords, t);
          const strong = st % 8 === 0;
          const target = tonicMid + lift + (b % 4 === 2 ? 3 : 0);
          let pitch;
          if (strong || (phraseEnd && lastOfBar)) {
            const pcs = ch ? pitchClasses(ch) : [key.tonic, key.tonic, key.tonic];
            const wantTonic = phraseEnd && lastOfBar && b === nbars - 1;
            const cand = pool.filter((p) => (wantTonic ? p % 12 === key.tonic : pcs.includes(p % 12)));
            const p = minBy(cand, (p) => Math.abs(p - prev) * 2 + Math.abs(p - target) + rng.below(3));
            pitch = p === null ? prev : p;
          } else {
            let idx = pool.indexOf(prev);
            if (idx < 0) idx = idiv(pool.length, 2);
            const dir = prev < target ? 1 : prev > target + 4 ? -1 : rng.nextF64() < 0.5 ? 1 : -1;
            const jump = rng.nextF64() < 0.25 ? 2 : 1;
            pitch = pool[clamp(idx + dir * jump, 0, pool.length - 1)];
          }
          const dur = phraseEnd && lastOfBar ? len * (PPQ / 4) : len * (PPQ / 4) - PPQ / 16;
          secNotes.push({ pitch, start: t, dur, velocity: 84 + rng.below(12) });
          prev = pitch;
        });
      }
      memory.push([s.kind, secNotes.slice()]);
      out.push(...secNotes);
    }
    return out;
  }
  function composeSong(req) {
    const rng = new Rng((0xc0ffee00 ^ mul(req.seed, 0x9e3779b9)) >>> 0);
    const major = req.key ? req.key.major : req.major !== null && req.major !== undefined ? req.major : !(req.style === "mediterranean" || req.style === "ballad");
    const key = req.key || { tonic: major ? [0, 7, 2, 5][rng.below(4)] : [9, 4, 2, 7][rng.below(4)], major };
    const bpm = req.bpm || DEFAULT_BPM[req.style];
    const sections = [];
    let t = 0;
    for (const [k, b] of req.form || FORM) { sections.push({ kind: k, start: t, dur: b * BAR }); t += b * BAR; }
    const chords = [];
    for (const s of sections) {
      const prog = progression(req.style, key.major, s.kind);
      const nb = s.dur / BAR;
      for (let b = 0; b < nb; b++) {
        let [root, q] = degree(key, prog[b % 4]);
        if (s.kind === "outro" && b === nb - 1) [root, q] = degree(key, key.major ? "I" : "i");
        chords.push({ start: s.start + b * BAR, dur: BAR, root, quality: q });
      }
    }
    const melody = writeMelody(sections, chords, key, req.style, rng);
    // key change for the last chorus and the ending, prepared by the new key's dominant (port of compose.rs)
    const lift = req.lift || 0;
    if (lift) {
      const last = sections.slice().reverse().find((x) => x.kind === "chorus");
      if (last) {
        const up = (pc) => (((pc + lift) % 12) + 12) % 12;
        for (const c of chords) if (c.start >= last.start) c.root = up(c.root);
        const pre = chords.find((c) => c.start + c.dur === last.start);
        if (pre) { pre.root = up((key.tonic + 7) % 12); pre.quality = "maj"; }
        for (const n of melody) if (n.start >= last.start) n.pitch = clamp(n.pitch + lift, 0, 127);
      }
    }
    const parts = req.roles.map((r) => [r, arrangePart(r, chords, sections, req.style, melody, req.seed)]);
    applyForm(parts, chords, sections, key, req.solo || null, req.seed);
    return { key, bpm, sections, chords, melody, parts };
  }
  // ---- solos and breaks (port of arrange.rs solo_line / apply_form; spec addendum 25.10 part 2) ----
  const ROLE_INDEX = { drums: 0, bass: 1, guitar: 2, piano: 3, strings: 4, winds: 5 };
  const SOLO_RANGE = { guitar: [55, 79], piano: [60, 84], winds: [67, 93], strings: [55, 86], bass: [40, 62] };
  function soloRole(parts, wanted) {
    const has = (r) => parts.some(([x]) => x === r);
    if (wanted && has(wanted) && SOLO_RANGE[wanted]) return wanted;
    return ["guitar", "piano", "winds", "strings"].find(has) || null;
  }
  function soloLine(role, chords, sec, key, seed) {
    const rg = SOLO_RANGE[role]; if (!rg) return [];
    const rng = new Rng((seed ^ 0x501050 ^ mul(ROLE_INDEX[role], 0x01010101)) >>> 0);
    const sc = scale(key), pool = [];
    for (let p = rg[0]; p <= rg[1]; p++) if (sc.includes(p % 12)) pool.push(p);
    if (!pool.length) return [];
    const bars = Math.max(idiv(sec.dur, BAR), 1), six = PPQ / 4, out = [];
    let prev = pool[idiv(pool.length, 3)];
    const minBy = (arr, f) => { let best = null, bk = null; for (const x of arr) { const k = f(x); if (bk === null || k < bk) { best = x; bk = k; } } return best; };
    for (let b = 0; b < bars; b++) {
      const bar = sec.start + b * BAR, prog = b / bars, lastBar = b === bars - 1;
      const step = lastBar ? 16 : prog < 1 / 3 ? 4 : prog < 2 / 3 ? 2 : 1;
      const target = pool[Math.trunc((pool.length - 1) * (0.3 + 0.55 * prog))];
      let pos = 0;
      while (pos < 16) {
        const t = bar + pos * six, len = step === 1 && pos >= 12 ? 4 : step;
        const ch = chords.find((c) => t >= c.start && t < c.start + c.dur) || null;
        let pitch;
        if (lastBar) { const p = minBy(pool.filter((p) => p % 12 === key.tonic), (p) => Math.abs(p - prev)); pitch = p === null ? prev : p; }
        else if (pos % 8 === 0) {
          const pcs = ch ? pitchClasses(ch) : [key.tonic, key.tonic, key.tonic];
          const p = minBy(pool.filter((p) => pcs.includes(p % 12)), (p) => Math.abs(p - prev) * 2 + Math.abs(p - target) + rng.below(3));
          pitch = p === null ? prev : p;
        } else {
          let idx = pool.indexOf(prev); if (idx < 0) idx = idiv(pool.length, 2);
          const dir = prev < target ? 1 : prev > target + 3 ? -1 : rng.nextF64() < 0.5 ? 1 : -1;
          const jump = rng.nextF64() < 0.2 ? 2 : 1;
          pitch = pool[clamp(idx + dir * jump, 0, pool.length - 1)];
        }
        const vel = Math.trunc(78 + 30 * prog) + rng.below(6);
        out.push({ pitch, start: t, dur: len * six - (len >= 4 ? six / 4 : 0), velocity: Math.min(vel, 120) });
        prev = pitch; pos += len;
      }
    }
    return out;
  }
  const byStartPitch = (a, b) => a.start - b.start || a.pitch - b.pitch;
  function applyForm(parts, chords, sections, key, wanted, seed) {
    const soloist = soloRole(parts, wanted), inside = (n, s) => n.start >= s.start && n.start < s.start + s.dur;
    for (const s of sections.filter((s) => s.kind === "solo")) {
      if (!soloist) break;
      const e = parts.find(([x]) => x === soloist);
      if (e) { e[1] = e[1].filter((n) => !inside(n, s)).concat(soloLine(soloist, chords, s, key, (seed ^ s.start) >>> 0)); e[1].sort(byStartPitch); }
    }
    for (const s of sections.filter((s) => s.kind === "break")) {
      const holder = ["piano", "strings"].find((r) => parts.some(([x]) => x === r));
      for (const e of parts) {
        e[1] = e[1].filter((n) => !inside(n, s));
        if (e[0] === holder) {
          const c = chords.find((c) => s.start >= c.start && s.start < c.start + c.dur);
          if (c) { for (const pc of pitchClasses(c)) e[1].push({ pitch: 60 + pc, start: s.start, dur: s.dur - PPQ / 4, velocity: 70 }); e[1].sort(byStartPitch); }
        }
      }
    }
  }

  // Starting mix of a new track (engine/compose/src/mix.rs, CRR-13 step 1): level, place, light EQ,
  // compression and reverb by role and style. Existing tracks are never touched.
  const r2 = (x) => Math.round(x * 100) / 100;
  function mixPreset(style, role) {
    const T = { lead: [0, 0, -2, 1, 1.5, 0.3, 0.2], melody: [0, 0, -2, 1, 1.5, 0.3, 0.2], vocals: [0, 0, -2, 1, 1.5, 0.3, 0.2], drums: [-3, 0, 1.5, 0, 1.5, 0.5, 0.12], bass: [-1, 0, 2, -1, -2, 0.45, 0], guitar: [-3, -0.35, -3, 0, 1.5, 0.2, 0.22], piano: [-3, 0.3, -2, 0, 1, 0.15, 0.25], strings: [-6, -0.2, -4, 0, 0, 0, 0.4], winds: [-5, 0.4, -4, 0, 1, 0.1, 0.3], flute: [-5, 0.4, -4, 0, 1, 0.1, 0.3] };
    let [g, pan, lo, mid, hi, comp, rev] = T[role] || [0, 0, 0, 0, 0, 0, 0];
    const [ck, rk] = { rock: [1.2, 0.7], ballad: [0.8, 1.3], acoustic: [0.8, 1.2], mediterranean: [1, 1.15], pop: [1, 1] }[style] || [1, 1];
    if (style === "rock" && role === "guitar") pan = -0.5;
    return { gain_db: g, pan, eq_low_db: lo, eq_mid_db: mid, eq_high_db: hi, comp: r2(Math.min(comp * ck, 1)), reverb: r2(Math.min(rev * rk, 1)) };
  }

  const api = { mixPreset, soloLine, applyForm, estimateKey, inScaleFraction, PPQ, BAR, STYLES, STYLE_HE, SECTION_HE, harmonize, extendOver, sectionsAround, arrangePart, composeSong, melodicFit, chordName, Rng };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HUMCompose = api;
})(typeof window !== "undefined" ? window : globalThis);

;
// HUM Web — Standard MIDI File reading and import planning.
// Line-by-line port of domain/project/src/smf.rs and apps/core/src/import.rs (plan_parts, key_of);
// tests/parity/midi checks that both give identical results.
(function (root) {
  "use strict";
  const MAX_FILE_BYTES = 4 * 1024 * 1024, MAX_NOTES = 200000, MAX_TRACKS = 256, MAX_IMPORT_TRACKS = 16;

  function reader(b) {
    const r = { b, i: 0 };
    r.u8 = () => { if (r.i >= r.b.length) throw "הקובץ קטוע"; return r.b[r.i++]; };
    r.take = (n) => { if (r.i + n > r.b.length) throw "הקובץ קטוע"; const s = r.b.subarray(r.i, r.i + n); r.i += n; return s; };
    r.be = (n) => { let v = 0; for (const x of r.take(n)) v = v * 256 + x; return v; };
    r.varlen = () => { let v = 0; for (let k = 0; k < 4; k++) { const x = r.u8(); v = v * 128 + (x & 0x7f); if (!(x & 0x80)) return v; } throw "מספר באורך לא תקין בקובץ"; };
    return r;
  }
  const isCtl = (c) => { const n = c.codePointAt(0); return n < 32 || (n >= 127 && n < 160); };

  function parse(bytes) {
    if (bytes.length > MAX_FILE_BYTES) throw "קובץ התווים גדול מדי (עד 4MB)";
    const r = reader(bytes);
    let magic;
    try { magic = r.take(4); } catch { throw "זה לא קובץ MIDI"; }
    if (String.fromCharCode(...magic) !== "MThd") throw "זה לא קובץ MIDI (חסרה כותרת MThd)";
    const hlen = r.be(4);
    if (hlen < 6) throw "כותרת קובץ MIDI לא תקינה";
    const format = r.be(2), ntracks = r.be(2), division = r.be(2);
    r.take(hlen - 6);
    if (format > 1) throw "קובצי MIDI מסוג 2 (תבניות נפרדות) לא נתמכים עדיין";
    if (division & 0x8000 || division === 0) throw "קובץ ה-MIDI משתמש בזמן מבוסס SMPTE — עדיין לא נתמך";
    const smf = { format, division, tempo_us: null, time_signature: null, key_signature: null, tracks: [] };
    let total = 0;
    for (let tn = 0; tn < Math.min(ntracks, MAX_TRACKS); tn++) {
      if (r.i >= bytes.length) break;
      const id = String.fromCharCode(...r.take(4)), len = r.be(4), chunk = r.take(len);
      if (id !== "MTrk") continue;
      const t = { name: null, programs: new Array(16).fill(null), notes: [] };
      const tr = reader(chunk);
      let tick = 0, status = 0;
      const open = Array.from({ length: 16 * 128 }, () => []);
      while (tr.i < chunk.length) {
        tick += tr.varlen();
        let b = tr.u8();
        if (b < 0x80) { if (!status) throw "אירוע MIDI ללא סטטוס"; tr.i -= 1; b = status; }
        if (b === 0xff) {
          const ty = tr.u8(), l = tr.varlen(), d = tr.take(l);
          if (ty === 0x2f) break;
          if (ty === 0x03 && t.name === null) t.name = Array.from(new TextDecoder("utf-8", { fatal: false }).decode(d)).filter((c) => !isCtl(c)).slice(0, 80).join("");
          else if (ty === 0x51 && l === 3 && smf.tempo_us === null) smf.tempo_us = (d[0] << 16) | (d[1] << 8) | d[2];
          else if (ty === 0x58 && l >= 2 && smf.time_signature === null) smf.time_signature = [d[0], 1 << Math.min(d[1], 6)];
          else if (ty === 0x59 && l === 2 && smf.key_signature === null) smf.key_signature = [d[0] > 127 ? d[0] - 256 : d[0], d[1] === 1];
          status = 0;
        } else if (b === 0xf0 || b === 0xf7) {
          tr.take(tr.varlen());
          status = 0;
        } else if (b >= 0x80 && b <= 0xef) {
          status = b;
          const ch = b & 0x0f, kind = b & 0xf0, d1 = tr.u8() & 0x7f, d2 = kind === 0xc0 || kind === 0xd0 ? 0 : tr.u8() & 0x7f;
          if (kind === 0x90 && d2 > 0) open[ch * 128 + d1].push([tick, d2]);
          else if (kind === 0x80 || kind === 0x90) {
            const o = open[ch * 128 + d1].pop();
            if (o) { t.notes.push({ channel: ch, pitch: d1, velocity: o[1], start: o[0], len: Math.max(1, tick - o[0]) }); total++; }
          } else if (kind === 0xc0 && t.programs[ch] === null) t.programs[ch] = d1;
          if (total > MAX_NOTES) throw "יותר מדי תווים בקובץ";
        } else throw "אירוע MIDI לא תקין";
      }
      open.forEach((stack, k) => { for (const [s, v] of stack) t.notes.push({ channel: Math.floor(k / 128), pitch: k % 128, velocity: v, start: s, len: Math.max(1, tick - s) }); });
      t.notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
      smf.tracks.push(t);
    }
    if (smf.tracks.every((t) => !t.notes.length)) throw "לא נמצאו תווים בקובץ";
    return smf;
  }

  function hint(name) {
    const n = name.toLowerCase(), has = (ks) => ks.some((k) => n.includes(k));
    if (has(["melody", "vocal", "voice", "lead", "sing", "מנגינה", "שירה", "קול"])) return "lead";
    if (has(["drum", "perc", "תופים", "דרבוקה"])) return "drums";
    if (has(["bass", "בס"])) return "bass";
    if (has(["guitar", "גיטרה"])) return "guitar";
    if (has(["piano", "keys", "פסנתר"])) return "piano";
    if (has(["string", "violin", "cello", "כינור", "מיתר"])) return "strings";
    if (has(["flute", "sax", "trumpet", "clarinet", "חליל"])) return "winds";
    return null;
  }
  const programRole = (p) => (p <= 7 || (p >= 16 && p <= 23) ? "piano" : p <= 31 ? "guitar" : p <= 39 ? "bass" : p <= 55 ? "strings" : p <= 79 ? "winds" : "synth");
  const rnd = (x) => (x < 0 ? -Math.round(-x) : Math.round(x)); // Rust f64::round

  function planParts(s) {
    const scale = (t) => rnd((t * 960) / s.division);
    let parts = [];
    s.tracks.forEach((t, ti) => {
      const chans = [...new Set(t.notes.map((n) => n.channel))].sort((a, b) => a - b);
      for (const ch of chans) {
        const notes = t.notes.filter((n) => n.channel === ch).map((n) => [n.pitch, scale(n.start), Math.max(1, scale(n.len)), Math.max(1, n.velocity)]);
        const name = t.name && t.name.trim() ? t.name : `ערוץ ${ti + 1}`;
        let role, sure;
        if (ch === 9) { role = "drums"; sure = true; }
        else if (hint(name)) { role = hint(name); sure = true; }
        else { role = t.programs[ch] !== null ? programRole(t.programs[ch]) : "piano"; sure = false; }
        parts.push({ p: { name, role, notes }, sure });
      }
    });
    parts = parts.map((x, i) => [x, i]).sort((a, b) => b[0].p.notes.length - a[0].p.notes.length || a[1] - b[1]).map((x) => x[0]);
    parts = parts.slice(0, MAX_IMPORT_TRACKS);
    if (!parts.some((x) => x.p.role === "lead")) {
      const mono = (n) => { let o = 0; for (let i = 1; i < n.length; i++) if (n[i][1] < n[i - 1][1] + n[i - 1][2] - 10) o++; return 1 - o / Math.max(1, n.length); };
      const mean = (p) => p.notes.reduce((a, n) => a + n[0], 0) / p.notes.length;
      let best = -1;
      parts.forEach((x, i) => {
        if (x.p.role === "drums" || x.p.role === "bass" || mono(x.p.notes) < 0.8 || x.p.notes.length < 4) return;
        if (best < 0 || mean(x.p) > mean(parts[best].p) || (mean(x.p) === mean(parts[best].p) && i < best)) best = i;
      });
      if (best >= 0) parts[best].p.role = "lead";
    }
    const ORDER = ["lead", "drums", "bass", "guitar", "piano", "strings", "winds", "synth"];
    const order = (r) => (ORDER.indexOf(r) < 0 ? 9 : ORDER.indexOf(r));
    return parts.map((x, i) => [x.p, i]).sort((a, b) => order(a[0].role) - order(b[0].role) || a[1] - b[1]).map((x) => x[0]);
  }

  function keyOf(s) {
    if (!s.key_signature || s.key_signature[0] < -7 || s.key_signature[0] > 7) return null;
    const [sf, minor] = s.key_signature, maj = (((sf * 7) % 12) + 12) % 12;
    return minor ? { tonic: (maj + 9) % 12, mode: "minor" } : { tonic: maj, mode: "major" };
  }

  const api = { parse, planParts, keyOf, MAX_IMPORT_TRACKS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HUMMidi = api;
})(typeof window !== "undefined" ? window : globalThis);

;
// HUM Web — loudness (ITU-R BS.1770-4 / EBU R128) and mastering.
// Line-by-line port of engine/audio-core/src/loudness.rs; tests/parity/loudness compares both.
(function (root) {
  "use strict";
  function kFilters(fs) {
    let f0 = 1681.974450955533, g = 3.999843853973347, q = 0.7071752369554196;
    let k = Math.tan((Math.PI * f0) / fs);
    const vh = Math.pow(10, g / 20), vb = Math.pow(vh, 0.4996667741545416);
    let a0 = 1 + k / q + k * k;
    const s1 = { b0: (vh + (vb * k) / q + k * k) / a0, b1: (2 * (k * k - vh)) / a0, b2: (vh - (vb * k) / q + k * k) / a0, a1: (2 * (k * k - 1)) / a0, a2: (1 - k / q + k * k) / a0 };
    f0 = 38.13547087602444; q = 0.5003270373238773;
    k = Math.tan((Math.PI * f0) / fs);
    a0 = 1 + k / q + k * k;
    const s2 = { b0: 1, b1: -2, b2: 1, a1: (2 * (k * k - 1)) / a0, a2: (1 - k / q + k * k) / a0 };
    return [s1, s2];
  }
  function filter(x, f) {
    const y = new Float64Array(x.length); let z1 = 0, z2 = 0;
    for (let i = 0; i < x.length; i++) { const v = x[i], o = f.b0 * v + z1; z1 = f.b1 * v - f.a1 * o + z2; z2 = f.b2 * v - f.a2 * o; y[i] = o; }
    return y;
  }
  const db = (x) => (x <= 0 ? -Infinity : 20 * Math.log10(x));
  const f32 = Math.fround;

  function measure(stereo, sampleRate) {
    const fs = sampleRate, n = stereo.length >> 1, [s1, s2] = kFilters(fs);
    const ch = (c) => { const x = new Float64Array(n); for (let i = 0; i < n; i++) x[i] = stereo[2 * i + c]; return filter(filter(x, s1), s2); };
    const l = ch(0), r = ch(1);
    const block = Math.round(0.4 * fs), hop = Math.round(0.1 * fs);
    // running sums keep it fast; the same blocks as the Rust loop
    const z = [];
    for (let start = 0; start + block <= n; start += hop) {
      let e = 0; for (let i = start; i < start + block; i++) e += l[i] * l[i] + r[i] * r[i];
      z.push(e / block);
    }
    const lufs = (e) => -0.691 + 10 * Math.log10(e);
    const abs = z.filter((e) => e > 0 && lufs(e) > -70);
    let integrated = -Infinity;
    if (abs.length) {
      const rel = lufs(abs.reduce((a, e) => a + e, 0) / abs.length) - 10;
      const gated = abs.filter((e) => lufs(e) > rel);
      if (gated.length) integrated = lufs(gated.reduce((a, e) => a + e, 0) / gated.length);
    }
    let sp = 0; for (let i = 0; i < stereo.length; i++) sp = Math.max(sp, Math.abs(stereo[i]));
    const tp = truePeakLin(stereo);
    return { integrated, true_peak: db(Math.max(tp, sp)), sample_peak: db(sp) };
  }

  const OS = 4, TAPS = 48;
  let kernel = null;
  function interpKernel() {
    if (kernel) return kernel;
    const c = (TAPS - 1) / 2; kernel = [];
    for (let p = 0; p < OS; p++) {
      const ph = new Float64Array(TAPS / OS); let sum = 0;
      for (let k = 0; k < TAPS / OS; k++) {
        const n = k * OS + p, t = (n - c) / OS;
        const sinc = Math.abs(t) < 1e-12 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
        const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * n) / (TAPS - 1)) + 0.08 * Math.cos((4 * Math.PI * n) / (TAPS - 1));
        ph[k] = sinc * w; sum += ph[k];
      }
      for (let k = 0; k < ph.length; k++) ph[k] /= sum;
      kernel.push(ph);
    }
    return kernel;
  }
  function truePeakLin(stereo) {
    const n = stereo.length >> 1, k = interpKernel(), half = TAPS / OS;
    let peak = 0, sp = 0;
    for (let i = 0; i < stereo.length; i++) sp = Math.max(sp, Math.abs(stereo[i]));
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < n; i++) {
        let near = false;
        for (let j = Math.max(0, i - 2); j < Math.min(n, i + 3); j++) if (Math.abs(stereo[2 * j + c]) >= sp * 0.5) { near = true; break; }
        if (!near) continue;
        for (const ph of k) {
          let acc = 0;
          for (let j = 0; j < ph.length; j++) { const idx = i + j - (half >> 1); if (idx >= 0 && idx < n) acc += ph[j] * stereo[2 * idx + c]; }
          peak = Math.max(peak, Math.abs(acc));
        }
      }
    }
    return peak;
  }

  const PRESETS = [
    { id: "natural", name_he: "טבעי", target_lufs: -16, ceiling_dbtp: -1 },
    { id: "streaming", name_he: "סטרימינג", target_lufs: -14, ceiling_dbtp: -1 },
    { id: "commercial", name_he: "מסחרי חזק", target_lufs: -10, ceiling_dbtp: -1 },
    { id: "broadcast", name_he: "שידור (EBU R128)", target_lufs: -23, ceiling_dbtp: -1 },
  ];
  const preset = (id) => PRESETS.find((p) => p.id === id) || null;

  function limit(stereo, sampleRate, ceilingDbtp, margin) {
    const ceiling = Math.pow(10, (ceilingDbtp - margin) / 20), n = stereo.length >> 1, fs = sampleRate, look = Math.floor(0.005 * fs);
    const need = new Float64Array(n);
    for (let i = 0; i < n; i++) { const pk = Math.max(Math.abs(stereo[2 * i]), Math.abs(stereo[2 * i + 1])) * 1.12; need[i] = pk > ceiling ? ceiling / pk : 1; }
    const env = new Float64Array(n).fill(1), dq = [];
    let head = 0;
    for (let i = 0; i < n + look; i++) {
      if (i < n) { while (dq.length > head && need[dq[dq.length - 1]] >= need[i]) dq.pop(); dq.push(i); }
      if (i >= look) { const o = i - look; while (dq.length > head && dq[head] + look < o) head++; if (o < n) env[o] = dq.length > head ? need[dq[head]] : 1; }
    }
    const rel = Math.exp(-1 / (0.12 * fs)), att = Math.exp(-1 / ((0.005 * fs) / 3));
    let gcur = 1, minG = 1;
    for (let i = 0; i < n; i++) {
      const t = env[i];
      gcur = t < gcur ? t + (gcur - t) * att : t + (gcur - t) * rel;
      const gg = Math.min(gcur, t);
      minG = Math.min(minG, gg);
      stereo[2 * i] = f32(stereo[2 * i] * gg); stereo[2 * i + 1] = f32(stereo[2 * i + 1] * gg);
    }
    return minG;
  }

  function master(stereo, sampleRate, p) {
    const before = measure(stereo, sampleRate);
    let gainDb = 0, maxRed = 1, cur = before, margin = 0.3;
    for (let pass = 0; pass < 8; pass++) {
      if (!isFinite(cur.integrated)) break;
      const gd = p.target_lufs - cur.integrated;
      if (pass > 0 && Math.abs(gd) < 0.2 && cur.true_peak <= p.ceiling_dbtp - 0.2) break;
      const g = Math.pow(10, gd / 20);
      for (let i = 0; i < stereo.length; i++) stereo[i] = f32(stereo[i] * g);
      gainDb += gd;
      maxRed = Math.min(maxRed, limit(stereo, sampleRate, p.ceiling_dbtp, margin));
      cur = measure(stereo, sampleRate);
      if (cur.true_peak > p.ceiling_dbtp - 0.2) margin += cur.true_peak - (p.ceiling_dbtp - 0.25);
      if (maxRed < Math.pow(10, -15 / 20)) break;
    }
    if (cur.true_peak > p.ceiling_dbtp - 0.2) {
      const trim = Math.pow(10, (p.ceiling_dbtp - 0.25 - cur.true_peak) / 20);
      for (let i = 0; i < stereo.length; i++) stereo[i] = f32(stereo[i] * trim);
      cur = measure(stereo, sampleRate);
    }
    return { preset: p, before, after: cur, gain_db: gainDb, max_reduction_db: -db(maxRed) };
  }

  // ---------------- quality control (port of loudness.rs analyse / loudness_range) ----------------
  const QC_VERSION = "HUM QC 1.0 (ITU-R BS.1770-4, EBU Tech 3342)";
  function kWeighted(stereo, fs) {
    const n = stereo.length >> 1, [s1, s2] = kFilters(fs);
    const ch = (c) => { const x = new Float64Array(n); for (let i = 0; i < n; i++) x[i] = stereo[2 * i + c]; return filter(filter(x, s1), s2); };
    return [ch(0), ch(1)];
  }
  function loudnessRange(stereo, sampleRate) {
    const fs = sampleRate, [l, r] = kWeighted(stereo, fs), hop = Math.round(0.1 * fs);
    if (!hop) return 0;
    const sub = [];
    for (let s0 = 0; s0 + hop <= l.length; s0 += hop) { let e = 0; for (let i = s0; i < s0 + hop; i++) e += l[i] * l[i] + r[i] * r[i]; sub.push(e); }
    if (sub.length < 30) return 0;
    const lufs = (e) => -0.691 + 10 * Math.log10(e);
    const st = [];
    for (let i = 0; i + 30 <= sub.length; i++) { let e = 0; for (let k = i; k < i + 30; k++) e += sub[k]; e /= 30 * hop; if (e > 0 && lufs(e) > -70) st.push(e); }
    if (!st.length) return 0;
    const rel = lufs(st.reduce((a, e) => a + e, 0) / st.length) - 20;
    const v = st.map(lufs).filter((x) => x > rel);
    if (v.length < 2) return 0;
    v.sort((a, b) => a - b);
    const pct = (p) => v[Math.round((v.length - 1) * p)];
    return pct(0.95) - pct(0.1);
  }
  function analyse(stereo, sampleRate) {
    const n = stereo.length >> 1, fs = sampleRate, loud = measure(stereo, sampleRate);
    let clipped = 0, nonfinite = 0, sl = 0, sr = 0, ll = 0, rr = 0, lr = 0, first = -1, last = -1;
    for (let i = 0; i < n; i++) {
      const a = stereo[2 * i], b = stereo[2 * i + 1];
      if (!isFinite(a) || !isFinite(b)) { nonfinite++; continue; }
      if (Math.abs(a) >= f32(0.9999)) clipped++;
      if (Math.abs(b) >= f32(0.9999)) clipped++;
      sl += a; sr += b; ll += a * a; rr += b * b; lr += a * b;
      if (Math.max(Math.abs(a), Math.abs(b)) > 0.001) { if (first < 0) first = i; last = i; }
    }
    const correlation = ll > 0 && rr > 0 ? lr / Math.sqrt(ll * rr) : 1;
    const mono = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) { const m = f32((stereo[2 * i] + stereo[2 * i + 1]) / 2); mono[2 * i] = m; mono[2 * i + 1] = m; }
    const ml = measure(mono, sampleRate).integrated;
    const monoDrop = isFinite(loud.integrated) && isFinite(ml) ? loud.integrated - ml : isFinite(loud.integrated) ? 99 : 0;
    const tailN = Math.min(Math.floor(0.01 * fs), n);
    let te = 0; for (let i = n - tailN; i < n; i++) { const a = stereo[2 * i], b = stereo[2 * i + 1]; te += a * a + b * b; }
    const tailRms = tailN > 0 ? Math.sqrt(te / (2 * tailN)) : 0;
    const nn = Math.max(n, 1);
    return { loud, lra: loudnessRange(stereo, sampleRate), clipped, nonfinite, dc: [sl / nn, sr / nn],
      lead_silence: first >= 0 ? first / fs : n / fs, tail_silence: last >= 0 ? (n - 1 - last) / fs : n / fs,
      abrupt_end: tailRms > Math.pow(10, -30 / 20), correlation, mono_drop: monoDrop, duration: n / fs };
  }
  const FMT_HE = { wav16: "WAV 16 ביט", wav24: "WAV 24 ביט", wav32f: "WAV 32 ביט צף" };
  // The Hebrew report (identical wording in apps/core/src/qc.rs). Never says "broadcast compliant" (spec 42.4).
  function qcReport(q, p, sampleRate, fmt) {
    const rows = [], add = (name, status, text) => rows.push({ name, status, text });
    const n1 = (x) => (isFinite(x) ? x.toFixed(1) : "שקט");
    if (p) {
      const ok = isFinite(q.loud.integrated) && Math.abs(q.loud.integrated - p.target_lufs) <= 1;
      add("עוצמה כוללת", ok ? "ok" : p.id === "broadcast" ? "fail" : "warn", `${n1(q.loud.integrated)} LUFS (יעד ${p.target_lufs}, סטייה מותרת 1)`);
      add("שיא אמיתי", q.loud.true_peak <= p.ceiling_dbtp ? "ok" : "fail", `${n1(q.loud.true_peak)} dBTP (תקרה ${p.ceiling_dbtp})`);
    } else {
      add("עוצמה כוללת", "ok", `${n1(q.loud.integrated)} LUFS (בלי יעד: לא נבחר מאסטרינג)`);
      add("שיא אמיתי", q.loud.true_peak <= -1 ? "ok" : "warn", `${n1(q.loud.true_peak)} dBTP (מומלץ עד -1)`);
    }
    add("טווח עוצמה", "ok", `${q.lra.toFixed(1)} LU`);
    add("עיוות (קטימה)", q.clipped === 0 ? "ok" : "fail", q.clipped === 0 ? "אין דגימות קטומות" : `${q.clipped} דגימות בעוצמה המרבית`);
    add("ערכים פגומים", q.nonfinite === 0 ? "ok" : "fail", q.nonfinite === 0 ? "אין" : `${q.nonfinite} דגימות פגומות`);
    const dc = Math.max(Math.abs(q.dc[0]), Math.abs(q.dc[1]));
    add("היסט DC", dc < 0.001 ? "ok" : "warn", dc < 0.001 ? "תקין" : `היסט של ${(dc * 100).toFixed(2)}%`);
    add("שקט בהתחלה", q.lead_silence <= 2 ? "ok" : "warn", `${q.lead_silence.toFixed(2)} שניות`);
    add("סוף השיר", q.abrupt_end ? "warn" : q.tail_silence <= 5 ? "ok" : "warn", q.abrupt_end ? "נקטע בפתאומיות, בלי דעיכה" : `${q.tail_silence.toFixed(2)} שניות של שקט בסוף`);
    const monoOk = q.correlation >= 0 && q.mono_drop <= 3;
    add("תאימות למונו", monoOk ? "ok" : "warn", `התאמה בין הערוצים ${q.correlation.toFixed(2)}, ירידה במונו ${q.mono_drop.toFixed(1)} LU`);
    add("ערוצים", "ok", "2 (סטריאו)");
    add("קצב דגימה", "ok", `${sampleRate} הרץ`);
    add("פורמט ועומק", "ok", FMT_HE[fmt] || FMT_HE.wav16);
    add("משך", "ok", `${q.duration.toFixed(1)} שניות`);
    const fails = rows.filter((r) => r.status === "fail").length, warns = rows.filter((r) => r.status === "warn").length;
    const nf = fails === 1 ? "בעיה אחת" : `${fails} בעיות`, nw = warns === 1 ? "הערה אחת" : `${warns} הערות`;
    const summary = fails ? `לא עבר: ${nf}${warns ? ` ו${nw}` : ""}` : warns ? `עבר עם ${nw}` : "עבר את כל הבדיקות שנבחרו";
    return { profile: p ? p.name_he : "בלי מאסטרינג", summary, rows, version: QC_VERSION };
  }
  function qcText(rep) {
    const mark = { ok: "✓", warn: "!", fail: "✗" };
    return [`דוח בדיקת איכות — ${rep.profile}`, rep.summary, "", ...rep.rows.map((r) => `${mark[r.status]} ${r.name}: ${r.text}`), "", `נמדד בעזרת ${rep.version}`,
      "הדוח מראה את תוצאות הבדיקות בלבד. לפני שליחה לתחנה, בדוק את דרישות המסירה שלה."].join("\n");
  }
  const api = { measure, master, PRESETS, preset, analyse, loudnessRange, qcReport, qcText, QC_VERSION };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HUMLoudness = api;
})(typeof window !== "undefined" ? window : globalThis);

;
// HUM Web Lab — recorded instruments (port of engine/audio-core/src/sampler.rs).
// Sample data is embedded by tools/build_lab.py as window.__HUM_SAMPLE_DATA__ (MP3, base64);
// it is decoded once in the background. Until then — or if decoding fails — the synth plays.
(function (root) {
  "use strict";
  const sets = {};
  let state = "none"; // none | loading | ready | failed
  const listeners = [];

  function b64ToBuf(b64) {
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u.buffer;
  }
  const failed = []; // instruments this browser could not decode (they play with the synth)
  const index = {}; // every instrument the build ships: id → its entry (zones not decoded yet for lazy ones)
  const pending = {}; // id → promise of a lazy instrument being decoded
  async function decodeSet(id, inst) {
    // one instrument that cannot be decoded here (e.g. FLAC in an old browser) falls back to the synth; the rest still load
    try {
      const zones = [];
      for (const z of inst.zones) {
        const ctx = new OfflineAudioContext(1, 1, inst.sample_rate);
        const bytes = z.b64 ? b64ToBuf(z.b64) : await (await fetch(new URL(z.url, new URL(root.__HUM_SAMPLE_INDEX__, location.href)))).arrayBuffer();
        const buf = await ctx.decodeAudioData(bytes);
        // stereo stays stereo; a sustain loop (frames) lets a held note sound past the recording (as sampler.rs)
        const R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : null;
        const lp = z.loop_start >= 0 && z.loop_end > z.loop_start + 1 ? [z.loop_start, Math.min(z.loop_end, buf.length)] : null;
        zones.push({ root: z.root, lo: z.lo, hi: z.hi, vel_lo: z.vel_lo, vel_hi: z.vel_hi, data: buf.getChannelData(0), right: R, loop: lp, sample_rate: buf.sampleRate });
      }
      sets[id] = { zones, one_shot: inst.one_shot, release_s: inst.release_s, gain: inst.gain, name_he: inst.name_he, credit: inst.credit };
      return true;
    } catch (e) { failed.push(id); return false; }
  }
  async function load() {
    let data = root.__HUM_SAMPLE_DATA__;
    if (state !== "none") return state;
    // web build: instruments are separate files listed in instruments/index.json
    if (!data && root.__HUM_SAMPLE_INDEX__ && typeof fetch !== "undefined") {
      try { data = await (await fetch(root.__HUM_SAMPLE_INDEX__)).json(); } catch { data = null; }
    }
    if (!data) return state;
    if (typeof OfflineAudioContext === "undefined") { state = "failed"; return state; }
    state = "loading";
    try {
      // the core band loads at once; the other instruments (marked lazy) only when a track first uses one
      for (const [id, inst] of Object.entries(data)) {
        index[id] = inst;
        if (!inst.lazy) await decodeSet(id, inst);
      }
      state = "ready";
    } catch (e) {
      state = "failed";
    }
    listeners.forEach((f) => f(state));
    return state;
  }
  /** Starts decoding a lazy instrument (once); resolves true when it can play recorded. Listeners hear when it is ready. */
  function ensure(id) {
    if (sets[id]) return Promise.resolve(true);
    if (!index[id] || failed.includes(id)) return Promise.resolve(false);
    if (!pending[id]) pending[id] = decodeSet(id, index[id]).then((ok) => { listeners.forEach((f) => f(state, id)); return ok; });
    return pending[id];
  }
  const has = (id) => !!index[id] && !failed.includes(id);
  const zoneFor = (set, p, v) => set.zones.find((z) => p >= z.lo && p <= z.hi && v >= z.vel_lo && v <= z.vel_hi);

  /** notes: [{start,len,pitch,velocity}] → {left, right, rest} (rest = notes the set does not cover). */
  function renderSampled(notes, set, sr) {
    const rel = Math.floor(set.release_s * sr), rest = [], plan = [];
    const layered = set.zones.some((z) => z.vel_lo > 1 || z.vel_hi < 127);
    for (const n of notes) {
      const z = zoneFor(set, n.pitch, n.velocity);
      if (!z) { rest.push(n); continue; }
      const step = Math.pow(2, (n.pitch - z.root) / 12) * z.sample_rate / sr;
      const natural = Math.floor((z.data.length - 1) / step);
      const len = set.one_shot ? natural : z.loop ? n.len + rel : Math.min(natural, n.len + rel);
      plan.push([n, z, step, len]);
    }
    let end = 0;
    for (const [n, , , len] of plan) end = Math.max(end, n.start + len);
    const L = new Float32Array(end), R = new Float32Array(end);
    for (const [n, z, step, len] of plan) {
      const v = n.velocity / 127, gain = set.gain * (set.one_shot ? 0.55 + 0.45 * v : layered ? 0.35 + 0.65 * Math.pow(v, 0.8) : Math.pow(v, 1.3));
      const fadeFrom = set.one_shot ? len : Math.min(n.len, len), fadeLen = Math.max(1, len - fadeFrom);
      const dl = z.data, dr = z.right || z.data, lp = !set.one_shot && z.loop;
      const at = (d, k) => (k < d.length ? d[k] : 0);
      for (let i = 0; i < len; i++) {
        let pos = i * step;
        if (lp && pos >= lp[1]) pos = lp[0] + ((pos - lp[0]) % (lp[1] - lp[0]));
        const k = Math.floor(pos), f = pos - k, k2 = lp && k + 1 >= lp[1] ? lp[0] : k + 1;
        let g = gain;
        if (i >= fadeFrom) g *= 1 - (i - fadeFrom) / fadeLen;
        const al = at(dl, k), ar = at(dr, k);
        L[n.start + i] += (al + (at(dl, k2) - al) * f) * g; R[n.start + i] += (ar + (at(dr, k2) - ar) * f) * g;
      }
    }
    return { left: L, right: R, rest };
  }
  const api = { load, ensure, has, index, renderSampled, sets, failed, get state() { return state; }, onChange: (f) => listeners.push(f) };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HUMSamples = api;
})(typeof window !== "undefined" ? window : globalThis);

;
/* HUM effects — the insert plugins of a track (studio architecture S3), built from Web Audio nodes so playback and
 * export use exactly the same sound. One table (DEFS) describes every effect and its parameters; the engine keeps a
 * track's chain as [{kind, on, p:{param: value}}] and validates it against the same table (domain: effects.rs).
 * Gate needs sample-level processing: an AudioWorklet loaded once per audio context (play and export alike).
 * All DSP here is written for HUM on standard Web Audio building blocks; no third-party code. */
(function () {
  "use strict";
  const root = typeof window !== "undefined" ? window : globalThis;
  // [name, min, max, default, unit, Hebrew label]; "steps" params take whole numbers
  const DEFS = {
    eq: { he: "אקולייזר", desc: "מעצב את הצליל: בס, אמצע, נוכחות ובהירות", group: "בסיסי", params: [
      ["hp", 20, 400, 20, "Hz", "חיתוך נמוכים"], ["low_g", -15, 15, 0, "dB", "בס"], ["low_f", 40, 400, 100, "Hz", "תדר בס"],
      ["mid_g", -15, 15, 0, "dB", "אמצע"], ["mid_f", 200, 5000, 900, "Hz", "תדר אמצע"], ["mid_q", 0.3, 6, 1, "", "רוחב אמצע"],
      ["pres_g", -15, 15, 0, "dB", "נוכחות"], ["pres_f", 1500, 10000, 4000, "Hz", "תדר נוכחות"], ["high_g", -15, 15, 0, "dB", "בהירות"], ["high_f", 4000, 16000, 10000, "Hz", "תדר בהירות"]] },
    comp: { he: "קומפרסור", desc: "מאזן בין חלקים חלשים לחזקים", group: "בסיסי", params: [
      ["threshold", -50, 0, -18, "dB", "סף"], ["ratio", 1, 20, 4, ":1", "יחס"], ["attack", 0.1, 100, 10, "ms", "התקפה"], ["release", 20, 1000, 150, "ms", "שחרור"], ["knee", 0, 30, 6, "dB", "רכות"], ["makeup", 0, 24, 3, "dB", "הגברה"]] },
    gate: { he: "גייט", desc: "משתיק רעש רקע בין המשפטים", group: "קול", params: [["threshold", -80, -10, -50, "dB", "סף"], ["release", 10, 500, 80, "ms", "שחרור"], ["range", -80, 0, -60, "dB", "עומק"]] },
    deesser: { he: "דה-אסר", desc: "מרכך שי\"ן וסמ\"ך חדים בשירה", group: "קול", params: [["freq", 3000, 10000, 6000, "Hz", "תדר"], ["threshold", -50, 0, -28, "dB", "סף"], ["amount", 1, 20, 6, ":1", "עוצמה"]] },
    saturation: { he: "סטורציה", desc: "חום של טייפ ומנורות", group: "צבע", params: [["drive", 0, 24, 6, "dB", "דחיפה"], ["tone", 1000, 16000, 9000, "Hz", "צבע"], ["mix", 0, 1, 1, "", "כמות"]] },
    amp: { he: "מגבר גיטרה", desc: "עיוות ורמקול של מגבר", group: "גיטרה", params: [["gain", 0, 40, 18, "dB", "עיוות"], ["tone", 1000, 8000, 3500, "Hz", "צבע"], ["level", -24, 6, -8, "dB", "עוצמה"]] },
    chorus: { he: "כורוס", desc: "צליל רחב ונוצץ", group: "מודולציה", params: [["rate", 0.1, 5, 0.8, "Hz", "מהירות"], ["depth", 0, 1, 0.5, "", "עומק"], ["mix", 0, 1, 0.5, "", "כמות"]] },
    delay: { he: "דיליי", desc: "הד שחוזר בקצב השיר", group: "חלל", params: [["div", 0, 4, 1, "steps", "קצב"], ["feedback", 0, 0.9, 0.35, "", "חזרות"], ["tone", 1000, 12000, 5000, "Hz", "בהירות"], ["mix", 0, 1, 0.3, "", "כמות"]] },
    reverb: { he: "ריוורב", desc: "חלל: חדר, אולם, פלטה", group: "חלל", params: [["size", 0, 1, 0.5, "", "גודל"], ["decay", 0.2, 8, 1.8, "s", "דעיכה"], ["predelay", 0, 150, 20, "ms", "השהיה"], ["damp", 1000, 16000, 7000, "Hz", "בהירות"], ["mix", 0, 1, 0.25, "", "כמות"]] },
    filter: { he: "פילטר", desc: "סינון תדרים", group: "יצירתי", params: [["type", 0, 2, 0, "steps", "סוג"], ["freq", 20, 20000, 2000, "Hz", "תדר"], ["q", 0.1, 20, 0.7, "", "תהודה"]] },
    tremolo: { he: "טרמולו", desc: "רטט עוצמה בקצב", group: "מודולציה", params: [["rate", 0.5, 12, 5, "Hz", "מהירות"], ["depth", 0, 1, 0.5, "", "עומק"]] },
    width: { he: "הרחבת סטריאו", desc: "מרחיב או מצמצם את התמונה", group: "מיקס", params: [["width", 0, 2, 1.4, "", "רוחב"]] },
    limiter: { he: "לימיטר", desc: "מונע עיוות ומעלה עוצמה", group: "מאסטר", params: [["gain", 0, 18, 3, "dB", "הגברה"], ["ceiling", -12, 0, -1, "dB", "תקרה"], ["release", 10, 500, 60, "ms", "שחרור"]] },
  };
  const DIVS = [["1/2", 2], ["1/4", 1], ["1/8", 0.5], ["1/8 מנוקד", 0.75], ["1/16", 0.25]];
  const FILTERS = ["lowpass", "highpass", "bandpass"], FILTERS_HE = ["מעביר נמוכים", "מעביר גבוהים", "פס"];
  const MAX_INSERTS = 8;
  const PRESETS = {
    eq: [["שירה בהירה", { hp: 90, low_g: -2, mid_g: -1.5, mid_f: 350, pres_g: 2.5, high_g: 3 }], ["פחות בוץ", { hp: 70, mid_g: -4, mid_f: 300, mid_q: 1.2 }], ["גיטרה אקוסטית", { hp: 80, low_g: -3, pres_g: 2, high_g: 2 }], ["בס עגול", { hp: 30, low_g: 3, mid_g: -2, mid_f: 700, high_g: -3 }]],
    comp: [["שירה", { threshold: -20, ratio: 3, attack: 8, release: 120, makeup: 4 }], ["תופים", { threshold: -16, ratio: 4, attack: 15, release: 90, makeup: 3 }], ["בס יציב", { threshold: -22, ratio: 5, attack: 20, release: 200, makeup: 5 }]],
    reverb: [["חדר קטן", { size: 0.25, decay: 0.8, predelay: 8, mix: 0.2 }], ["אולם", { size: 0.8, decay: 3.2, predelay: 30, mix: 0.3 }], ["פלטה", { size: 0.5, decay: 1.8, predelay: 12, damp: 9000, mix: 0.25 }]],
    delay: [["הד קצר", { div: 4, feedback: 0.2, mix: 0.2 }], ["רבע בקצב", { div: 1, feedback: 0.35, mix: 0.25 }], ["שמינית מנוקדת", { div: 3, feedback: 0.4, mix: 0.25 }]],
  };
  const clampP = (kind, name, v) => { const d = (DEFS[kind] || { params: [] }).params.find((x) => x[0] === name); if (!d) return null; let x = Math.min(d[2], Math.max(d[1], Number(v))); if (d[4] === "steps") x = Math.round(x); return Number.isFinite(x) ? x : d[3]; };
  function defaults(kind) { const o = {}; for (const d of DEFS[kind].params) o[d[0]] = d[3]; return o; }
  /** A clean insert: known kind, known params only, clamped — or throws a Hebrew message. */
  function normalize(ins) {
    if (!ins || !DEFS[ins.kind]) throw "אפקט לא מוכר";
    const p = defaults(ins.kind);
    for (const [k, v] of Object.entries(ins.p || {})) { const c = clampP(ins.kind, k, v); if (c !== null) p[k] = c; }
    return { kind: ins.kind, on: ins.on !== false, p };
  }
  const db = (x) => Math.pow(10, x / 20);

  // ---- builders: each returns {input, output}; nodes are connected inside ----
  const irCache = new Map();
  function reverbIR(ctx, size, decay, damp) {
    const key = `${ctx.sampleRate}:${size.toFixed(2)}:${decay.toFixed(2)}:${Math.round(damp)}`;
    if (irCache.has(key)) return irCache.get(key);
    const sr = ctx.sampleRate, n = Math.max(1, Math.floor(sr * Math.min(8, decay * 1.3 + 0.05))), b = ctx.createBuffer(2, n, sr);
    const cD = 1 - Math.exp((-2 * Math.PI * damp) / sr), early = Math.floor(sr * (0.004 + 0.03 * size));
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
    for (let c = 0; c < 2; c++) {
      const d = b.getChannelData(c); let lp = 0;
      for (let i = 0; i < n; i++) {
        const t = i / sr, env = Math.exp((-6.91 * t) / decay) * (i < early ? i / early : 1);
        const cc = 1 - (1 - cD) * Math.min(1, t / Math.max(0.05, decay)); // bright at first, darker as it decays
        lp += cc * (rnd() - lp);
        d[i] = lp * env * (0.35 + 0.65 * size);
      }
    }
    // normalise the energy so mix=1 is about as loud as the dry sound
    let e = 0; for (let c = 0; c < 2; c++) { const d = b.getChannelData(c); for (let i = 0; i < n; i++) e += d[i] * d[i]; }
    const g = 1 / Math.sqrt(Math.max(1e-9, e / 2)) * 0.5; for (let c = 0; c < 2; c++) { const d = b.getChannelData(c); for (let i = 0; i < n; i++) d[i] *= g; }
    irCache.set(key, b);
    if (irCache.size > 24) irCache.delete(irCache.keys().next().value);
    return b;
  }
  function wet(ctx, input, wetNode, mix) { // dry + wet → output
    const out = ctx.createGain(), dry = ctx.createGain(), w = ctx.createGain();
    dry.gain.value = 1 - mix; w.gain.value = mix;
    input.connect(dry).connect(out); wetNode.connect(w).connect(out);
    return out;
  }
  function curve(drive, hard) {
    const n = 2048, c = new Float32Array(n), k = db(drive);
    for (let i = 0; i < n; i++) { const x = (i / (n - 1)) * 2 - 1; c[i] = hard ? Math.max(-1, Math.min(1, x * k)) * 0.5 + Math.tanh(x * k) * 0.5 : Math.tanh(x * k) / Math.tanh(k); }
    return c;
  }
  const B = {
    eq(ctx, p) {
      const mk = (type, f, g, q) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; if (g !== undefined) b.gain.value = g; if (q) b.Q.value = q; return b; };
      const chain = [mk("highpass", p.hp, undefined, 0.707), mk("lowshelf", p.low_f, p.low_g), mk("peaking", p.mid_f, p.mid_g, p.mid_q), mk("peaking", p.pres_f, p.pres_g, 1.1), mk("highshelf", p.high_f, p.high_g)];
      for (let i = 1; i < chain.length; i++) chain[i - 1].connect(chain[i]);
      return { input: chain[0], output: chain[chain.length - 1], nodes: chain };
    },
    comp(ctx, p) {
      const c = ctx.createDynamicsCompressor(); c.threshold.value = p.threshold; c.ratio.value = p.ratio; c.attack.value = p.attack / 1000; c.release.value = p.release / 1000; c.knee.value = p.knee;
      const g = ctx.createGain(); g.gain.value = db(p.makeup); c.connect(g); return { input: c, output: g };
    },
    gate(ctx, p) {
      if (!ctx.__humWorklet) { const g = ctx.createGain(); return { input: g, output: g }; } // not loaded yet: pass through
      const n = new AudioWorkletNode(ctx, "hum-gate", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2], parameterData: { threshold: p.threshold, release: p.release, range: p.range } });
      return { input: n, output: n };
    },
    deesser(ctx, p) { // the band above `freq` is compressed hard; the rest passes untouched (crossover sums flat)
      const input = ctx.createGain(), out = ctx.createGain();
      const lo = ctx.createBiquadFilter(); lo.type = "lowpass"; lo.frequency.value = p.freq; lo.Q.value = 0.5;
      const hi = ctx.createBiquadFilter(); hi.type = "highpass"; hi.frequency.value = p.freq; hi.Q.value = 0.5;
      const c = ctx.createDynamicsCompressor(); c.threshold.value = p.threshold; c.ratio.value = p.amount; c.attack.value = 0.001; c.release.value = 0.05; c.knee.value = 3;
      input.connect(lo).connect(out); input.connect(hi).connect(c).connect(out);
      return { input, output: out };
    },
    saturation(ctx, p) {
      const input = ctx.createGain(), ws = ctx.createWaveShaper(); ws.curve = curve(p.drive, false); ws.oversample = "4x";
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = p.tone; lp.Q.value = 0.5;
      input.connect(ws).connect(lp);
      return { input, output: wet(ctx, input, lp, p.mix) };
    },
    amp(ctx, p) {
      const input = ctx.createGain(), hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 90;
      const pre = ctx.createGain(); pre.gain.value = db(p.gain) / 6;
      const ws = ctx.createWaveShaper(); ws.curve = curve(Math.min(40, p.gain), true); ws.oversample = "4x";
      const cab1 = ctx.createBiquadFilter(); cab1.type = "lowpass"; cab1.frequency.value = p.tone; cab1.Q.value = 0.9;
      const cab2 = ctx.createBiquadFilter(); cab2.type = "peaking"; cab2.frequency.value = 1800; cab2.gain.value = 3; cab2.Q.value = 1;
      const out = ctx.createGain(); out.gain.value = db(p.level);
      input.connect(hp).connect(pre).connect(ws).connect(cab1).connect(cab2).connect(out);
      return { input, output: out };
    },
    chorus(ctx, p) {
      const input = ctx.createGain(), sum = ctx.createGain(); sum.gain.value = 0.5;
      for (const [ph, base] of [[0, 0.012], [Math.PI, 0.017]]) {
        const d = ctx.createDelay(0.05); d.delayTime.value = base;
        const lfo = ctx.createOscillator(); lfo.frequency.value = p.rate; const amt = ctx.createGain(); amt.gain.value = 0.004 * p.depth;
        lfo.connect(amt).connect(d.delayTime); lfo.start(ctx.currentTime + (ph ? 0.5 / Math.max(0.1, p.rate) : 0));
        input.connect(d).connect(sum);
      }
      return { input, output: wet(ctx, input, sum, p.mix) };
    },
    delay(ctx, p, bpm) {
      const input = ctx.createGain(), t = Math.min(4.9, (60 / (bpm || 120)) * DIVS[p.div][1]);
      const d = ctx.createDelay(5); d.delayTime.value = t;
      const fb = ctx.createGain(); fb.gain.value = p.feedback;
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = p.tone;
      input.connect(d); d.connect(lp); lp.connect(fb); fb.connect(d);
      return { input, output: wet(ctx, input, lp, p.mix) };
    },
    reverb(ctx, p) {
      const input = ctx.createGain(), pd = ctx.createDelay(0.2); pd.delayTime.value = p.predelay / 1000;
      const cv = ctx.createConvolver(); cv.normalize = false; cv.buffer = reverbIR(ctx, p.size, p.decay, p.damp);
      input.connect(pd).connect(cv);
      return { input, output: wet(ctx, input, cv, p.mix) };
    },
    filter(ctx, p) { const f = ctx.createBiquadFilter(); f.type = FILTERS[p.type] || "lowpass"; f.frequency.value = p.freq; f.Q.value = p.q; return { input: f, output: f }; },
    tremolo(ctx, p) {
      const g = ctx.createGain(); g.gain.value = 1 - p.depth / 2;
      const lfo = ctx.createOscillator(); lfo.frequency.value = p.rate; const amt = ctx.createGain(); amt.gain.value = p.depth / 2;
      lfo.connect(amt).connect(g.gain); lfo.start();
      return { input: g, output: g };
    },
    width(ctx, p) { // mid/side: side scaled by width
      const input = ctx.createGain(); input.channelCount = 2; input.channelCountMode = "explicit";
      const sp = ctx.createChannelSplitter(2), mg = ctx.createChannelMerger(2);
      const m = (a) => { const g = ctx.createGain(); g.gain.value = a; return g; };
      input.connect(sp);
      // L' = M + wS = (L+R)/2 + w(L-R)/2 ; R' = M - wS
      const w = p.width, a = (1 + w) / 2, b = (1 - w) / 2;
      const ll = m(a), rl = m(b), lr = m(b), rr = m(a);
      sp.connect(ll, 0); sp.connect(rl, 1); sp.connect(lr, 0); sp.connect(rr, 1);
      ll.connect(mg, 0, 0); rl.connect(mg, 0, 0); lr.connect(mg, 0, 1); rr.connect(mg, 0, 1);
      return { input, output: mg };
    },
    limiter(ctx, p) {
      const pre = ctx.createGain(); pre.gain.value = db(p.gain);
      const c = ctx.createDynamicsCompressor(); c.threshold.value = p.ceiling - 0.5; c.ratio.value = 20; c.knee.value = 0; c.attack.value = 0.001; c.release.value = p.release / 1000;
      const ceil = ctx.createGain(); ceil.gain.value = 1;
      pre.connect(c).connect(ceil);
      return { input: pre, output: ceil };
    },
  };
  /** Builds a track's insert chain: input → [effects that are on] → output. */
  function chain(ctx, inserts, bpm) {
    const input = ctx.createGain(); let last = input;
    for (const raw of inserts || []) {
      if (!raw || raw.on === false || !DEFS[raw.kind]) continue;
      let ins; try { ins = normalize(raw); } catch { continue; }
      const u = B[ins.kind](ctx, ins.p, bpm);
      last.connect(u.input); last = u.output;
    }
    return { input, output: last };
  }
  // ---- the gate worklet (sample-accurate envelope follower) ----
  const GATE_SRC = `class G extends AudioWorkletProcessor{static get parameterDescriptors(){return[{name:"threshold",defaultValue:-50},{name:"release",defaultValue:80},{name:"range",defaultValue:-60}]}
constructor(){super();this.env=0;this.g=1}
process(i,o,pp){const x=i[0],y=o[0];if(!x||!x.length)return true;const th=Math.pow(10,pp.threshold[0]/20),fl=Math.pow(10,pp.range[0]/20),rel=Math.exp(-1/(sampleRate*pp.release[0]/1000)),att=Math.exp(-1/(sampleRate*0.001));
for(let n=0;n<x[0].length;n++){let m=0;for(let c=0;c<x.length;c++)m=Math.max(m,Math.abs(x[c][n]));this.env=m>this.env?m:this.env*0.9995+m*0.0005;const t=this.env>th?1:fl;this.g=t>this.g?att*this.g+(1-att)*t:rel*this.g+(1-rel)*t;
for(let c=0;c<y.length;c++)y[c][n]=(x[Math.min(c,x.length-1)][n])*this.g}return true}}registerProcessor("hum-gate",G);`;
  let gateUrl = null;
  async function prepare(ctx) {
    if (!ctx || ctx.__humWorklet || !ctx.audioWorklet) return;
    try {
      gateUrl = gateUrl || URL.createObjectURL(new Blob([GATE_SRC], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(gateUrl); ctx.__humWorklet = true;
    } catch { /* gate passes through where worklets are not available */ }
  }
  root.HUMEffects = { DEFS, DIVS, FILTERS_HE, PRESETS, MAX_INSERTS, defaults, normalize, chain, prepare, reverbIR };
  if (typeof module !== "undefined") module.exports = root.HUMEffects;
})();

;
/* HUM vocal tuning ("כוונון קול", studio architecture — remaining base item): pulls a sung recording toward the notes of
 * the song's scale. Pitch is followed with the same pYIN tracker the humming transcription uses (transcribe.js);
 * the sound is re-pitched with TD-PSOLA (pitch-synchronous overlap-add), so the timing and the voice's character stay.
 * - strength 0…1: how far each note is pulled toward the scale note (1 = all the way).
 * - speed_ms: how fast the correction follows (fast ≈ the "auto-tune" effect, slow ≈ natural).
 * Unvoiced sound (breaths, s, sh) passes unchanged. Written for HUM; no third-party code. */
(function (root) {
  "use strict";
  const MAJOR = [0, 2, 4, 5, 7, 9, 11], MINOR = [0, 2, 3, 5, 7, 8, 10];
  /** Allowed pitch classes: the key's scale, or all twelve (chromatic). */
  function scaleOf(tonic, mode) {
    if (mode === "chromatic") return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    return (mode === "major" ? MAJOR : MINOR).map((x) => (x + tonic) % 12);
  }
  function nearest(m, pcs) {
    let best = Math.round(m), bd = 99;
    for (let o = Math.floor(m) - 2; o <= Math.ceil(m) + 2; o++) if (pcs.includes(((o % 12) + 12) % 12) && Math.abs(o - m) < bd) { bd = Math.abs(o - m); best = o; }
    return best;
  }
  /** The correction in semitones per analysis frame (null where unvoiced), smoothed by speed_ms. */
  function correctionCurve(track, pcs, strength, speedMs) {
    const n = track.midi.length, hop = track.hop_sec, a = Math.exp(-hop / Math.max(0.001, speedMs / 1000));
    const out = new Array(n).fill(null); let c = 0, prevVoiced = false;
    for (let i = 0; i < n; i++) {
      const m = track.midi[i];
      if (m === null || !Number.isFinite(m)) { out[i] = null; prevVoiced = false; continue; }
      const target = (nearest(m, pcs) - m) * strength;
      c = prevVoiced ? a * c + (1 - a) * target : target; // a new note starts corrected; inside a note the speed applies
      out[i] = c; prevVoiced = true;
    }
    return out;
  }
  /**
   * TD-PSOLA with a time-varying ratio. Analysis marks every period (detected pitch where voiced, a fixed 150 Hz
   * period elsewhere), aligned to the waveform's peaks; synthesis marks spaced period/ratio; 2-period Hann grains;
   * the output is normalised by the summed windows, so unvoiced parts (ratio 1) come back exactly as recorded.
   */
  function psola(x, sr, f0At, ratioAt) {
    const N = x.length, out = new Float32Array(N), wsum = new Float32Array(N);
    const per = (i) => { const f = f0At(i); return Math.max(32, Math.min(sr / 60, Math.round(sr / (f || 150)))); };
    // analysis marks, each moved to the local peak so grains line up with the voice's cycles
    const marks = [];
    let p = per(0);
    for (let i = p; i < N - 1; ) {
      p = per(i);
      let best = i, bv = -1;
      if (f0At(i)) for (let j = Math.max(0, i - (p >> 2)); j < Math.min(N, i + (p >> 2)); j++) { const v = Math.abs(x[j]); if (v > bv) { bv = v; best = j; } }
      if (marks.length && best <= marks[marks.length - 1] + (p >> 1)) best = marks[marks.length - 1] + p;
      marks.push(best); i = best + p;
    }
    if (!marks.length) return x.slice();
    let k = 0;
    for (let ts = marks[0]; ts < N; ) {
      while (k + 1 < marks.length && Math.abs(marks[k + 1] - ts) <= Math.abs(marks[k] - ts)) k++;
      const m = marks[k], P = per(m), r = f0At(m) ? ratioAt(m) : 1;
      const step = k + 1 < marks.length ? marks[k + 1] - m : P; // the real spacing here: ratio 1 reproduces the input exactly
      for (let j = -P; j < P; j++) {
        const si = m + j, di = Math.round(ts) + j;
        if (si < 0 || si >= N || di < 0 || di >= N) continue;
        const w = 0.5 - 0.5 * Math.cos((Math.PI * (j + P)) / P);
        out[di] += x[si] * w; wsum[di] += w;
      }
      ts += Math.max(8, step / r); // fractional: no rounding drift in the new pitch
    }
    for (let i = 0; i < N; i++) out[i] = wsum[i] > 1e-3 ? out[i] / wsum[i] : x[i];
    return out;
  }
  /** Tune a mono recording. opts: {tonic, mode: "major"|"minor"|"chromatic", strength, speed_ms}. Returns {pcm, frames_corrected, mean_cents}. */
  function tune(pcm, sr, opts) {
    const T = root.HUMTranscribe;
    if (!T) throw new Error("tracker missing");
    const track = T.pitchTrack(pcm, 1, sr), pcs = scaleOf(opts.tonic | 0, opts.mode || "minor");
    const curve = correctionCurve(track, pcs, Math.max(0, Math.min(1, opts.strength ?? 0.8)), opts.speed_ms ?? 40);
    const hopS = track.hop_sec * sr, n = curve.length;
    const frame = (i) => Math.min(n - 1, Math.max(0, Math.floor(i / hopS)));
    const f0At = (i) => { const m = track.midi[frame(i)]; return m === null ? 0 : 440 * Math.pow(2, (m - 69) / 12); };
    const ratioAt = (i) => { const c = curve[frame(i)]; return c === null ? 1 : Math.pow(2, c / 12); };
    const out = psola(pcm, sr, f0At, ratioAt);
    let cnt = 0, sum = 0; for (const c of curve) if (c !== null) { cnt++; sum += Math.abs(c) * 100; }
    return { pcm: out, frames_corrected: cnt, mean_cents: cnt ? sum / cnt : 0 };
  }
  root.HUMTune = { tune, scaleOf, nearest, correctionCurve, psola };
  if (typeof module !== "undefined" && module.exports) module.exports = root.HUMTune;
})(typeof window !== "undefined" ? window : globalThis);

;
// HUM Web Lab — browser test build (no install).
// A line-by-line port of the Rust rules in domain/project (hebrew.rs, command.rs,
// model.rs) and apps/core (router). Parity with the Rust code is enforced by
// tests/parity (the same 46+ utterances must give identical results).
// Sound comes from the browser (Web Audio), not from the HUM Windows engine.
(function () {
  "use strict";
  const root = typeof window !== "undefined" ? window : globalThis;

  // ---------------- model ----------------
  const ROLE_HE = {
    melody: "מנגינה (זמזום)", vocals: "שירה", drums: "תופים", bass: "בס", guitar: "גיטרה", piano: "פסנתר",
    strings: "כלי קשת", winds: "כלי נשיפה", middle_eastern: "כלי מזרח־תיכוני", synth: "סינתיסייזר", lead: "מנגינה בכלי", other: "ערוץ",
  };
  const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
  const MIN_GAIN = -60, MAX_GAIN = 12, MAX_TRACKS = 64;
  const FX_EQ_RANGE = 12;
  // Instruments (same list and names as domain/project Instrument). The recorded sound has the same id; when it is
  // not available (still loading, or an old browser) the synth plays the patch in SYNTH_OF.
  const INSTRUMENT_HE = { auto: "אוטומטי", epiano: "פסנתר חשמלי", soft_lead: "סינתיסייזר רך", flute: "חליל", strings: "כלי קשת", piano: "פסנתר", bass: "בס חשמלי", guitar: "גיטרה", drums: "מערכת תופים",
    electric_guitar: "גיטרה חשמלית", nylon_guitar: "גיטרה קלאסית", violin: "כינור סולו", cello: "צ'לו", double_bass: "קונטרבס", harp: "נבל",
    trumpet: "חצוצרה", trombone: "טרומבון", horn: "קרן יער", tuba: "טובה", clarinet: "קלרינט", oboe: "אבובה", sax: "סקסופון", harmonica: "מפוחית",
    accordion: "אקורדיון", organ: "עוגב", marimba: "מרימבה", vibraphone: "ויברפון", glockenspiel: "פעמונים", world_percussion: "כלי הקשה עולמיים" };
  const SYNTH_OF = { electric_guitar: "guitar", nylon_guitar: "guitar", violin: "strings", cello: "strings", double_bass: "bass", harp: "guitar",
    trumpet: "soft_lead", trombone: "soft_lead", horn: "strings", tuba: "bass", clarinet: "flute", oboe: "flute", sax: "soft_lead", harmonica: "flute",
    accordion: "soft_lead", organ: "epiano", marimba: "epiano", vibraphone: "epiano", glockenspiel: "epiano", world_percussion: "drums" };
  const synthPatch = (inst) => SYNTH_OF[inst] || inst;
  const FX_OFF = Object.freeze({ eq_low_db: 0, eq_mid_db: 0, eq_high_db: 0, comp: 0, reverb: 0 });

  function keyLabel(k) {
    const t = k.tonic % 12;
    const flat = k.mode === "major" ? [5, 10, 3, 8, 1, 6].includes(t) : [2, 7, 0, 5, 10, 3].includes(t);
    return `${(flat ? FLAT_NAMES : NOTE_NAMES)[t]} ${k.mode === "major" ? "Major" : "Minor"}`;
  }
  const defaultKind = (role) => (role === "melody" || role === "vocals" ? "audio" : "midi");

  // ---------------- Hebrew interpreter (port of hebrew.rs) ----------------
  function normalize(input) {
    let s = "";
    for (const c of input) {
      const cp = c.codePointAt(0);
      if (cp >= 0x0591 && cp <= 0x05c7) continue;
      if ("׳'`’".includes(c)) continue;
      s += "-–—,.!?:;\"״()".includes(c) ? " " : c;
    }
    return s.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  }
  function stems(word) {
    const out = [word];
    let w = word;
    for (let i = 0; i < 2; i++) {
      const chars = Array.from(w);
      if (chars.length > 2 && "הולבשמכ".includes(chars[0])) { w = chars.slice(1).join(""); out.push(w); } else break;
    }
    return out;
  }
  const hasAny = (words, keys) => words.some((w) => stems(w).some((s) => keys.includes(s)));
  const containsPhrase = (text, ps) => ps.some((p) => text.includes(p));
  const ROLE_WORDS = [
    ["drums", ["תופים", "תוף", "דראמס", "מתופף", "drums"]],
    ["bass", ["בס", "בסים", "bass"]],
    ["guitar", ["גיטרה", "גיטרות", "guitar"]],
    ["piano", ["פסנתר", "קלידים", "קלידן", "piano"]],
    ["vocals", ["שירה", "ווקאל", "ווקאלים", "זמר", "זמרת", "קול", "vocals"]],
    ["melody", ["מנגינה", "זמזום", "melody"]],
    ["strings", ["כינור", "כינורות", "מיתרים", "צלו", "strings"]],
    ["winds", ["חליל", "סקסופון", "חצוצרה", "קלרינט", "נשיפה"]],
    ["middle_eastern", ["עוד", "דרבוקה", "קאנון", "סאז", "בוזוקי"]],
    ["synth", ["סינתיסייזר", "סינת", "synth"]],
    ["lead", ["ליד", "מוביל", "lead"]],
  ];
  const findRole = (words) => { const f = ROLE_WORDS.find(([, k]) => hasAny(words, k)); return f ? f[0] : null; };
  const NUM_WORDS = [["אחד", 1], ["אחת", 1], ["שניים", 2], ["שתיים", 2], ["שני", 2], ["שתי", 2], ["שלוש", 3], ["שלושה", 3], ["ארבע", 4], ["ארבעה", 4], ["חמש", 5], ["חמישה", 5], ["שש", 6], ["שישה", 6], ["שבע", 7], ["שבעה", 7], ["שמונה", 8], ["תשע", 9], ["תשעה", 9], ["עשר", 10], ["עשרה", 10], ["חצי", 0.5]];
  function findNumber(words) {
    for (const w of words) {
      const digits = Array.from(w).filter((c) => (c >= "0" && c <= "9") || c === ".").join("");
      if (digits && /[0-9]/.test(w)) { const v = Number(digits); if (!Number.isNaN(v) && digits !== ".") return v; }
    }
    for (const w of words) for (const s of stems(w)) { const f = NUM_WORDS.find(([k]) => k === s); if (f) return f[1]; }
    return null;
  }
  let ACTIVE = null; // the track selected on screen while one sentence is interpreted (interpretWith)
  function resolveTrack(p, role) {
    const m = p.tracks.filter((t) => t.role === role);
    if (ACTIVE && m.length > 1 && m.some((t) => t.id === ACTIVE)) return { id: ACTIVE }; // several of this kind: the selected one
    if (m.length === 1) return { id: m[0].id };
    if (m.length === 0) return { err: { kind: "unsupported", msg: `אין בפרויקט ערוץ ${ROLE_HE[role]}. אפשר לכתוב "הוסף ${ROLE_HE[role]}"` } };
    return { err: { kind: "ask", msg: `יש כמה ערוצי ${ROLE_HE[role]}: ${m.map((t) => t.name).join(", ")}. לאיזה מהם התכוונת?` } };
  }
  const SECTION_WORDS = ["בית", "פזמון", "פתיחה", "סיום", "גשר", "אינטרו", "אאוטרו", "מעבר", "בתים", "פזמונים"];
  const NOTES = [["דו", 0], ["רה", 2], ["מי", 4], ["פה", 5], ["פא", 5], ["סול", 7], ["לה", 9], ["סי", 11], ["c", 0], ["d", 2], ["e", 4], ["f", 5], ["g", 7], ["a", 9], ["b", 11]];
  function findKey(words, text) {
    let tonic = null;
    for (const w of words) { const f = NOTES.find(([n]) => n === w); if (f) tonic = f[1]; }
    if (tonic === null) return null;
    let pc = tonic;
    if (text.includes("דיאז") || text.includes("#")) pc = (pc + 1) % 12;
    else if (text.includes("במול")) pc = (pc + 11) % 12;
    let mode;
    if (containsPhrase(text, ["מינור", "minor"])) mode = "minor";
    else if (containsPhrase(text, ["מזור", "מגור", "major"])) mode = "major";
    else return null;
    return { tonic: pc, mode };
  }
  const LATER = [
    [["דיליי", "השהיה", "כורוס", "דיסטורשן"], "האפקט הזה יגיע בשלב ההפקה האלקטרונית. כרגע יש: הד, דחיסה, ובהיר/חם יותר"],
    [["מאסטר", "מאסטרינג", "לופנס", "סטמים"], 'מאסטרינג וייצוא כל כלי בנפרד נמצאים בלשונית "מאסטר" במיקסר: בוחרים יעד עוצמה ולוחצים "הורד שיר עם מאסטרינג"'],
  ];
  const GENERIC = "עדיין לא הבנתי את הבקשה. כרגע אפשר: להקליט, לנגן ולעצור, לשנות עוצמה של ערוץ, להשתיק, סולו, שמאלה/ימינה, להוסיף ערוץ, לשנות קצב או סולם, לבטל ולבצע שוב";

  const shiftHow = (n) => (n === 1 ? "בחצי טון" : n === 2 ? "בטון" : n === 3 ? "בטון וחצי" : n === 12 ? "באוקטבה" : `ב-${n} חצאי טונים`);
  // proposed: {type:"undo"|"redo"|"add",role} | {type:"action", action:{...}}
  // a sentence that acts on one track but names no instrument means the track selected on screen (as hebrew.rs interpret_with)
  const TRACK_VERBS = ["ריוורב", "הד", "אקו", "חלל", "קומפרסור", "דחיסה", "קומפרסיה", "תדחוס", "דחוס", "אקולייזר", "איקיו", "eq",
    "הנמך", "תנמיך", "הנמיכי", "הגבר", "תגביר", "הגבירי", "עוצמה", "ווליום", "חזק", "חלש", "חזק יותר", "שקט",
    "השתקה", "השתק", "תשתיק", "השתיקי", "מיוט", "סולו", "שמאלה", "ימינה", "שמאל", "ימין", "בהיר", "חם", "עמום", "חד", "צלול", "רך"];
  function interpretWith(input, p, activeTrack) {
    const t = activeTrack && p.tracks.find((x) => x.id === activeTrack);
    if (!t) return interpret(input, p);
    const words = normalize(input).split(" ");
    const rw = ROLE_WORDS.find(([r]) => r === t.role);
    const text = rw && !findRole(words) && hasAny(words, TRACK_VERBS) ? `${input} ${rw[1][0]}` : input;
    ACTIVE = t.id;
    try { return interpret(text, p); } finally { ACTIVE = null; }
  }
  function interpret(input, p) {
    const text = normalize(input);
    const words = text.split(" ");
    if (!text) return { kind: "unsupported", msg: "לא נכתבה הוראה" };
    const ready = (proposed, msg) => ({ kind: "ready", proposed, msg });
    const mentionsMute = hasAny(words, ["השתקה", "השתק", "תשתיק", "השתיקי", "מיוט"]);
    const mentionsSolo = hasAny(words, ["סולו"]);
    if (!mentionsMute && !mentionsSolo && findRole(words) === null) {
      if (containsPhrase(text, ["בצע שוב", "תבצע שוב", "חזור על", "תחזיר את מה שביטלת", "redo"])) return ready({ type: "redo" }, "ביצוע מחדש של הפעולה שבוטלה");
      if (hasAny(words, ["בטל", "תבטל", "בטלי", "undo", "אנדו"])) return ready({ type: "undo" }, "ביטול הפעולה האחרונה");
    }
    // ---- vocal range (stage 5) — before the role words, since "קול" also names the vocals track ----
    const rangeWords = containsPhrase(text, ["טווח הקול", "טווח קולי", "טווח הקולי", "הטווח שלי", "טווח שלי", "פרופיל קולי", "פרופיל הקול"]);
    const testWords = containsPhrase(text, ["בדיקת קול", "בדוק את הקול", "תבדוק את הקול", "תבדוק לי את הקול", "בדקי את הקול", "מדוד את הטווח", "תמדוד את הטווח", "למדוד את הטווח", "מדידת טווח", "כייל מחדש", "כיול מחדש"]);
    if (rangeWords || testWords) {
      if (hasAny(words, ["מחק", "תמחק", "מחקי", "שכח", "תשכח", "שכחי"])) return { kind: "confirm", proposed: { type: "vocal_delete" }, msg: "למחוק את טווח הקול השמור? אפשר למדוד שוב בכל זמן" };
      if (hasAny(words, ["שמור", "תשמור", "שמרי", "תזכור", "זכור"])) return ready({ type: "vocal_save" }, "שמירת טווח הקול שנמדד לפעם הבאה");
      if (hasAny(words, ["סיים", "תסיים", "סיימתי", "עצור", "תעצור", "עצרי", "די"])) return ready({ type: "vocal_stop" }, "סיום בדיקת הקול");
      return ready({ type: "vocal_start" }, "בדיקת טווח הקול");
    }
    if (containsPhrase(text, ["שנוח לי", "נוח לי", "שמתאים לקול", "מתאים לקול", "מתאים לטווח", "גבוה מדי בשבילי", "נמוך מדי בשבילי", "גבוה לי מדי", "נמוך לי מדי"]))
      return ready({ type: "recommend_key" }, "המלצה על סולם שמתאים לקול שלך");
    // ---- an existing song by name (stage 5b) — before arrangement words like "פלייבק" ----
    if (hasAny(words, ["פלייבק", "ליווי", "עיבוד", "קריוקי", "תעבד", "תעשה", "תכין", "מצא", "תמצא", "חפש", "תחפש"])) {
      const i = text.indexOf("לשיר ");
      if (i >= 0) {
        const rest = text.slice(i + "לשיר ".length).trim();
        const j = rest.lastIndexOf(" של ");
        const t = (j >= 0 ? rest.slice(0, j) : rest).trim(), a = j >= 0 ? rest.slice(j + 4).trim() : null;
        if (t && !["הזה", "שלי", "שלנו", "החדש", "הזאת", "זה"].includes(t)) return ready({ type: "existing", title: t, artist: a || null }, "פלייבק לשיר קיים — דרכים חוקיות");
      }
    }
    // ---- composition and arrangement (stage 4) ----
    const style = findStyle(words, text);
    const composeWords = hasAny(words, ["תלחין", "הלחן", "להלחין", "תלחיני", "תמציא"]) || containsPhrase(text, ["תכתוב שיר", "כתוב שיר", "תכתוב לי שיר", "שיר חדש ב", "שיר מקורי"]);
    if (composeWords) {
      const major = hasAny(words, ["שמח", "שמחה", "עליז", "עליזה", "קצבי", "אופטימי"]) ? true : hasAny(words, ["עצוב", "עצובה", "מרגש", "מרגשת", "רגשי", "רגשית", "נוגה", "מלנכולי"]) ? false : null;
      const st = style || (hasAny(words, ["מרגש", "מרגשת", "רגשי", "רגשית"]) ? "ballad" : null);
      const n = findNumber(words);
      const bpm = n !== null && n >= 40 && n <= 240 ? n : null;
      const solo = soloAsk(words);
      let withSolo = solo ? `, עם סולו${solo.role ? " " + ROLE_HE[solo.role] : ""}${solo.at === "end" ? " בסוף" : " אחרי הפזמון השני"}` : "";
      const lift = containsPhrase(text, ["שינוי סולם", "מודולציה", "עליית סולם", "סולם עולה", "תעלה את הסולם", "להעלות את הסולם", "סולם גבוה יותר בסוף"]) ? (containsPhrase(text, ["טון שלם", "שני חצאי טונים"]) ? 2 : 1) : 0;
      if (lift) withSolo += ", ועליית סולם בפזמון האחרון";
      return ready({ type: "compose", style: st, major, bpm, roles: allRoles(words), solo, lift }, `הלחנת שיר מקורי בסגנון ${st ? STYLE_HE[st] : "פופ"}${withSolo} (בפרויקט חדש)`);
    }
    const other = hasAny(words, ["אחר", "אחרת", "אחרים", "אחרות", "חדש", "חדשה", "חדשים", "חדשות", "שונה", "שונים"]);
    const harmonyWords = hasAny(words, ["אקורדים", "הרמוניה", "אקורד"]);
    const arrangeWords = hasAny(words, ["עיבוד", "תעבד", "עבד", "לעבד", "תעבדי", "פלייבק", "ליווי", "תלווה", "עיבודים"]);
    const styleChange = style !== null && (hasAny(words, ["תעשה", "תהפוך", "שיהיה", "בסגנון", "סגנון", "יותר"]) || words.length <= 2);
    if (harmonyWords && (other || hasAny(words, ["מרגשת", "מרגש", "עשירה", "מעניינת", "שמחה", "עצובה"])))
      return ready({ type: "arrange", style, roles: [], only: null, new_harmony: true }, "הרמוניה חלופית (אקורדים אחרים) — המנגינה נשארת");
    if (other && !arrangeWords) {
      const r = findRole(words);
      if (["drums", "bass", "guitar", "piano", "strings", "winds"].includes(r)) return ready({ type: "arrange", style: null, roles: [], only: r, new_harmony: false }, `${ROLE_HE[r]} בגרסה אחרת — שאר העיבוד נשאר`);
    }
    if (arrangeWords || styleChange) return ready({ type: "arrange", style, roles: allRoles(words), only: null, new_harmony: false }, `עיבוד למנגינה${style ? ` בסגנון ${STYLE_HE[style]}` : ""}`);

    // ---- humming → notes (stage 3) ----
    const notesWord = hasAny(words, ["תווים", "תו", "נוטות", "מידי", "midi"]);
    if ((notesWord && hasAny(words, ["הפוך", "תהפוך", "הפכי", "המר", "תמיר", "המירי", "תרשום", "רשום", "כתוב", "תכתוב", "תמלל", "תעתיק", "לתווים"])) || hasAny(words, ["לתווים", "תמלל", "תמלול"]) || containsPhrase(text, ["זמזום לתווים", "את הזמזום לתווים"]))
      return ready({ type: "transcribe" }, "הפיכת ההקלטה האחרונה לתווים");
    if (containsPhrase(text, ["יישר לקצב", "תיישר לקצב", "ליישר לקצב", "יישור לקצב", "על הקצב", "יישר את התווים", "תיישר את התווים", "קוונטיזציה", "quantize"]))
      return ready({ type: "quantize" }, "יישור התווים לקצב");
    if (containsPhrase(text, ["תחליף כלי", "החלף כלי", "תחליף את הכלי", "החלף את הכלי", "בצליל של", "תנגן את המנגינה ב", "נגן את המנגינה ב", "שינוי כלי", "תשנה כלי", "שנה כלי", "שנה את הכלי", "תשנה את הכלי"])) {
      const INSTR = [["flute", ["חליל", "חלילית"]], ["strings", ["כינור", "כינורות", "מיתרים", "קשת"]], ["soft_lead", ["סינת", "סינתיסייזר", "סינטי"]], ["epiano", ["פסנתר"]], ["auto", ["אוטומטי", "רגיל"]]];
      const hit = INSTR.find(([, k]) => hasAny(words, k));
      if (!hit) return { kind: "ask", msg: "לאיזה צליל להחליף? אפשר: פסנתר חשמלי, סינתיסייזר רך, חליל, כלי קשת" };
      const lead = p.tracks.find((t) => t.role === "lead");
      if (!lead) return { kind: "unsupported", msg: 'עדיין אין מנגינה בכלי. קודם צריך להפוך את הזמזום לתווים ("הפוך לתווים")' };
      return ready({ type: "action", action: { a: "set_instrument", track_id: lead.id, instrument: hit[0] } }, `המנגינה תנוגן ב${INSTRUMENT_HE[hit[0]]}`);
    }
    if (hasAny(words, ["קצב", "טמפו", "bpm"]) || containsPhrase(text, ["מהר יותר", "לאט יותר", "תאיץ", "תאט", "האט"])) {
      const cur = p.bpm;
      let target;
      const n = findNumber(words);
      if (n !== null && n >= 20) target = n;
      else if (containsPhrase(text, ["מהר", "תאיץ", "הגבר", "תעלה", "העלה", "הרם"])) target = Math.round(cur * 1.1);
      else if (containsPhrase(text, ["לאט", "תאט", "האט", "הורד", "תוריד", "הנמך"])) target = Math.round(cur * 0.9);
      else return { kind: "ask", msg: `הקצב כרגע ${cur.toFixed(0)}. לאיזה קצב לשנות?` };
      return ready({ type: "action", action: { a: "set_tempo", bpm: target } }, `שינוי הקצב מ-${cur.toFixed(0)} ל-${target.toFixed(0)}`);
    }
    if (hasAny(words, ["סולם", "טון", "טונים", "אוקטבה"]) || containsPhrase(text, ["מינור", "מזור", "major", "minor", "חצאי טונים"])) {
      const up = hasAny(words, ["העלה", "תעלה", "העלי", "הרם", "תרים", "הרימי", "למעלה", "גבוה"]);
      const down = hasAny(words, ["הורד", "תוריד", "הורידי", "הנמך", "תנמיך", "למטה", "נמוך"]);
      if (up !== down) {
        let amount = null;
        const n = findNumber(words);
        if (containsPhrase(text, ["חצי טון"])) amount = 1;
        else if (containsPhrase(text, ["אוקטבה"])) amount = 12;
        else if (n !== null && n >= 1 && n <= 12 && Number.isInteger(n)) amount = containsPhrase(text, ["חצאי טו", "חצי טונים"]) ? n : hasAny(words, ["טונים", "טון"]) ? n * 2 : n;
        else if (containsPhrase(text, ["טון וחצי"])) amount = 3;
        else if (hasAny(words, ["טון"])) amount = 2;
        if (amount === null) return { kind: "ask", msg: 'בכמה להזיז? למשל: "העלה את הסולם בחצי טון" או "הורד את הסולם בטון"' };
        const st = up ? amount : -amount;
        const to = { tonic: (((p.key.tonic + st) % 12) + 12) % 12, mode: p.key.mode };
        return ready({ type: "action", action: { a: "transpose", semitones: st } }, `השיר ${up ? "עלה" : "ירד"} ${shiftHow(amount)}: מ-${keyLabel(p.key)} ל-${keyLabel(to)}. התופים וההקלטות לא משתנים`);
      }
      const key = findKey(words, text);
      if (key) return ready({ type: "action", action: { a: "set_key", key } }, `שינוי הסולם מ-${keyLabel(p.key)} ל-${keyLabel(key)}`);
      return { kind: "ask", msg: 'לאיזה סולם? למשל: "סולם לה מינור" או "סולם דו מז׳ור"' };
    }
    const stopWords = hasAny(words, ["עצור", "תעצור", "עצרי", "הפסק", "תפסיק", "סיים", "תסיים"]);
    if (hasAny(words, ["הקלט", "תקליט", "הקליטי", "הקלטה", "להקליט"])) {
      if (stopWords) return ready({ type: "recstop" }, "עצירת ההקלטה ושמירתה");
      const r = findRole(words);
      return ready({ type: "rec", role: r }, r ? `התחלת הקלטה לערוץ ${ROLE_HE[r]}` : "התחלת הקלטה");
    }
    if (findRole(words) === null) {
      if (hasAny(words, ["נגן", "תנגן", "נגני", "תשמיע", "השמע"])) return ready({ type: "play" }, "נגינה");
      if (stopWords && words.length <= 3) return ready({ type: "stop" }, "עצירה");
    }
    const fxi = effects(words, text, p);
    if (fxi) return fxi;
    // ---- export to an audio file (stage 7, basic) ----
    if (!hasAny(words, ["מאסטר", "מאסטרינג", "סטמים"]) && (hasAny(words, ["ייצא", "תייצא", "לייצא", "ייצוא", "ייצאי", "הורדה", "להורדה", "wav", "mp3"]) || containsPhrase(text, ["הורד את השיר", "תוריד את השיר", "תוריד לי את השיר", "שמור כקובץ", "קובץ שמע", "לקובץ שמע"])))
      return ready({ type: "export", confirmed: false }, "ייצוא השיר לקובץ שמע");
    const laterHit = LATER.find(([k]) => hasAny(words, k));
    const later = laterHit ? laterHit[1] : null;
    if (hasAny(words, ["זמזם", "תזמזם", "זמזמי"])) return { kind: "unsupported", msg: 'כדי לזמזם: לחץ על כפתור המיקרופון (או כתוב "תקליט"), זמזם, ואז "הפוך לתווים"' };
    if (hasAny(words, ["הוסף", "תוסיף", "הוסיפי", "תוסיפי", "צרף"])) {
      if (later) return { kind: "unsupported", msg: later };
      const r = findRole(words);
      if (r) return ready({ type: "add", role: r }, `הוספת ערוץ ${ROLE_HE[r]} ריק (אפשר לבקש שינגן לפי השיר)`);
      return { kind: "ask", msg: "איזה כלי להוסיף? למשל: גיטרה, תופים, בס, פסנתר, כינור, חליל, עוד, דרבוקה" };
    }
    const role = findRole(words);
    if (role === null) return { kind: "unsupported", msg: later || GENERIC };
    const rt = resolveTrack(p, role);
    if (rt.err) return rt.err;
    const trackId = rt.id;
    const name = p.tracks.find((t) => t.id === trackId).name;
    const sectionScoped = hasAny(words, SECTION_WORDS);
    const wrap = (action, summary) =>
      sectionScoped
        ? { kind: "confirm", proposed: { type: "action", action }, msg: `${summary} — בכל השיר. שינוי רק בקטע מסוים יגיע עם האוטומציה (שלב 6). לבצע על כל השיר?` }
        : ready({ type: "action", action }, summary);
    if (mentionsMute) {
      const unmute = hasAny(words, ["בטל", "תבטל", "הסר", "תחזיר", "החזר"]);
      return wrap({ a: "set_mute", track_id: trackId, mute: !unmute }, `${unmute ? "ביטול" : ""} השתקה של ${name}`.trim());
    }
    if (mentionsSolo) {
      const off = hasAny(words, ["בטל", "תבטל", "הסר"]);
      return wrap({ a: "set_solo", track_id: trackId, solo: !off }, `${off ? "ביטול" : ""} סולו ל${name}`.trim());
    }
    if (hasAny(words, ["שמאלה", "שמאל"]) || hasAny(words, ["ימינה", "ימין"]) || hasAny(words, ["מרכז", "אמצע"])) {
      const amount = hasAny(words, ["קצת", "מעט", "טיפה"]) ? 0.3 : hasAny(words, ["לגמרי", "מלא"]) ? 1.0 : 0.5;
      let pan, label;
      if (hasAny(words, ["שמאלה", "שמאל"])) { pan = -amount; label = "שמאלה"; }
      else if (hasAny(words, ["ימינה", "ימין"])) { pan = amount; label = "ימינה"; }
      else { pan = 0; label = "למרכז"; }
      return wrap({ a: "set_pan", track_id: trackId, pan }, `הזזת ${name} ${label}`);
    }
    const down = hasAny(words, ["הנמך", "תנמיך", "הנמיכי", "הורד", "תוריד", "הורידי", "חלש", "שקט"]) || containsPhrase(text, ["פחות חזק", "יותר חלש", "יותר בשקט"]);
    const up = hasAny(words, ["הגבר", "תגביר", "הגבירי", "הרם", "תרים", "הרימי", "העלה", "תעלה", "חזק", "חזקה"]);
    if (down || up) {
      const n = findNumber(words);
      const amount = n !== null && n > 0 && n <= 60 ? n : hasAny(words, ["קצת", "מעט", "טיפה"]) ? 2 : hasAny(words, ["הרבה", "מאוד"]) ? 6 : 3;
      const delta = down ? -amount : amount;
      return wrap({ a: "gain_delta", track_id: trackId, delta }, `${down ? "הנמכת" : "הגברת"} ${name} ב-${amount} דציבל`);
    }
    if (later) return { kind: "unsupported", msg: later };
    return { kind: "unsupported", msg: `הבנתי שמדובר ב${name}, אבל לא מה לעשות. אפשר: להנמיך, להגביר, להשתיק, סולו, שמאלה/ימינה` };
  }

  // ---- basic effects (port of hebrew.rs `effects`) ----
  const REVERB_WORDS = ["ריוורב", "הד", "אקו", "חלל"];
  const COMP_WORDS = ["קומפרסור", "דחיסה", "קומפרסיה", "תדחוס", "דחוס"];
  const EQ_WORDS = ["אקולייזר", "איקיו", "eq"];
  const BRIGHT = ["יותר בהיר", "בהיר יותר", "יותר חד", "חד יותר", "יותר צלול", "צלול יותר", "תבהיר", "להבהיר"];
  const WARM = ["יותר חם", "חם יותר", "יותר עמום", "עמום יותר", "פחות חד", "יותר רך", "רך יותר", "תחמם", "לחמם"];
  const REMOVE = ["בלי", "ללא", "הסר", "תסיר", "הסירי", "בטל", "תבטל", "תוריד", "הורד", "הורידי", "תעיף", "אפס", "תאפס"];
  function fxHebrew(f) {
    const v = [];
    if (f.reverb > 0) v.push(`הד ${(f.reverb * 100).toFixed(0)}%`);
    if (f.comp > 0) v.push(`דחיסה ${(f.comp * 100).toFixed(0)}%`);
    for (const [x, n] of [[f.eq_low_db, "באסים"], [f.eq_mid_db, "אמצע"], [f.eq_high_db, "גבוהים"]]) if (x !== 0) v.push(`${n} ${x > 0 ? "+" : ""}${x.toFixed(1)}`);
    return v.length ? v.join(", ") : "ללא";
  }
  const pct = (v) => `${(v * 100).toFixed(0)}%`;
  function effects(words, text, p) {
    const rev = hasAny(words, REVERB_WORDS), comp = hasAny(words, COMP_WORDS);
    const bright = containsPhrase(text, BRIGHT), warm = containsPhrase(text, WARM), genericEq = hasAny(words, EQ_WORDS);
    if (!(rev || comp || bright || warm || genericEq)) return null;
    const role = findRole(words);
    if (role === null) {
      const what = rev ? "הד" : comp ? "דחיסה" : "שינוי צליל";
      return { kind: "ask", msg: `לאיזה ערוץ לעשות ${what}? למשל: "תוסיף הד לגיטרה"` };
    }
    const rt = resolveTrack(p, role);
    if (rt.err) return rt.err;
    const t = p.tracks.find((x) => x.id === rt.id);
    const name = t.name;
    const fx = { ...(t.fx || FX_OFF) };
    const less = hasAny(words, ["פחות"]);
    const remove = !less && hasAny(words, REMOVE);
    const little = hasAny(words, ["קצת", "מעט", "טיפה"]);
    const lots = hasAny(words, ["הרבה", "מאוד"]);
    const more = hasAny(words, ["יותר"]);
    const step = (cur, base) => remove ? 0 : less ? Math.max(cur - (little ? 0.1 : 0.2), 0) : more && cur > 0 ? Math.min(cur + (little ? 0.1 : 0.2), 1) : little ? base * 0.6 : lots ? Math.min(base * 1.8, 1) : base;
    let summary;
    if (rev) {
      const before = fx.reverb;
      fx.reverb = Math.round(step(before, 0.35) * 100) / 100;
      summary = fx.reverb === 0 ? `הסרת ההד מערוץ ${name}` : fx.reverb < before ? `פחות הד לערוץ ${name} (${pct(fx.reverb)})` : `הד לערוץ ${name} (${pct(fx.reverb)})`;
    } else if (comp) {
      const before = fx.comp;
      fx.comp = Math.round(step(before, 0.5) * 100) / 100;
      summary = fx.comp === 0 ? `הסרת הדחיסה מערוץ ${name}` : fx.comp < before ? `פחות דחיסה לערוץ ${name} (${pct(fx.comp)})` : `דחיסה לערוץ ${name} (${pct(fx.comp)})`;
    } else if (bright || warm) {
      const amt = little ? 2 : lots ? 6 : 3;
      if (bright) {
        fx.eq_high_db = Math.min(fx.eq_high_db + amt, FX_EQ_RANGE);
        summary = `צליל בהיר יותר לערוץ ${name} (+${amt.toFixed(0)} דציבל בגבוהים)`;
      } else {
        fx.eq_high_db = Math.max(fx.eq_high_db - amt, -FX_EQ_RANGE);
        const low = Math.round(amt * 2 / 3 * 2) / 2;
        fx.eq_low_db = Math.min(fx.eq_low_db + low, FX_EQ_RANGE);
        summary = `צליל חם יותר לערוץ ${name} (-${amt.toFixed(0)} דציבל בגבוהים, +${low} בבאסים)`;
      }
    } else if (remove) {
      fx.eq_low_db = 0; fx.eq_mid_db = 0; fx.eq_high_db = 0;
      summary = `איפוס האקולייזר של ${name}`;
    } else {
      return { kind: "ask", msg: `איך לשנות את הצליל של ${name}? למשל: "בהיר יותר" או "חם יותר"` };
    }
    const action = { a: "set_fx", track_id: rt.id, fx };
    return hasAny(words, SECTION_WORDS)
      ? { kind: "confirm", proposed: { type: "action", action }, msg: `${summary} — בכל השיר. שינוי רק בקטע מסוים יגיע עם האוטומציה (שלב 6). לבצע על כל השיר?` }
      : { kind: "ready", proposed: { type: "action", action }, msg: summary };
  }

  const STYLE_HE = { pop: "פופ", rock: "רוק", mediterranean: "ים תיכוני", acoustic: "אקוסטי", ballad: "בלדה" };
  function findStyle(words, text) {
    if (containsPhrase(text, ["ים תיכוני", "ים תיכונית", "יםתיכוני"]) || hasAny(words, ["מזרחי", "מזרחית", "ים-תיכוני", "מזרח-תיכוני"])) return "mediterranean";
    const t = [["pop", ["פופ", "פופי"]], ["rock", ["רוק", "רוקי", "רוקיסטי"]], ["acoustic", ["אקוסטי", "אקוסטית", "אקוסטיק"]], ["ballad", ["בלדה", "בלדות"]]];
    const f = t.find(([, k]) => hasAny(words, k));
    return f ? f[0] : null;
  }
  // port of hebrew.rs solo_ask: "סולו גיטרה", "גיטרה לבד" — the instrument named next to the word, and where
  function soloAsk(words) {
    const at = words.findIndex((w) => stems(w).some((s) => ["סולו", "סולואים", "לבד", "אינסטרומנטלי"].includes(s)));
    if (at < 0) return null;
    const MEL = ["guitar", "piano", "strings", "winds"];
    const melodic = (w) => { const f = ROLE_WORDS.find(([r, k]) => MEL.includes(r) && hasAny([w], k)); return f ? f[0] : null; };
    let role = null;
    for (const i of [at - 1, at + 1, at + 2]) { if (i < 0 || i >= words.length) continue; const r = melodic(words[i]); if (r) { role = r; break; } }
    if (!role) role = allRoles(words).find((r) => MEL.includes(r)) || null;
    return { role, at: hasAny(words, ["בסוף", "סוף", "סיום", "מסיים", "לסיום"]) ? "end" : "chorus2" };
  }
  const allRoles = (words) => ROLE_WORDS.filter(([r, k]) => hasAny(words, k) && ["drums", "bass", "guitar", "piano", "strings", "winds"].includes(r)).map(([r]) => r);
  function canon(pr) {
    if (["undo", "redo", "play", "stop", "recstop", "transcribe", "quantize", "recommend_key"].includes(pr.type)) return pr.type;
    if (pr.type.startsWith("vocal_")) return pr.type;
    if (pr.type === "existing") return `existing:${pr.title || "-"}:${pr.artist || "-"}`;
    if (pr.type === "export") return `export:${pr.confirmed}`;
    if (pr.type === "rec") return `rec:${pr.role || "-"}`;
    if (pr.type === "add") return `add:${pr.role}`;
    if (pr.type === "arrange") return `arrange:${pr.style || "-"}:${pr.roles.join(",")}:${pr.only || "-"}:${pr.new_harmony}`;
    if (pr.type === "compose") return `compose:${pr.style || "-"}:${pr.major === null ? "-" : pr.major}:${pr.bpm === null ? "-" : pr.bpm}:${pr.roles.join(",")}${pr.solo ? `:solo=${pr.solo.role || "-"}@${pr.solo.at}` : ""}${pr.lift ? `:lift=${pr.lift}` : ""}`;
    const a = pr.action;
    switch (a.a) {
      case "gain_delta": return `gain:${a.track_id}:${a.delta}`;
      case "set_mute": return `mute:${a.track_id}:${a.mute}`;
      case "set_solo": return `solo:${a.track_id}:${a.solo}`;
      case "set_pan": return `pan:${a.track_id}:${a.pan}`;
      case "set_tempo": return `tempo:${a.bpm}`;
      case "set_key": return `key:${a.key.tonic}:${a.key.mode}`;
      case "transpose": return `transpose:${a.semitones}`;
      case "set_rights": return `rights:${a.id}`;
      case "set_instrument": return `inst:${a.track_id}:${a.instrument}`;
      case "set_fx": return `fx:${a.track_id}:${a.fx.eq_low_db}:${a.fx.eq_mid_db}:${a.fx.eq_high_db}:${a.fx.comp}:${a.fx.reverb}`;
      default: return `other:${a.a}`;
    }
  }

  // ---------------- executor (port of command.rs) ----------------
  function validName(s) {
    const t = String(s).trim();
    if (!t || Array.from(t).length > 80 || /[\u0000-\u001f\u007f]/.test(t)) throw "שם לא תקין (1 עד 80 תווים, ללא תווי בקרה)";
    return t;
  }
  function track(p, id) { const t = p.tracks.find((x) => x.id === id); if (!t) throw `לא נמצא ערוץ: ${id}`; return t; }
  // apply returns the inverse action; throws a Hebrew message on validation failure
  function validNotes(notes) {
    if (notes.length > 20000) throw "יותר מדי תווים בקליפ";
    for (const n of notes) if (!(n.pitch >= 0 && n.pitch <= 127 && n.velocity >= 1 && n.velocity <= 127 && n.start_tick >= 0 && n.dur_ticks > 0)) throw "תו לא תקין (גובה 0–127, עוצמה 1–127, אורך חיובי, לא לפני תחילת הקליפ)";
  }
  // the track an edit would change, when that edit is refused on a locked track (command.rs locked_target)
  function lockedTarget(p, a) {
    const midi = (id) => ((p.midi_clips || []).find((c) => c.id === id) || {}).track_id;
    const clip = (id) => ((p.clips || []).find((c) => c.id === id) || {}).track_id;
    const tid = { remove_track: a.track_id, set_instrument: a.track_id, add_midi_clip: a.clip && a.clip.track_id, add_clip: a.clip && a.clip.track_id, remove_midi_clip: midi(a.clip_id), set_midi_notes: midi(a.clip_id), set_midi_clip_start: midi(a.clip_id), remove_clip: clip(a.clip_id), set_clip_start: clip(a.clip_id), set_clip_edit: clip(a.clip_id) }[a.a];
    const t = tid && p.tracks.find((x) => x.id === tid);
    return t && t.locked ? t.name : null;
  }
  const validColor = (c) => typeof c === "string" && /^#[0-9a-fA-F]{6}$/.test(c);
  function apply(p, a) {
    const lockedName = lockedTarget(p, a);
    if (lockedName) throw `הערוץ ${lockedName} נעול. אפשר לבטל את הנעילה ואז לערוך`;
    switch (a.a) {
      case "rename_track": { const t = track(p, a.track_id); const old = t.name; t.name = validName(a.name); return { a: "rename_track", track_id: a.track_id, name: old }; }
      case "set_track_color": {
        if (a.color !== null && !validColor(a.color)) throw "צבע לא תקין";
        const t = track(p, a.track_id); const old = t.color ?? null; t.color = a.color; return { a: "set_track_color", track_id: a.track_id, color: old };
      }
      case "set_track_lock": { const t = track(p, a.track_id); const old = !!t.locked; t.locked = !!a.locked; return { a: "set_track_lock", track_id: a.track_id, locked: old }; }
      case "set_track_inserts": { // effects.rs: the whole chain at once, validated against the effects table
        const t = track(p, a.track_id), E = root.HUMEffects;
        if (!Array.isArray(a.inserts) || a.inserts.length > (E ? E.MAX_INSERTS : 8)) throw "עד 8 אפקטים בערוץ";
        const next = E ? a.inserts.map((x) => E.normalize(x)) : a.inserts;
        const old = JSON.parse(JSON.stringify(t.inserts || [])); t.inserts = next;
        return { a: "set_track_inserts", track_id: a.track_id, inserts: old };
      }
      case "set_track_delay_send": {
        if (!(a.level >= 0 && a.level <= 1)) throw "שליחה בין 0 ל-1";
        const t = track(p, a.track_id), old = t.delay_send || 0; t.delay_send = a.level;
        return { a: "set_track_delay_send", track_id: a.track_id, level: old };
      }
      case "set_marker": { // command.rs SetMarker: same id = change
        const m = a.marker;
        if (!m || !(Number.isInteger(m.tick) && m.tick >= 0) || !m.id) throw "מקום הסמן לא תקין";
        const mk = { id: m.id, tick: m.tick, name: validName(m.name) };
        p.markers = p.markers || [];
        const i = p.markers.findIndex((x) => x.id === mk.id);
        if (i >= 0) { const old = p.markers[i]; p.markers[i] = mk; p.markers.sort((x, y) => x.tick - y.tick); return { a: "set_marker", marker: old }; }
        if (p.markers.length >= 200) throw "יותר מדי סמנים";
        p.markers.push(mk); p.markers.sort((x, y) => x.tick - y.tick);
        return { a: "remove_marker", id: mk.id };
      }
      case "remove_marker": {
        const i = (p.markers || []).findIndex((x) => x.id === a.id);
        if (i < 0) throw "הסמן לא נמצא";
        const [m] = p.markers.splice(i, 1); return { a: "set_marker", marker: m };
      }
      case "set_track_automation": {
        const pts = a.points || [];
        if (pts.length > 2000 || pts.some(([t, d]) => !(Number.isInteger(t) && t >= 0) || !(d >= -60 && d <= 12))) throw "נקודות אוטומציה לא תקינות";
        const t = track(p, a.track_id), old = t.automation || [];
        t.automation = pts.map((x) => [x[0], x[1]]).sort((x, y) => x[0] - y[0]);
        return { a: "set_track_automation", track_id: a.track_id, points: old };
      }
      case "set_track_arm": { const t = track(p, a.track_id); const old = !!t.armed; t.armed = !!a.armed; return { a: "set_track_arm", track_id: a.track_id, armed: old }; }
      case "set_track_kind": { // command.rs SetTrackKind: only while the track holds no clips
        if (!["audio", "midi", "bus"].includes(a.kind)) throw "סוג ערוץ לא מוכר";
        const t = track(p, a.track_id);
        const has = (p.clips || []).some((c) => c.track_id === a.track_id) || (p.midi_clips || []).some((c) => c.track_id === a.track_id);
        if (t.kind !== a.kind && has) throw "אפשר לשנות סוג רק לערוץ ריק";
        const old = t.kind; t.kind = a.kind; return { a: "set_track_kind", track_id: a.track_id, kind: old };
      }
      case "move_track": {
        const i = p.tracks.findIndex((t) => t.id === a.track_id);
        if (i < 0) throw `לא נמצא ערוץ: ${a.track_id}`;
        const [t] = p.tracks.splice(i, 1);
        p.tracks.splice(Math.min(a.to_index, p.tracks.length), 0, t);
        return { a: "move_track", track_id: a.track_id, to_index: i };
      }
      case "add_track": {
        if (p.tracks.length >= MAX_TRACKS) throw `אפשר עד ${MAX_TRACKS} ערוצים`;
        p.tracks.push({ id: a.track_id, name: a.name ? validName(a.name) : ROLE_HE[a.role], role: a.role, kind: defaultKind(a.role), gain_db: 0, pan: 0, mute: false, solo: false, fx: { ...FX_OFF }, instrument: "auto", color: null, locked: false, armed: false, inserts: [], delay_send: 0, automation: [], sends: [], output: null, group: false });
        return { a: "remove_track", track_id: a.track_id };
      }
      case "remove_track": {
        const i = p.tracks.findIndex((t) => t.id === a.track_id);
        if (i < 0) throw `לא נמצא ערוץ: ${a.track_id}`;
        const [t] = p.tracks.splice(i, 1);
        return { a: "restore_track", index: i, track: t };
      }
      case "restore_track": p.tracks.splice(Math.min(a.index, p.tracks.length), 0, { ...a.track }); return { a: "remove_track", track_id: a.track.id };
      case "gain_delta":
      case "gain_set": {
        const t = track(p, a.track_id);
        const target = a.a === "gain_set" ? a.db : t.gain_db + a.delta;
        if (!Number.isFinite(target) || target < MIN_GAIN || target > MAX_GAIN) throw `עוצמה חייבת להיות בין ${MIN_GAIN} ל-+${MAX_GAIN} דציבל (התבקש ${target.toFixed(1)})`;
        const old = t.gain_db; t.gain_db = target; return { a: "gain_set", track_id: a.track_id, db: old };
      }
      case "set_pan": {
        if (!Number.isFinite(a.pan) || a.pan < -1 || a.pan > 1) throw "מיקום סטריאו חייב להיות בין שמאל מלא לימין מלא";
        const t = track(p, a.track_id); const old = t.pan; t.pan = a.pan; return { a: "set_pan", track_id: a.track_id, pan: old };
      }
      case "set_mute": { const t = track(p, a.track_id); const old = t.mute; t.mute = a.mute; return { a: "set_mute", track_id: a.track_id, mute: old }; }
      case "set_solo": { const t = track(p, a.track_id); const old = t.solo; t.solo = a.solo; return { a: "set_solo", track_id: a.track_id, solo: old }; }
      case "set_fx": {
        const f = a.fx;
        for (const k of ["eq_low_db", "eq_mid_db", "eq_high_db"]) if (!Number.isFinite(f[k]) || Math.abs(f[k]) > FX_EQ_RANGE) throw "אקולייזר חייב להיות בין -12 ל-+12 דציבל";
        for (const k of ["comp", "reverb"]) if (!Number.isFinite(f[k]) || f[k] < 0 || f[k] > 1) throw "ערך אפקט חייב להיות בין 0 ל-100%";
        const t = track(p, a.track_id); const old = { ...(t.fx || FX_OFF) }; t.fx = { ...f }; return { a: "set_fx", track_id: a.track_id, fx: old };
      }
      case "set_clip_media": { // command.rs SetClipMedia: the old file stays (undo goes back to it)
        p.media = p.media || [];
        if (a.media) { if (a.media.id !== a.media_id) throw "קובץ שמע לא תקין"; if (!p.media.some((m) => m.id === a.media.id)) p.media.push({ ...a.media }); }
        else if (!p.media.some((m) => m.id === a.media_id)) throw "קובץ השמע לא נמצא";
        const c = (p.clips || []).find((x) => x.id === a.clip_id); if (!c) throw "ההקלטה לא נמצאה";
        const old = c.media_id; c.media_id = a.media_id; return { a: "set_clip_media", clip_id: a.clip_id, media_id: old };
      }
      case "set_track_group": { const t = track(p, a.track_id); if (a.group && t.kind !== "bus") throw "רק ערוץ קבוצה יכול להיות תיקייה"; const old = !!t.group; t.group = !!a.group; return { a: "set_track_group", track_id: a.track_id, group: old }; }
      case "set_track_output": { // command.rs SetTrackOutput
        const t = track(p, a.track_id);
        if (a.bus_id) { const b = p.tracks.find((x) => x.id === a.bus_id); if (a.bus_id === a.track_id || !b || b.kind !== "bus" || !b.group || t.kind === "bus") throw "מנתבים ערוץ רגיל לתיקייה או לערוץ קבוצה"; }
        const old = t.output || null; t.output = a.bus_id || null; return { a: "set_track_output", track_id: a.track_id, bus_id: old };
      }
      case "set_track_send": { // command.rs SetTrackSend
        if (!(a.level >= 0 && a.level <= 1)) throw "שליחה בין 0 ל-1";
        const b = p.tracks.find((x) => x.id === a.bus_id);
        if (a.track_id === a.bus_id || !b || b.kind !== "bus" || b.group) throw "שולחים רק לערוץ אפקט";
        const t = track(p, a.track_id), cur = t.sends || [], old = (cur.find((x) => x[0] === a.bus_id) || [0, 0])[1];
        t.sends = cur.filter((x) => x[0] !== a.bus_id); if (a.level > 0) t.sends.push([a.bus_id, a.level]);
        return { a: "set_track_send", track_id: a.track_id, bus_id: a.bus_id, level: old };
      }
      case "set_master_gain": { // command.rs SetMasterGain
        if (!(a.db >= -60 && a.db <= 6)) throw "עוצמה ראשית בין ‎-60 ל-‎+6 dB";
        const old = p.master_db || 0; p.master_db = a.db; return { a: "set_master_gain", db: old };
      }
      case "set_meter": { // command.rs SetMeter
        const ok = [[2, 4], [3, 4], [4, 4], [5, 4], [6, 8], [7, 8], [9, 8], [12, 8]].some(([n, d]) => n === a.numerator && d === a.denominator);
        if (!ok) throw "משקל לא נתמך";
        const old = p.meter || [4, 4]; p.meter = [a.numerator, a.denominator]; return { a: "set_meter", numerator: old[0], denominator: old[1] };
      }
      case "set_tempo": {
        if (!Number.isFinite(a.bpm) || a.bpm < 20 || a.bpm > 400) throw "קצב חייב להיות בין 20 ל-400";
        const old = p.bpm; p.bpm = a.bpm; return { a: "set_tempo", bpm: old };
      }
      case "set_key": { const old = p.key; p.key = { ...a.key }; return { a: "set_key", key: old }; }
      case "rename_project": { const old = p.name; p.name = validName(a.name); return { a: "rename_project", name: old }; }
      case "add_clip": {
        const tt = p.tracks.find((t) => t.id === a.clip.track_id);
        if (!tt) throw `לא נמצא ערוץ: ${a.clip.track_id}`;
        if (tt.kind === "bus") throw "בערוץ אפקט אין הקלטות";
        p.clips = p.clips || []; p.media = p.media || [];
        if (a.media && !p.media.some((m) => m.id === a.media.id)) p.media.push({ ...a.media });
        p.clips.push({ ...a.clip });
        return { a: "remove_clip", clip_id: a.clip.id };
      }
      case "remove_clip": {
        const i = (p.clips || []).findIndex((c) => c.id === a.clip_id);
        if (i < 0) throw "הקליפ לא נמצא";
        const [clip] = p.clips.splice(i, 1);
        return { a: "restore_clip", index: i, clip };
      }
      case "restore_clip": { p.clips = p.clips || []; p.clips.splice(Math.min(a.index, p.clips.length), 0, { ...a.clip }); return { a: "remove_clip", clip_id: a.clip.id }; }
      case "add_midi_clip": {
        const t = p.tracks.find((x) => x.id === a.clip.track_id);
        if (!t) throw `לא נמצא ערוץ: ${a.clip.track_id}`;
        if (t.kind !== "midi") throw "אפשר לשים תווים רק בערוץ של כלי נגינה";
        validNotes(a.clip.notes);
        p.midi_clips = p.midi_clips || [];
        p.midi_clips.push(JSON.parse(JSON.stringify(a.clip)));
        return { a: "remove_midi_clip", clip_id: a.clip.id };
      }
      case "remove_midi_clip": {
        const i = (p.midi_clips || []).findIndex((c) => c.id === a.clip_id);
        if (i < 0) throw "קליפ התווים לא נמצא";
        const [clip] = p.midi_clips.splice(i, 1);
        return { a: "restore_midi_clip", index: i, clip };
      }
      case "restore_midi_clip": { p.midi_clips = p.midi_clips || []; p.midi_clips.splice(Math.min(a.index, p.midi_clips.length), 0, JSON.parse(JSON.stringify(a.clip))); return { a: "remove_midi_clip", clip_id: a.clip.id }; }
      case "set_clip_edit": {
        if (!(Number.isInteger(a.start_tick) && a.start_tick >= 0)) throw "אי אפשר להזיז לפני תחילת השיר";
        const e = a.edit || {};
        if (e.trim_len === 0 || !(e.trim_start >= 0) || (e.fade_in_ms || 0) > 60000 || (e.fade_out_ms || 0) > 60000 || (e.gain_cdb || 0) < -4800 || (e.gain_cdb || 0) > 2400) throw "קיצוץ או דהייה לא תקינים";
        const c = (p.clips || []).find((x) => x.id === a.clip_id);
        if (!c) throw "ההקלטה לא נמצאה";
        const old = { start_tick: c.start_tick, edit: { ...(c.edit || { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0 }) } };
        c.start_tick = a.start_tick; c.edit = { trim_start: e.trim_start | 0, trim_len: e.trim_len || null, fade_in_ms: e.fade_in_ms | 0, fade_out_ms: e.fade_out_ms | 0, mute: !!e.mute, gain_cdb: e.gain_cdb | 0 };
        return { a: "set_clip_edit", clip_id: a.clip_id, ...old };
      }
      case "set_clip_start": {
        if (!(Number.isInteger(a.start_tick) && a.start_tick >= 0)) throw "אי אפשר להזיז לפני תחילת השיר";
        const c = (p.clips || []).find((x) => x.id === a.clip_id);
        if (!c) throw "ההקלטה לא נמצאה";
        const old = c.start_tick; c.start_tick = a.start_tick;
        return { a: "set_clip_start", clip_id: a.clip_id, start_tick: old };
      }
      case "set_midi_clip_start": {
        if (!(Number.isInteger(a.start_tick) && a.start_tick >= 0)) throw "אי אפשר להזיז לפני תחילת השיר";
        const c = (p.midi_clips || []).find((x) => x.id === a.clip_id);
        if (!c) throw "קליפ התווים לא נמצא";
        const old = c.start_tick; c.start_tick = a.start_tick;
        return { a: "set_midi_clip_start", clip_id: a.clip_id, start_tick: old };
      }
      case "set_midi_notes": {
        validNotes(a.notes);
        const c = (p.midi_clips || []).find((x) => x.id === a.clip_id);
        if (!c) throw "קליפ התווים לא נמצא";
        const old = c.notes;
        c.notes = a.notes.map((n) => ({ ...n })).sort((x, y) => x.start_tick - y.start_tick || x.pitch - y.pitch);
        return { a: "set_midi_notes", clip_id: a.clip_id, notes: old };
      }
      case "set_clip_generator": {
        const c = (p.midi_clips || []).find((x) => x.id === a.clip_id);
        if (!c) throw "קליפ התווים לא נמצא";
        const old = c.generator === undefined ? null : c.generator; c.generator = a.generator;
        return { a: "set_clip_generator", clip_id: a.clip_id, generator: old };
      }
      case "set_harmony": {
        for (const c of a.harmony.chords) if (!(c.root >= 0 && c.root <= 11 && c.start_tick >= 0 && c.dur_ticks > 0 && ["maj", "min", "dim"].includes(c.quality))) throw "אקורד לא תקין";
        for (const x of a.harmony.sections) if (!(x.start_tick >= 0 && x.dur_ticks > 0 && ["intro", "verse", "chorus", "bridge", "outro", "pre_chorus", "solo", "break"].includes(x.kind))) throw "חלק בשיר לא תקין";
        const old = p.harmony || { style: null, chords: [], sections: [] };
        p.harmony = JSON.parse(JSON.stringify(a.harmony));
        return { a: "set_harmony", harmony: old };
      }
      case "shift_timeline": {
        const h = p.harmony || { chords: [], sections: [] };
        const starts = [...(p.clips || []), ...(p.midi_clips || []), ...h.chords, ...h.sections].map((x) => x.start_tick);
        if (starts.length && Math.min(...starts) + a.delta < 0) throw "אי אפשר להזיז לפני תחילת השיר";
        for (const x of [...(p.clips || []), ...(p.midi_clips || []), ...h.chords, ...h.sections]) x.start_tick += a.delta;
        return { a: "shift_timeline", delta: -a.delta };
      }
      case "set_rights": {
        const USER = ["own_work", "licensed", "public_domain", "unknown"];
        const r = a.record, short = (x, n) => x === null || x === undefined || Array.from(String(x)).length <= n;
        if (r) {
          if (r.id !== a.id || !USER.includes(r.user_statement) || !["midi_file", "audio_file"].includes(r.asset_type)) throw "רישום זכויות לא תקין";
          if (!short(r.source, 200) || !short(r.license_identifier, 200) || !short(r.title, 200) || !short(r.artist, 200) || (r.assets || []).length > 128) throw "פרטי הזכויות ארוכים מדי";
          if (r.verified_at && !r.evidence_uri) throw "אימות זכויות דורש מסמך או קישור כראיה";
        }
        p.rights = p.rights || [];
        const i = p.rights.findIndex((x) => x.id === a.id);
        let old = null;
        if (i >= 0 && r) { old = p.rights[i]; p.rights[i] = JSON.parse(JSON.stringify(r)); }
        else if (i >= 0) { old = p.rights.splice(i, 1)[0]; }
        else if (r) p.rights.push(JSON.parse(JSON.stringify(r)));
        else throw "רישום הזכויות לא נמצא";
        return { a: "set_rights", id: a.id, record: old };
      }
      case "transpose": {
        const st = a.semitones;
        if (!Number.isInteger(st) || st === 0 || Math.abs(st) > 24) throw "אפשר להזיז את הסולם עד שתי אוקטבות";
        const drums = p.tracks.filter((t) => resolveInst(t) === "drums").map((t) => t.id);
        const pitched = (p.midi_clips || []).filter((c) => !drums.includes(c.track_id));
        if (pitched.some((c) => c.notes.some((n) => n.pitch + st < 0 || n.pitch + st > 127))) throw "השינוי יוציא תווים מחוץ לטווח שאפשר לנגן";
        for (const c of pitched) for (const n of c.notes) n.pitch += st;
        const pc = (x) => (((x + st) % 12) + 12) % 12;
        for (const c of (p.harmony || { chords: [] }).chords) c.root = pc(c.root);
        p.key = { tonic: pc(p.key.tonic), mode: p.key.mode };
        return { a: "transpose", semitones: -st };
      }
      case "set_instrument": {
        if (!(a.instrument in INSTRUMENT_HE)) throw "כלי לא מוכר";
        const t = track(p, a.track_id); const old = t.instrument || "auto"; t.instrument = a.instrument;
        return { a: "set_instrument", track_id: a.track_id, instrument: old };
      }
      case "batch": {
        const backup = JSON.parse(JSON.stringify(p));
        const inv = [];
        try { for (const x of a.actions) inv.push(apply(p, x)); } catch (e) { Object.keys(p).forEach((k) => delete p[k]); Object.assign(p, backup); throw e; }
        return { a: "batch", label: a.label, actions: inv.reverse() };
      }
      case "restore_snapshot": {
        const s = a.state;
        if (s.tracks.length > MAX_TRACKS) throw `אפשר עד ${MAX_TRACKS} ערוצים`;
        const ids = new Set();
        for (const t of s.tracks) {
          validName(t.name);
          if (!t.id || ids.has(t.id)) throw "מזהי ערוצים כפולים בגרסה";
          ids.add(t.id);
          if (!(t.gain_db >= MIN_GAIN && t.gain_db <= MAX_GAIN) || !(t.pan >= -1 && t.pan <= 1)) throw "ערכי עוצמה או סטריאו לא תקינים בגרסה";
          const f = t.fx || FX_OFF;
          if ([f.eq_low_db, f.eq_mid_db, f.eq_high_db].some((x) => !(Math.abs(x) <= FX_EQ_RANGE)) || [f.comp, f.reverb].some((x) => !(x >= 0 && x <= 1))) throw "אפקטים לא תקינים בגרסה";
        }
        validName(s.name);
        if (!(s.bpm >= 20 && s.bpm <= 400)) throw "קצב חייב להיות בין 20 ל-400";
        const before = JSON.parse(JSON.stringify(p));
        p.name = s.name; p.bpm = s.bpm; p.key = { ...s.key }; p.tracks = JSON.parse(JSON.stringify(s.tracks));
        p.clips = JSON.parse(JSON.stringify(s.clips || [])); p.midi_clips = JSON.parse(JSON.stringify(s.midi_clips || []));
        p.harmony = JSON.parse(JSON.stringify(s.harmony || { style: null, chords: [], sections: [] }));
        p.media = p.media || [];
        for (const m of s.media || []) if (!p.media.some((x) => x.id === m.id)) p.media.push({ ...m });
        return { a: "restore_snapshot", label: `לפני שחזור "${a.label}"`, state: before };
      }
    }
    throw "פקודה לא מוכרת";
  }

  // ---------------- storage (browser) ----------------
  const LS = "hum-lab-v1";
  function loadAll() { try { return JSON.parse(localStorage.getItem(LS) || "{}"); } catch { return {}; } }
  function saveAll(db) { try { localStorage.setItem(LS, JSON.stringify(db)); return true; } catch { return false; } }
  let db = loadAll();
  db.projects = db.projects || {};
  db.snapshots = db.snapshots || {};
  let memoryOnly = !saveAll(db);

  let seq = 0;
  const newId = (pfx) => `${pfx}_${Date.now().toString(36)}${(seq++).toString(36)}`;

  // ---------------- session (port of apps/core) ----------------
  let session = null; // {project, undo:[], redo:[]}
  const INPUT_KEY = "hum-input-device";
  let monitorOn = false;
  function micConstraints() {
    let id = null; try { id = localStorage.getItem(INPUT_KEY); } catch { /* private window */ }
    return { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, ...(id ? { deviceId: { ideal: id } } : {}) } };
  }
  let pending = null;

  function persist() {
    if (!session) return true;
    db.projects[session.project.id] = { project: session.project, updated_at: new Date().toISOString() };
    return memoryOnly ? true : saveAll(db);
  }

  // ---------------- transport + playback (Web Audio) ----------------
  // Recordings and note clips play through Web Audio here, with the same effects chain as the
  // installed app (EQ → compressor → fader → pan, reverb send after the fader).
  const tr = { playing: false, metronome: true, clickPlay: false, startCtx: 0, offset: 0, ctx: null, timer: 0, nextBeat: 0, beatIndex: 0, nodes: [], meters: new Map(), master: null };
  const PPQ = 960;
  const bpmNow = () => (session ? session.project.bpm : 120);
  const t2s = (tick) => (tick * 60) / (bpmNow() * PPQ);
  const s2t = (sec) => Math.round((sec * bpmNow() * PPQ) / 60);
  function ensureCtx() {
    try { tr.ctx = tr.ctx || new (window.AudioContext || window.webkitAudioContext)(); } catch { tr.ctx = null; }
    if (tr.ctx && tr.ctx.resume) tr.ctx.resume();
    if (tr.ctx && root.HUMEffects && !tr.ctx.__humWorkletAsked) { tr.ctx.__humWorkletAsked = true; root.HUMEffects.prepare(tr.ctx).then(refreshPlayback); }
    return tr.ctx;
  }
  function nowSeconds() {
    if (!(tr.playing && tr.ctx)) return tr.offset;
    const now = tr.ctx.currentTime;
    if (tr.prev && now < tr.startCtx) return tr.prev.offset + (now - tr.prev.startCtx); // before a scheduled loop jump
    return tr.offset + (now - tr.startCtx);
  }
  // Loop region (as engine/audio-core TransportControl::set_loop): the jump back is scheduled ahead on the
  // audio clock, so the song continues from the loop start without a gap.
  function loopTick() {
    if (!tr.loop || !tr.playing) return;
    const now = tr.ctx.currentTime;
    if (tr.prev && now >= tr.startCtx) tr.prev = null;
    if (tr.prev) return; // a jump is already scheduled
    const pos = nowSeconds();
    let at = tr.startCtx + (tr.loop.end - tr.offset);
    if (pos >= tr.loop.end || pos < tr.loop.start - 0.5) at = now + 0.03; // outside the loop: go to its start now
    if (at > now + 0.15) return;
    const old = { nodes: tr.nodes, master: tr.master };
    for (const n of old.nodes) { try { n.stop(at); } catch { /* already stopped */ } }
    tr.prev = { offset: tr.offset, startCtx: tr.startCtx };
    tr.nodes = []; tr.offset = tr.loop.start; tr.startCtx = at;
    startSources();
    setTimeout(() => { try { old.master && old.master.disconnect(); } catch { /* fine */ } }, Math.max(0, (at - now) * 1000) + 300);
    const spb = unitSec(), beats = tr.offset / spb;
    tr.beatIndex = Math.ceil(beats - 1e-9); tr.nextBeat = at + (tr.beatIndex - beats) * spb;
  }
  // the metronome counts the meter's unit (quarters in x/4, eighths in x/8); tempo is always quarter notes per minute
  const meterNow = () => (session && session.project.meter) || [4, 4];
  const unitSec = () => (60 / bpmNow()) * (4 / meterNow()[1]);
  function click(time, accent) {
    const o = tr.ctx.createOscillator(), g = tr.ctx.createGain();
    o.frequency.value = accent ? 1500 : 1000;
    g.gain.setValueAtTime(accent ? 0.5 : 0.32, time);
    g.gain.exponentialRampToValueAtTime(0.0008, time + 0.08);
    o.connect(g).connect(tr.ctx.destination); o.start(time); o.stop(time + 0.09);
  }
  function schedule() {
    loopTick();
    const spb = unitSec();
    while (tr.nextBeat < tr.ctx.currentTime + 0.12) {
      // the click guides a recording; in the studio the metronome button also lets it sound while playing (studio architecture §3.1)
      if ((tr.metronome && rec && rec.started) || tr.clickPlay) click(tr.nextBeat, tr.beatIndex % meterNow()[0] === 0);
      tr.nextBeat += spb; tr.beatIndex++;
    }
  }
  // --- audio data (recordings: memory + IndexedDB when the browser allows it) ---
  const audioStore = new Map(); // media id -> {pcm: Float32Array (mono), sr}
  const rendered = new Map(); // midi clip id -> {key, buf}
  function idb() {
    return new Promise((res) => {
      try {
        const r = indexedDB.open("hum-lab-audio", 1);
        r.onupgradeneeded = () => r.result.createObjectStore("media");
        r.onsuccess = () => res(r.result);
        r.onerror = () => res(null);
      } catch { res(null); }
    });
  }
  async function idbPut(id, v) { try { const d = await idb(); if (!d) return false; return await new Promise((res) => { const t = d.transaction("media", "readwrite"); t.objectStore("media").put(v, id); t.oncomplete = () => res(true); t.onerror = () => res(false); }); } catch { return false; } }
  async function idbGet(id) { try { const d = await idb(); if (!d) return null; return await new Promise((res) => { const q = d.transaction("media").objectStore("media").get(id); q.onsuccess = () => res(q.result || null); q.onerror = () => res(null); }); } catch { return null; } }
  async function loadMedia(p) {
    for (const m of p.media || []) if (!audioStore.has(m.id)) { const v = await idbGet(m.id); if (v) audioStore.set(m.id, v); }
  }
  function peaksOf(pcm, sr) {
    const bin = Math.max(1, Math.floor(sr / 20)), out = [];
    for (let i = 0; i < pcm.length; i += bin) { let m = 0; for (let k = i; k < Math.min(i + bin, pcm.length); k++) m = Math.max(m, Math.abs(pcm[k])); out.push(Math.round(Math.min(1, m) * 1000) / 1000); }
    return out;
  }
  const resolveInst = (t) => {
    const i = t.instrument || "auto";
    if (i !== "auto") return i;
    return { strings: "strings", winds: "flute", synth: "soft_lead", lead: "soft_lead", piano: "piano", bass: "bass", guitar: "guitar", middle_eastern: "guitar", drums: "drums" }[t.role] || "epiano";
  };
  // recorded instrument for a track (same rule as apps/core sampled_for): guitar is electric in rock
  function sampledFor(t) {
    const S = root.HUMSamples;
    if (!S || S.state !== "ready") return null;
    const inst = resolveInst(t), rock = session && session.project.harmony && session.project.harmony.style === "rock";
    const id = inst === "guitar" ? (rock && S.sets.electric_guitar ? "electric_guitar" : "guitar") : inst;
    if (S.sets[id]) return id;
    // an instrument that loads on first use: start decoding it; the synth plays until it is ready (then it re-renders)
    if (S.has && S.has(id)) S.ensure(id);
    return null;
  }
  function midiBuffer(t, c) {
    const ctx = tr.ctx, sr = ctx.sampleRate, patch = synthPatch(resolveInst(t)), sid = sampledFor(t);
    const key = JSON.stringify([patch, sid, sr, bpmNow(), c.notes]);
    const hit = rendered.get(c.id);
    if (hit && hit.key === key) return hit.buf;
    const notes = c.notes.map((n) => ({ start: Math.round(t2s(n.start_tick) * sr), len: Math.max(1, Math.round(t2s(n.dur_ticks) * sr)), pitch: n.pitch, velocity: n.velocity }));
    let r;
    if (sid) {
      r = root.HUMSamples.renderSampled(notes, root.HUMSamples.sets[sid], sr);
      if (r.rest.length) {
        const x = root.HUMSynth.render(r.rest, patch, sr);
        if (x.left.length > r.left.length) { const L = new Float32Array(x.left.length), R = new Float32Array(x.left.length); L.set(r.left); R.set(r.right); r = { left: L, right: R }; }
        for (let i = 0; i < x.left.length; i++) { r.left[i] += x.left[i]; r.right[i] += x.right[i]; }
      }
    } else r = root.HUMSynth.render(notes, patch, sr);
    if (!r.left.length) return null;
    const buf = ctx.createBuffer(2, r.left.length, sr);
    buf.copyToChannel(r.left, 0); buf.copyToChannel(r.right, 1);
    rendered.set(c.id, { key, buf });
    return buf;
  }
  // a recording as the clip plays it: trimmed and faded copy, cached (as apps/core audio.rs build_graph); the media is untouched
  const clipCache = new Map();
  function clipBuffer(p, c) {
    const m = (p.media || []).find((x) => x.id === c.media_id) || {};
    const buf = audioBuffer(m), e = c.edit || {};
    if (!buf || !(e.trim_start || e.trim_len || e.fade_in_ms || e.fade_out_ms || e.gain_cdb)) return buf;
    const off = Math.min(buf.length, e.trim_start || 0), end = e.trim_len ? Math.min(buf.length, off + e.trim_len) : buf.length;
    const fi = Math.floor(((e.fade_in_ms || 0) * buf.sampleRate) / 1000), fo = Math.floor(((e.fade_out_ms || 0) * buf.sampleRate) / 1000);
    const key = `${m.id}@${off}-${end}-${fi}-${fo}-${e.gain_cdb || 0}`, cg = Math.pow(10, (e.gain_cdb || 0) / 2000);
    if (clipCache.has(key)) return clipCache.get(key);
    const n = Math.max(1, end - off), out = tr.ctx.createBuffer(buf.numberOfChannels, n, buf.sampleRate);
    for (let ch = 0; ch < buf.numberOfChannels; ch++) {
      const src = buf.getChannelData(ch), dst = out.getChannelData(ch);
      for (let i = 0; i < n; i++) {
        let g = cg;
        if (i < fi) g *= i / fi;
        if (fo > 0 && i + fo > n) g *= (n - i) / fo;
        dst[i] = src[off + i] * g;
      }
    }
    clipCache.set(key, out);
    return out;
  }
  function audioBuffer(m) {
    const a = audioStore.get(m.id);
    if (!a) return null;
    if (!a.buf) { a.buf = tr.ctx.createBuffer(1, a.pcm.length, a.sr); a.buf.copyToChannel(a.pcm, 0); }
    return a.buf;
  }
  // ---- effects, the same chain as engine/audio-core/src/fx.rs + mixer.rs: EQ → compressor before the
  // fader, reverb send after it. The reverb is Freeverb (linear), so its impulse response, computed with the
  // same delay lengths, reproduces it exactly through a convolver.
  const irCache = new Map();
  function reverbIR(ctx) {
    const sr = ctx.sampleRate;
    if (irCache.has(sr)) return irCache.get(sr);
    const scale = sr / 44100, len = (n, side) => Math.max(1, Math.floor((n + side * 23) * scale));
    const N = Math.ceil(2.5 * sr), ir = ctx.createBuffer(2, N, sr);
    for (let side = 0; side < 2; side++) {
      const combs = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617].map((n) => ({ buf: new Float32Array(len(n, side)), i: 0, store: 0 }));
      const aps = [556, 441, 341, 225].map((n) => ({ buf: new Float32Array(len(n, side)), i: 0 }));
      const out = ir.getChannelData(side);
      for (let t = 0; t < N; t++) {
        const x = t === 0 ? 1 : 0; let acc = 0;
        for (const c of combs) { const y = c.buf[c.i]; c.store = y * 0.8 + c.store * 0.2; c.buf[c.i] = x + c.store * 0.84; if (++c.i === c.buf.length) c.i = 0; acc += y; }
        for (const a of aps) { const b = a.buf[a.i], y = b - acc; a.buf[a.i] = acc + b * 0.5; if (++a.i === a.buf.length) a.i = 0; acc = y; }
        out[t] = acc * 0.03; // input gain 0.015 on (L + R); the convolver is fed the mono mix (L + R) / 2
      }
    }
    irCache.set(sr, ir);
    return ir;
  }
  // the node a track's sources play into; `post` is the track's output after gain and pan
  function fxInput(ctx, t, gainNode, post, reverbBus) {
    const f = t.fx || FX_OFF;
    let head = gainNode;
    const chain = [];
    if (f.eq_low_db || f.eq_mid_db || f.eq_high_db) {
      const mk = (type, hz, db, q) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = hz; b.gain.value = db; if (q) b.Q.value = q; return b; };
      chain.push(mk("lowshelf", 120, f.eq_low_db), mk("peaking", 1000, f.eq_mid_db, 0.9), mk("highshelf", 8000, f.eq_high_db));
    }
    if (f.comp > 0) {
      const a = Math.min(1, f.comp), c = ctx.createDynamicsCompressor(), thr = -8 - 22 * a, ratio = 1.5 + 4.5 * a;
      c.threshold.value = thr; c.ratio.value = ratio; c.knee.value = 0; c.attack.value = 0.01; c.release.value = 0.12;
      const mu = ctx.createGain(); mu.gain.value = Math.pow(10, (-thr * (1 - 1 / ratio) * 0.5) / 20);
      c.connect(mu); chain.push([c, mu]);
    }
    if (chain.length) {
      const first = Array.isArray(chain[0]) ? chain[0][0] : chain[0];
      let prev = null;
      for (const n of chain) { const inN = Array.isArray(n) ? n[0] : n, outN = Array.isArray(n) ? n[1] : n; if (prev) prev.connect(inN); prev = outN; }
      prev.connect(gainNode); head = first;
    }
    if (f.reverb > 0 && reverbBus) { const s = ctx.createGain(); s.gain.value = Math.min(1, f.reverb) * 0.6; post.connect(s); s.connect(reverbBus); }
    return head;
  }
  // a track's inserts (effects.js) in front of its channel strip; returns the node its sources play into
  function trackInput(ctx, t, g, out, rbus, dbus) {
    const head = fxInput(ctx, t, g, out, rbus);
    if (dbus && t.delay_send > 0) { const s = ctx.createGain(); s.gain.value = Math.min(1, t.delay_send); out.connect(s); s.connect(dbus); }
    if (!root.HUMEffects || !(t.inserts && t.inserts.some((x) => x.on !== false))) return head;
    const c = root.HUMEffects.chain(ctx, t.inserts, bpmNow());
    c.output.connect(head);
    return c.input;
  }
  // the shared echo (delay send): a quarter note with a few repeats, darker each time
  function delayBusFor(ctx, dest) {
    const bus = ctx.createGain(), d = ctx.createDelay(5), fb = ctx.createGain(), lp = ctx.createBiquadFilter();
    d.delayTime.value = Math.min(4.9, 60 / bpmNow()); fb.gain.value = 0.35; lp.type = "lowpass"; lp.frequency.value = 5000;
    bus.connect(d); d.connect(lp); lp.connect(fb); fb.connect(d); lp.connect(dest);
    return bus;
  }
  function reverbBusFor(ctx, dest) {
    const bus = ctx.createGain(); bus.channelCount = 1; bus.channelCountMode = "explicit"; bus.channelInterpretation = "speakers";
    const cv = ctx.createConvolver(); cv.normalize = false; cv.buffer = reverbIR(ctx);
    bus.connect(cv); cv.connect(dest);
    return bus;
  }
  // volume automation (command.rs SetTrackAutomation): the fader follows the drawn line from `pos` on
  function automate(param, t, at, pos) {
    const pts = (t.automation || []).map(([k, d]) => [t2s(k), Math.pow(10, d / 20)]);
    if (!pts.length) return false;
    const val = (x) => { if (x <= pts[0][0]) return pts[0][1]; for (let i = 1; i < pts.length; i++) if (x <= pts[i][0]) { const [a0, v0] = pts[i - 1], [a1, v1] = pts[i]; return v0 + ((v1 - v0) * (x - a0)) / Math.max(1e-9, a1 - a0); } return pts[pts.length - 1][1]; };
    param.cancelScheduledValues(0); param.setValueAtTime(val(pos), at);
    for (const [x, v] of pts) if (x > pos) param.linearRampToValueAtTime(v, at + (x - pos));
    return true;
  }
  function startSources() {
    const ctx = tr.ctx, p = session && session.project;
    if (!p) return;
    tr.master = ctx.createGain(); tr.master.gain.value = 0.8 * Math.pow(10, (p.master_db || 0) / 20); tr.master.connect(ctx.destination);
    const aud = audible(p), rbus = p.tracks.some((t) => t.fx && t.fx.reverb > 0) ? reverbBusFor(ctx, tr.master) : null;
    const dbus = p.tracks.some((t) => t.delay_send > 0) ? delayBusFor(ctx, tr.master) : null;
    tr.meters = new Map();
    const busIn = new Map(), outs = new Map();
    const strip = (t, on) => { // gain → pan → meter → master, after the track's effects; returns [input, output]
      const g = ctx.createGain(); g.gain.value = on ? Math.pow(10, t.gain_db / 20) : 0;
      if (on) automate(g.gain, t, tr.startCtx, tr.offset);
      const an = ctx.createAnalyser(); an.fftSize = 1024;
      let out = g;
      if (ctx.createStereoPanner) { const pn = ctx.createStereoPanner(); pn.pan.value = t.pan; g.connect(pn); out = pn; }
      out.connect(an); tr.meters.set(t.id, an);
      const dest = t.output && busIn.get(t.output); // a group/folder channel, else the main output
      an.connect(dest || tr.master);
      return [trackInput(ctx, t, g, out, rbus, dbus), out];
    };
    // effect channels first (they play whatever is sent to them; solo elsewhere does not silence them)
    p.tracks.forEach((t) => { if (t.kind === "bus") { const [inp, out] = strip(t, !t.mute); busIn.set(t.id, inp); outs.set(t.id, out); } });
    p.tracks.forEach((t, i) => {
      if (t.kind === "bus") return;
      const [inp, out] = strip(t, aud[i]);
      for (const [bid, lv] of t.sends || []) { const bi = busIn.get(bid); if (bi && lv > 0) { const sg = ctx.createGain(); sg.gain.value = lv; out.connect(sg); sg.connect(bi); } }
      const go = (buf, startSec) => {
        if (!buf) return;
        const pos = tr.offset;
        if (pos >= startSec + buf.duration) return;
        const src = ctx.createBufferSource(); src.buffer = buf; src.connect(inp);
        src.start(tr.startCtx + Math.max(0, startSec - pos), Math.max(0, pos - startSec));
        tr.nodes.push(src);
      };
      for (const c of (p.clips || []).filter((c) => c.track_id === t.id && !(c.edit && c.edit.mute))) go(clipBuffer(p, c), t2s(c.start_tick));
      for (const c of (p.midi_clips || []).filter((c) => c.track_id === t.id)) go(midiBuffer(t, c), t2s(c.start_tick));
    });
  }
  function stopSources() {
    for (const n of tr.nodes) { try { n.stop(); } catch { /* already stopped */ } }
    tr.nodes = [];
    if (tr.master) { try { tr.master.disconnect(); } catch { /* ignore */ } tr.master = null; }
  }
  function play() {
    if (tr.playing) return;
    if (!ensureCtx()) return;
    const spb = unitSec();
    const beats = tr.offset / spb;
    tr.beatIndex = Math.ceil(beats - 1e-9);
    tr.startCtx = tr.ctx.currentTime + 0.05;
    tr.nextBeat = tr.startCtx + (tr.beatIndex - beats) * spb;
    tr.playing = true;
    startSources();
    schedule(); tr.timer = setInterval(schedule, 25);
  }
  function stop() { if (!tr.playing) return; tr.offset = nowSeconds(); tr.prev = null; tr.playing = false; clearInterval(tr.timer); stopSources(); }
  /** After an edit while playing: rebuild the sound from the current position. */
  function refreshPlayback() { if (tr.playing) { stop(); play(); } }
  function meterOf(id) {
    const an = tr.playing && tr.meters.get(id);
    if (!an) return 0;
    const d = new Float32Array(an.fftSize); an.getFloatTimeDomainData(d);
    let m = 0; for (const v of d) m = Math.max(m, Math.abs(v));
    return Math.round(m * 1000) / 1000;
  }

  // --- recording from the browser microphone (or an uploaded recording) ---
  let rec = null;
  // role: the older "record for this part" path (creates the part's track); trackId: the studio path — the armed audio
  // track receives the take and no track is ever created (studio architecture §1.3). countIn: clicks before the take.
  async function recordStart(role, trackId, countIn, punch) {
    if (rec) return reply("error", trackId ? "כבר מקליט. רווח עוצר" : 'כבר מקליט. לעצירה כתוב "עצור הקלטה"');
    if (trackId) {
      const t0 = session && session.project.tracks.find((x) => x.id === trackId);
      if (!t0) return reply("error", "הערוץ להקלטה לא נמצא");
      if (t0.locked) return reply("error", `הערוץ ${t0.name} נעול`);
      if (t0.kind === "midi") return midiRecordStart(t0, countIn);
    }
    if (!session) api.new_project({ name: "שיר חדש" });
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(micConstraints());
    } catch (e) {
      return reply("error", trackId ? "הדפדפן לא נתן גישה למיקרופון. מאשרים בסמל המנעול שליד כתובת האתר, או מייבאים קובץ שמע (קובץ ← ייבוא קובץ שמע)" : 'הדפדפן לא נתן גישה למיקרופון בדף הבדיקה. אפשר במקום זה להקליט זמזום בטלפון ולהעלות אותו בכפתור "העלה הקלטה"');
    }
    let t;
    if (trackId) t = session.project.tracks.find((x) => x.id === trackId);
    else {
      const r = role || "melody";
      t = session.project.tracks.find((x) => x.role === r);
      if (!t) { run({ a: "add_track", track_id: newId("trk"), role: r }); t = session.project.tracks.find((x) => x.role === r); }
    }
    const ctx = tr.ctx, src = ctx.createMediaStreamSource(stream);
    const node = ctx.createScriptProcessor(4096, 1, 1), sink = ctx.createGain();
    sink.gain.value = 0;
    rec = { stream, src, node, sink, chunks: [], frames: 0, peak: 0, trackId: t.id, sr: ctx.sampleRate, startSec: nowSeconds(), device: (stream.getAudioTracks()[0] || {}).label || "מיקרופון",
      studio: !!trackId, loop: tr.loop ? { ...tr.loop } : null, punch: punch && punch[1] - punch[0] >= 0.2 ? punch : null };
    const cur = rec; // the callback may fire once more after stop: it writes to its own record only
    node.onaudioprocess = (ev) => {
      if (cur.done || !cur.started) return;
      const d = ev.inputBuffer.getChannelData(0);
      cur.chunks.push(new Float32Array(d)); cur.frames += d.length;
      for (const v of d) cur.peak = Math.max(cur.peak, Math.abs(v));
    };
    src.connect(node); node.connect(sink); sink.connect(ctx.destination);
    if (monitorOn) { rec.mon = ctx.createGain(); rec.mon.gain.value = 0.9; src.connect(rec.mon).connect(ctx.destination); } // hear yourself (headphones)
    const n = Number(countIn) > 0 ? meterNow()[0] : 0; // count-in: one bar of the song's meter
    const begin = () => {
      if (cur.done) return;
      if (tr.playing) stop();
      play(); cur.started = true; cur.startSec = tr.offset; cur.counting = 0;
    };
    if (n > 0) { // count-in: n clicks, then the song and the take start together
      const spb = unitSec(), t0 = ctx.currentTime + 0.05;
      if (tr.playing) stop();
      for (let k = 0; k < n; k++) click(t0 + k * spb, k % meterNow()[0] === 0);
      cur.counting = n; cur.countUntil = t0 + n * spb;
      setTimeout(begin, Math.max(0, (t0 + n * spb - ctx.currentTime) * 1000));
    } else begin();
    return reply("done", trackId ? `מקליט ל"${t.name}". מיקרופון: ${rec.device}. לעצירה: רווח או כפתור העצירה` : `מקליט ל${ROLE_HE[t.role] || t.name}. מיקרופון: ${rec.device}. לעצירה: לחץ שוב על המיקרופון`);
  }
  // ---- playing an instrument live (computer keyboard, on-screen keys, MIDI keyboard) and recording what is played ----
  // A note is rendered once with the track's own sound (sampled or synth) and cached; it plays through the track's
  // own effects, so what you hear while playing is what the recording will sound like.
  const liveCache = new Map(), liveVoices = new Map(), liveIn = new Map();
  function liveBuffer(t, pitch, vel) {
    const ctx = tr.ctx, sr = ctx.sampleRate, patch = synthPatch(resolveInst(t)), sid = sampledFor(t), vb = Math.max(1, Math.min(8, Math.round(vel / 16)));
    const key = `${patch}:${sid}:${pitch}:${vb}:${sr}`;
    if (liveCache.has(key)) return liveCache.get(key);
    const notes = [{ start: 0, len: Math.round(sr * (patch === "drums" ? 1.2 : 3)), pitch, velocity: vb * 16 - 1 }];
    let r = sid ? root.HUMSamples.renderSampled(notes, root.HUMSamples.sets[sid], sr) : null;
    if (!r || !r.left.length || (r.rest && r.rest.length)) r = root.HUMSynth.render(notes, patch, sr);
    if (!r.left.length) return null;
    const b = ctx.createBuffer(2, r.left.length, sr); b.copyToChannel(r.left, 0); b.copyToChannel(r.right, 1);
    liveCache.set(key, b); if (liveCache.size > 400) liveCache.delete(liveCache.keys().next().value);
    return b;
  }
  function liveInput(t) {
    const ctx = tr.ctx, rev = session.project.revision, hit = liveIn.get(t.id);
    if (hit && hit.rev === rev) return hit.input;
    if (hit) { try { hit.out.disconnect(); } catch { /* fine */ } }
    const g = ctx.createGain(); g.gain.value = Math.pow(10, t.gain_db / 20);
    let out = g; if (ctx.createStereoPanner) { const pn = ctx.createStereoPanner(); pn.pan.value = t.pan; g.connect(pn); out = pn; }
    const m = ctx.createGain(); m.gain.value = 0.8; out.connect(m); m.connect(ctx.destination);
    const input = trackInput(ctx, t, g, out, null, null);
    liveIn.set(t.id, { rev, input, out: m });
    return input;
  }
  function liveNote(trackId, pitch, velocity, on) {
    if (!session || !ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    const t = session.project.tracks.find((x) => x.id === trackId);
    if (!t || t.kind !== "midi") return reply("error", "מנגנים בערוץ של כלי וירטואלי");
    pitch = Math.max(0, Math.min(127, Math.round(pitch))); velocity = Math.max(1, Math.min(127, Math.round(velocity || 96)));
    const k = `${trackId}:${pitch}`, ctx = tr.ctx, now = ctx.currentTime;
    const old = liveVoices.get(k);
    if (old) { old.env.gain.setTargetAtTime(0, now, 0.03); old.src.stop(now + 0.2); liveVoices.delete(k); }
    if (on) {
      const buf = liveBuffer(t, pitch, velocity); if (!buf) return reply("error", "אין צליל לתו הזה");
      const src = ctx.createBufferSource(), env = ctx.createGain(); src.buffer = buf; env.gain.value = 1;
      src.connect(env).connect(liveInput(t)); src.start(now);
      liveVoices.set(k, { src, env });
    } else if (old) { /* released above */ }
    if (rec && rec.midi && rec.started && rec.trackId === trackId) {
      const at = nowSeconds();
      if (on) rec.open.set(pitch, { at, velocity });
      else { const o = rec.open.get(pitch); if (o) { rec.open.delete(pitch); rec.notes.push({ pitch, start: o.at, end: Math.max(o.at + 0.03, at), velocity: o.velocity }); } }
    }
    return { status: "done" };
  }
  function midiRecordStart(t, countIn) {
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    rec = { midi: true, trackId: t.id, notes: [], open: new Map(), startSec: nowSeconds(), sr: 1, frames: 0, peak: 0, studio: true, device: "מקלדת" };
    const cur = rec, n = Number(countIn) > 0 ? meterNow()[0] : 0;
    const begin = () => { if (cur.done) return; if (tr.playing) stop(); play(); cur.started = true; cur.startSec = tr.offset; };
    if (n > 0) {
      const spb = unitSec(), t0 = tr.ctx.currentTime + 0.05;
      if (tr.playing) stop();
      for (let k = 0; k < n; k++) click(t0 + k * spb, k === 0);
      setTimeout(begin, Math.max(0, (t0 + n * spb - tr.ctx.currentTime) * 1000));
    } else begin();
    return reply("done", `מקליט נגינה ל"${t.name}". מנגנים במקלדת; עצירה ברווח`);
  }
  function midiRecordStop(r) {
    stop();
    if (!r.started) return reply("done", "ההקלטה בוטלה לפני שהתחילה");
    const end = nowSeconds();
    for (const [pitch, o] of r.open) r.notes.push({ pitch, start: o.at, end: Math.max(o.at + 0.05, end), velocity: o.velocity });
    if (!r.notes.length) return reply("error", "לא נוגנו תווים, ולכן לא נשמר כלום");
    const barS = (60 / bpmNow()) * (4 / meterNow()[1]) * meterNow()[0], from = Math.floor(Math.min(...r.notes.map((x) => x.start)) / barS) * barS, base = s2t(from);
    const notes = r.notes.map((x) => ({ pitch: x.pitch, start_tick: Math.max(0, s2t(x.start) - base), dur_ticks: Math.max(1, s2t(x.end) - s2t(x.start)), velocity: x.velocity, confidence: 1 })).sort((a, b) => a.start_tick - b.start_tick);
    const id = newId("mcl");
    // playing over an earlier part replaces it (Ctrl+Z brings it back) — two parts sounding at once is never what was meant
    const endT = base + Math.max(...notes.map((n) => n.start_tick + n.dur_ticks)), acts = [];
    let replaced = 0;
    for (const c of (session.project.midi_clips || []).filter((c) => c.track_id === r.trackId)) {
      const ce = c.start_tick + c.notes.reduce((m, n) => Math.max(m, n.start_tick + n.dur_ticks), 0);
      if (c.start_tick < endT && ce > base) { acts.push({ a: "remove_midi_clip", clip_id: c.id }); replaced++; }
    }
    acts.push({ a: "add_midi_clip", clip: { id, track_id: r.trackId, start_tick: base, notes, source_clip: null, generator: null } });
    try { run({ a: "batch", label: "הקלטת נגינה", actions: acts }); } catch (e) { return reply("error", String(e)); }
    if (replaced) { const res = reply("done", `נשמרו ${notes.length} תווים במקום הקטע הקודם (Ctrl+Z מחזיר אותו)`); res.clip_id = id; return res; }
    const res = reply("done", `נשמרו ${notes.length} תווים. רווח כדי לשמוע; עריכה בעורך התווים`); res.clip_id = id; return res;
  }
  function addRecording(pcm, sr, trackId, startSec, label) {
    const mediaId = newId("rec"), clipId = newId("clp");
    audioStore.set(mediaId, { pcm, sr });
    idbPut(mediaId, { pcm, sr });
    run({ a: "add_clip", clip: { id: clipId, track_id: trackId, media_id: mediaId, start_tick: Math.max(0, s2t(startSec)) }, media: { id: mediaId, frames: pcm.length, sample_rate: sr, origin: label } });
    return clipId;
  }
  function recordStop() {
    if (!rec) return reply("error", "אין הקלטה פעילה");
    if (rec.midi) { const r = rec; rec = null; r.done = true; return midiRecordStop(r); }
    const r = rec; rec = null; r.done = true; r.node.onaudioprocess = null;
    try { r.src.disconnect(); r.node.disconnect(); r.sink.disconnect(); if (r.mon) r.mon.disconnect(); } catch { /* ignore */ }
    r.stream.getTracks().forEach((x) => x.stop());
    stop();
    if (!r.started) return reply("done", "ההקלטה בוטלה לפני שהתחילה");
    if (r.frames < r.sr * 0.3) return reply("error", "ההקלטה קצרה מדי — לא נשמרה");
    const pcm = new Float32Array(r.frames);
    let o = 0; for (const c of r.chunks) { pcm.set(c, o); o += c.length; }
    if (!r.studio) {
      try { addRecording(pcm, r.sr, r.trackId, r.startSec, "recording"); } catch (e) { return reply("error", String(e)); }
      return reply("done", `ההקלטה נשמרה (${(pcm.length / r.sr).toFixed(1)} שניות). אפשר לנגן, או "הפוך זמזום לתווים"`);
    }
    // studio: cycle passes become takes, punch keeps only its range, older overlapping takes stay but are muted
    const sr = r.sr, passes = [];
    const at = (sec) => Math.max(0, Math.min(pcm.length, Math.round(sec * sr)));
    if (r.punch) {
      const a = Math.max(r.punch[0], r.startSec), b = r.punch[1];
      if (b - a >= 0.2 && at(b - r.startSec) > at(a - r.startSec)) passes.push([pcm.slice(at(a - r.startSec), at(b - r.startSec)), a]);
    } else if (r.loop && r.startSec < r.loop.end) {
      const len = r.loop.end - r.loop.start;
      let from = 0, start = r.startSec, dur = r.loop.end - r.startSec;
      while (from < pcm.length) {
        const n = Math.min(pcm.length - from, Math.round(dur * sr));
        if (n >= sr * 0.5 || !passes.length) passes.push([pcm.slice(from, from + n), start]);
        from += Math.round(dur * sr); start = r.loop.start; dur = len;
      }
    } else passes.push([pcm, r.startSec]);
    if (!passes.length || passes.every((x) => x[0].length < sr * 0.3)) return reply("error", "לא הוקלט כלום בטווח הפאנץ'");
    const p = session.project, actions = [], ids = [];
    for (const [x, st] of passes) {
      const mediaId = newId("rec"), clipId = newId("clp");
      audioStore.set(mediaId, { pcm: x, sr }); idbPut(mediaId, { pcm: x, sr });
      actions.push({ a: "add_clip", clip: { id: clipId, track_id: r.trackId, media_id: mediaId, start_tick: Math.max(0, s2t(st)), edit: { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0, mute: false } }, media: { id: mediaId, frames: x.length, sample_rate: sr, origin: "recording" } });
      ids.push([clipId, st, x.length / sr]);
    }
    // the last complete pass is heard (a short tail pass after it is kept, muted); earlier passes and older takes are kept, muted
    let heard = ids.length - 1;
    if (ids.length > 1 && r.loop && ids[heard][2] < 0.5 * (r.loop.end - r.loop.start)) heard--;
    const lastId = ids[heard][0];
    for (const [id] of ids.filter((x, k) => k !== heard)) { const k = actions.findIndex((x) => x.clip && x.clip.id === id); actions[k].clip.edit.mute = true; }
    const lo = Math.min(...ids.map((x) => x[1])), hi = Math.max(...ids.map((x) => x[1] + x[2]));
    for (const c of (p.clips || []).filter((c) => c.track_id === r.trackId && !(c.edit && c.edit.mute))) {
      const m = (p.media || []).find((x) => x.id === c.media_id) || {}, e = { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0, mute: false, gain_cdb: 0, ...(c.edit || {}) };
      const msr = m.sample_rate || sr, frames = Math.min(Math.max(0, (m.frames || 0) - e.trim_start), e.trim_len || Infinity), cs = t2s(c.start_tick), cl = frames / msr;
      if (!(cs < hi - 0.05 && cs + cl > lo + 0.05)) continue;
      if (!r.punch) { actions.push({ a: "set_clip_edit", clip_id: c.id, start_tick: c.start_tick, edit: { ...e, mute: true } }); continue; }
      // punch: keep the take before and after the punched bars; only the part under the new take becomes a muted take
      const a0 = Math.max(0, Math.round((lo - cs) * msr)), b0 = Math.min(frames, Math.round((hi - cs) * msr)), edge = 0.05 * msr;
      if (a0 > edge) actions.push({ a: "set_clip_edit", clip_id: c.id, start_tick: c.start_tick, edit: { ...e, trim_len: a0, fade_out_ms: 0 } });
      const mid = { ...e, trim_start: e.trim_start + a0, trim_len: Math.max(1, b0 - a0), fade_in_ms: 0, fade_out_ms: 0, mute: true };
      if (a0 > edge) actions.push({ a: "add_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("clp"), start_tick: s2t(cs + a0 / msr), edit: mid } });
      else actions.push({ a: "set_clip_edit", clip_id: c.id, start_tick: s2t(cs + a0 / msr), edit: mid });
      if (frames - b0 > edge) actions.push({ a: "add_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("clp"), start_tick: s2t(cs + b0 / msr), edit: { ...e, trim_start: e.trim_start + b0, trim_len: frames - b0, fade_in_ms: 0 } } });
    }
    try { run({ a: "batch", label: ids.length > 1 ? `הקלטה בלולאה: ${ids.length} טייקים` : "הקלטה", actions }); } catch (e) { return reply("error", String(e)); }
    const secs = ids.reduce((m, x) => m + x[2], 0);
    const res = reply("done", ids.length > 1 ? `נשמרו ${ids.length} טייקים חדשים. שומעים את האחרון; את הטוב ביותר בוחרים בלחיצה על "טייקים" בקטע` : r.punch ? `הקטע המסומן הוקלט מחדש (${secs.toFixed(1)} שניות); ההקלטה הקודמת נשמרה כטייק` : `ההקלטה נשמרה (${secs.toFixed(1)} שניות). רווח כדי לשמוע`);
    res.clip_id = lastId; return res;
  }
  // a take captured by the voice conversation, judged to be humming: stored like any recording
  function importPcm(pcm, sr, label) {
    if (!session) api.new_project({ name: "שיר חדש" });
    if (!(pcm && pcm.length >= sr * 0.3)) return reply("error", "ההקלטה קצרה מדי");
    let t = session.project.tracks.find((x) => x.role === "melody");
    if (!t) { run({ a: "add_track", track_id: newId("trk"), role: "melody" }); t = session.project.tracks.find((x) => x.role === "melody"); }
    try { addRecording(pcm, sr, t.id, 0, label || "voice"); } catch (e) { return reply("error", String(e)); }
    return reply("done", `הזמזום נשמר (${(pcm.length / sr).toFixed(1)} שניות)`);
  }
  // opts (studio): {track_id, at_sec} puts the file on that audio track at that time; {new_track: true} makes an audio
  // track named after the file. Without opts: the older hum path (the melody track, at the start).
  async function importAudio(file, opts) {
    if (!session) api.new_project({ name: "שיר חדש" });
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    let buf;
    try { buf = await tr.ctx.decodeAudioData(await file.arrayBuffer()); } catch { return reply("error", "לא הצלחתי לקרוא את הקובץ. נסה הקלטה בפורמט ‎m4a, mp3 או wav"); }
    if (buf.duration > 180) return reply("error", "ההקלטה ארוכה מ-3 דקות. אפשר לחתוך אותה ולהעלות חלק");
    const pcm = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) pcm[i] += d[i] / buf.numberOfChannels; }
    if (opts && (opts.track_id || opts.new_track)) {
      let tid = opts.track_id;
      if (!tid) { const r = api.create_track({ role: "other", kind: "audio", name: String(file.name || "קובץ").replace(/\.[^.]+$/, "").slice(0, 60) }); if (r.status !== "done") return r; tid = r.track_id; }
      const tt = session.project.tracks.find((x) => x.id === tid);
      if (!tt || tt.kind !== "audio") return reply("error", "מייבאים שמע לערוץ שמע");
      try { addRecording(pcm, buf.sampleRate, tid, Math.max(0, Number(opts.at_sec) || 0), "import"); } catch (e) { return reply("error", String(e)); }
      const res = reply("done", `"${file.name}" נוסף לערוץ "${tt.name}" (${buf.duration.toFixed(1)} שניות)`); res.track_id = tid; return res;
    }
    let t = session.project.tracks.find((x) => x.role === "melody");
    if (!t) { run({ a: "add_track", track_id: newId("trk"), role: "melody" }); t = session.project.tracks.find((x) => x.role === "melody"); }
    try { addRecording(pcm, buf.sampleRate, t.id, 0, "import"); } catch (e) { return reply("error", String(e)); }
    return reply("done", `ההקלטה "${file.name}" נטענה (${buf.duration.toFixed(1)} שניות). עכשיו אפשר "הפוך זמזום לתווים"`);
  }

  // --- vocal range (port of apps/core/src/vocal.rs) — the test recording never joins the song ---
  const HE_NOTES = ["דו", "דו#", "רה", "מי♭", "מי", "פה", "פה#", "סול", "סול#", "לה", "סי♭", "סי"];
  const noteHe = (m) => `${HE_NOTES[m % 12]} ${Math.floor(m / 12) - 1}`;
  const PROFILE_KEY = "hum-vocal-profile";
  let vocalRange = null, vocalRec = null;
  function savedProfile() {
    try { const j = JSON.parse(localStorage.getItem(PROFILE_KEY) || "null"); return j && j.lowest <= j.comfort_low && j.comfort_low <= j.comfort_high && j.comfort_high <= j.highest ? j : null; } catch { return null; }
  }
  const currentRange = () => vocalRange || savedProfile();
  function describeRange(r) {
    const span = r.highest - r.lowest;
    const spanHe = span >= 22 ? "כשתי אוקטבות" : span >= 15 ? "כאוקטבה וחצי" : span >= 11 ? "כאוקטבה" : `${span} חצאי טונים`;
    const sure = r.confidence >= 0.7 ? "המדידה אמינה" : r.confidence >= 0.4 ? "המדידה חלקית — שירה ארוכה יותר תדייק אותה" : "המדידה קצרה מדי ולכן לא בטוחה. כדאי לשיר לפחות 15 שניות, מהצליל הכי נמוך שנוח לך ועד הכי גבוה";
    if (r.method === "imitation") return `הטווח שנמדד: מ־${noteHe(r.lowest)} עד ${noteHe(r.highest)} (${spanHe}). האזור הנוח לך: ${noteHe(r.comfort_low)} עד ${noteHe(r.comfort_high)}.`;
    return `הטווח שנמדד: מ־${noteHe(r.lowest)} עד ${noteHe(r.highest)} (${spanHe}). האזור הנוח לך: ${noteHe(r.comfort_low)} עד ${noteHe(r.comfort_high)}. ${sure}.`;
  }
  const shiftHe = (st) => `${st > 0 ? "להעלות" : "להוריד"} ${shiftHow(Math.abs(st))}`;
  function melodyNotes() {
    if (!session) return [];
    const p = session.project;
    const lead = p.tracks.filter((t) => t.role === "lead").map((t) => t.id);
    let v = (p.midi_clips || []).filter((c) => lead.includes(c.track_id)).flatMap((c) => c.notes.map((n) => [n.pitch, n.dur_ticks]));
    if (v.length) return v;
    const mel = p.tracks.filter((t) => ["flute", "soft_lead", "epiano"].includes(resolveInst(t)) && t.role !== "piano").map((t) => t.id);
    return (p.midi_clips || []).filter((c) => mel.includes(c.track_id)).flatMap((c) => c.notes.map((n) => [n.pitch, n.dur_ticks]));
  }
  function adviceText(a) {
    if (a.semitones === 0) return `המנגינה של השיר כבר יושבת טוב בקול שלך (${(a.fit_before * 100).toFixed(0)}% ממנה באזור הנוח).`;
    const k = session.project.key, to = { tonic: (((k.tonic + a.semitones) % 12) + 12) % 12, mode: k.mode };
    return `ההמלצה: ${shiftHe(a.semitones)} את השיר — מ-${keyLabel(k)} ל-${keyLabel(to)}. כך ${(a.fit_after * 100).toFixed(0)}% מהמנגינה יהיו באזור הנוח לך (עכשיו ${(a.fit_before * 100).toFixed(0)}%).`;
  }
  function vocalFromSamples(pcm, sr) {
    let r;
    try { r = root.HUMTranscribe.analyzeRange(pcm, 1, sr); } catch (e) { return reply("error", String(e)); }
    vocalRange = r;
    let msg = describeRange(r);
    const a = root.HUMTranscribe.recommendTransposition(melodyNotes(), r);
    if (a && session) msg += " " + adviceText(a);
    msg += ' כדי שאזכור את הטווח גם בפעם הבאה, כתוב "שמור את טווח הקול שלי" (אפשר למחוק בכל זמן).';
    return reply("done", msg);
  }
  // ---- guided range check by imitation (spec addendum 25.10 part 8): the studio plays a tone, the person sings it back ----
  function guideTone(midi, seconds) {
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    const ctx = tr.ctx, t0 = ctx.currentTime + 0.05, f = 440 * Math.pow(2, (midi - 69) / 12), d = Math.max(0.4, seconds || 1.2);
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(0.22, t0 + 0.04); g.gain.setValueAtTime(0.22, t0 + d - 0.15); g.gain.linearRampToValueAtTime(0, t0 + d);
    for (const [mul, amp] of [[1, 1], [2, 0.35], [3, 0.12]]) { const o = ctx.createOscillator(), a = ctx.createGain(); o.frequency.value = f * mul; a.gain.value = amp; o.connect(a).connect(g); o.start(t0); o.stop(t0 + d + 0.02); }
    g.connect(ctx.destination);
    root.__HUM_GUIDE__ = { midi, until: performance.now() + (d + 0.1) * 1000 };
    return { status: "done", message: "", seconds: d };
  }
  // the attempts become a range in the same form as the free singing test, so key advice works the same
  function guideResult(attempts) {
    const ok = attempts.filter((a) => a.sung !== null && Math.abs(a.cents_off) <= 100 && a.held_seconds >= 0.4);
    if (ok.length < 3) return reply("error", "לא הצלחתי לשמוע מספיק צלילים. נסה שוב במקום שקט, קרוב למיקרופון");
    const steady = ok.filter((a) => a.stability_cents <= 30 && a.held_seconds >= 0.8);
    const sung = ok.map((a) => a.sung), comf = (steady.length >= 2 ? steady : ok).map((a) => a.sung);
    const r = (x) => Math.max(0, Math.min(127, Math.round(x)));
    const lowest = r(Math.min(...sung)), highest = r(Math.max(...sung));
    const confidence = Math.round(Math.min(1, (ok.length / 8) * 0.6 + (steady.length / Math.max(ok.length, 1)) * 0.4) * 100) / 100;
    vocalRange = { lowest, highest, comfort_low: r(Math.min(...comf)), comfort_high: r(Math.max(...comf)), center: r((Math.min(...comf) + Math.max(...comf)) / 2),
      stability_cents: Math.round(ok.reduce((x, a) => x + a.stability_cents, 0) / ok.length), voiced_seconds: Math.round(ok.reduce((x, a) => x + a.voiced_seconds, 0) * 10) / 10, confidence, method: "imitation" };
    let msg = "מצוין! הבנתי אילו צלילים נוחים לך. עכשיו נוכל להתאים את הפלייבק לקול שלך. " + describeRange(vocalRange);
    if (confidence < 0.5) msg += " הבדיקה הייתה קצרה, אז אני עוד לא בטוח לגמרי. אם תרצה, נעשה עוד בדיקה קצרה.";
    const adv = root.HUMTranscribe.recommendTransposition(melodyNotes(), vocalRange);
    if (adv && session) msg += " " + adviceText(adv);
    const fit = sectionFit(vocalRange);
    if (fit) msg += " " + fit;
    msg += ' כדי שאזכור את הטווח גם בפעם הבאה, אמור "שמור את טווח הקול שלי" (אפשר למחוק בכל זמן).';
    return reply("done", msg);
  }
  // verse and chorus checked separately (spec addendum 25.10 part 8): a key that suits the verse may push the chorus
  // too high; the melody of each part is compared with the comfortable zone that was just measured
  function sectionFit(r) {
    if (!session || !session.project.harmony) return "";
    const p = session.project, secs = p.harmony.sections || [];
    const lead = p.tracks.filter((t) => t.role === "lead").map((t) => t.id);
    const notes = (p.midi_clips || []).filter((c) => lead.includes(c.track_id)).flatMap((c) => c.notes.map((n) => ({ pitch: n.pitch, t: c.start_tick + n.start_tick, d: n.dur_ticks })));
    const part = (kind) => { const ss = secs.filter((x) => x.kind === kind); return notes.filter((n) => ss.some((x) => n.t >= x.start_tick && n.t < x.start_tick + x.dur_ticks)); };
    const judge = (ns) => { if (!ns.length) return null; const tot = ns.reduce((x, n) => x + n.d, 0); return { high: ns.filter((n) => n.pitch > r.comfort_high).reduce((x, n) => x + n.d, 0) / tot, low: ns.filter((n) => n.pitch < r.comfort_low).reduce((x, n) => x + n.d, 0) / tot }; };
    const v = judge(part("verse")), c = judge(part("chorus"));
    if (!v || !c) return "";
    const ok = (j) => j.high <= 0.15 && j.low <= 0.15;
    const one = (name, j) => (j.high > 0.15 ? `ב${name} יש צלילים גבוהים מהאזור הנוח לך` : j.low > 0.15 ? `ב${name} יש צלילים נמוכים מהאזור הנוח לך` : `ה${name} יושב טוב בקול שלך`);
    if (ok(v) && ok(c)) return "בדקתי את הבית ואת הפזמון בנפרד: שניהם יושבים טוב בקול שלך.";
    return `בדקתי את הבית ואת הפזמון בנפרד: ${one("בית", v)}, ו${one("פזמון", c)}.`;
  }
  // ---- singing coach (spec addendum 25.10 part 9): after a vocal take, sung pitch and entry against the melody ----
  // Kind and true: what went well first, at most a few notes to work on, never a score or a comparison.
  // The recording itself is never changed.
  let lastCoach = null;
  function coachReview(lvl, accuracy) {
    const ACC = Number(accuracy) > 0 ? Number(accuracy) : 35;
    if (!session) return reply("error", "אין פרויקט פתוח");
    const p = session.project;
    const voc = p.tracks.find((t) => t.role === "vocals");
    const clips = voc ? (p.clips || []).filter((c) => c.track_id === voc.id) : [];
    const clip = clips[clips.length - 1];
    if (!clip) return reply("error", 'עדיין אין הקלטת שירה. לחץ על "שיר", שיר עם השיר, ואז בקש משוב');
    const a = audioStore.get(clip.media_id);
    if (!a) return reply("error", "ההקלטה עוד לא נטענה. נסה שוב בעוד רגע");
    const lead = p.tracks.find((t) => t.role === "lead");
    const target = [];
    for (const mc of (p.midi_clips || []).filter((c) => lead && c.track_id === lead.id)) for (const n of mc.notes) target.push({ pitch: n.pitch, t0: t2s(mc.start_tick + n.start_tick), t1: t2s(mc.start_tick + n.start_tick + n.dur_ticks) });
    if (!target.length) return reply("error", "אין עדיין מנגינה להשוות אליה. זמזם מנגינה או הלחן שיר, ואז שיר עליה");
    const tr = root.HUMTranscribe.pitchTrack(a.pcm, 1, a.sr), clip0 = t2s(clip.start_tick), clip1 = clip0 + a.pcm.length / a.sr;
    const rows = [];
    for (const n of target) {
      if (n.t1 <= clip0 || n.t0 >= clip1 || n.t1 - n.t0 < 0.15) continue;
      const i0 = Math.max(0, Math.floor((n.t0 - clip0) / tr.hop_sec)), i1 = Math.min(tr.midi.length, Math.ceil((n.t1 - clip0) / tr.hop_sec));
      const seg = tr.midi.slice(i0, i1), v = seg.filter((x) => x !== null);
      if (v.length < Math.max(3, seg.length * 0.4)) { rows.push({ ...n, sung: null }); continue; }
      const m = v.slice().sort((x, y) => x - y)[Math.floor(v.length / 2)];
      let c = ((m - n.pitch) % 12 + 12) % 12; if (c > 6) c -= 12; // the octave is the singer's choice
      const first = seg.findIndex((x) => x !== null);
      rows.push({ ...n, sung: Math.round(m * 100) / 100, cents: Math.round(c * 100), entry: Math.round(first * tr.hop_sec * 1000) });
    }
    const sung = rows.filter((r) => r.sung !== null);
    if (sung.length < 3) return reply("error", "לא שמעתי מספיק שירה על המנגינה. נסה שוב, קרוב יותר למיקרופון");
    const good = sung.filter((r) => Math.abs(r.cents) <= ACC).length, close = sung.filter((r) => Math.abs(r.cents) > ACC && Math.abs(r.cents) <= 100).length;
    const late = sung.filter((r) => r.entry > 180).length;
    const work = sung.filter((r) => Math.abs(r.cents) > ACC).sort((x, y) => Math.abs(y.cents) - Math.abs(x.cents)).slice(0, lvl === "easy" ? 1 : 3);
    const pct = Math.round((good / sung.length) * 100);
    let msg = pct >= 80 ? "יפה מאוד! רוב הצלילים יצאו מדויקים." : pct >= 50 ? "יפה, את רוב הדרך עשית נכון." : "התחלה טובה. נעבוד יחד על כמה צלילים.";
    if (prevCoach && pct > prevCoach + 5) msg += " הפעם יצא מדויק יותר מהניסיון הקודם. מצוין!";
    if (work.length) {
      const w = work[0], dir = w.cents > 0 ? "קצת גבוה" : "קצת נמוך";
      msg += ` בצליל ${noteHe(w.pitch)} (בשנייה ${w.t0.toFixed(1)}) יצא ${dir}. בוא ננסה לדייק אותו עוד קצת: אמור "תרגל את הצליל".`;
    }
    if (lvl !== "easy" && late >= Math.max(2, sung.length * 0.25)) msg += " בכמה משפטים הכניסה הייתה מעט מאוחרת. כדאי להתחיל לשיר עם הפעמה.";
    if (lvl === "pro") msg += ` (${good} מדויקים, ${close} קרובים, ${sung.length - good - close} רחוקים מתוך ${sung.length} צלילים; סטייה ממוצעת ${Math.round(sung.reduce((x, r) => x + Math.abs(r.cents), 0) / sung.length)} סנט.)`;
    prevCoach = pct;
    lastCoach = { rows, work, clip: clip.id };
    const r = reply("done", msg); r.coach = { rows: rows.map((x) => ({ t0: x.t0, t1: x.t1, pitch: x.pitch, sung: x.sung })), practice: work.map((x) => x.pitch) };
    return r;
  }
  let prevCoach = null;
  async function vocalStart() {
    if (rec || vocalRec) return reply("error", "כבר מקליט. קודם עצור את ההקלטה");
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia(micConstraints()); }
    catch { return reply("error", 'הדפדפן לא נתן גישה למיקרופון בדף הבדיקה. אפשר להקליט את עצמך שר בטלפון ולהעלות בכפתור "בדיקת קול מקובץ"'); }
    stop();
    const ctx = tr.ctx, src = ctx.createMediaStreamSource(stream), node = ctx.createScriptProcessor(4096, 1, 1), sink = ctx.createGain();
    sink.gain.value = 0;
    vocalRec = { stream, src, node, sink, chunks: [], frames: 0, sr: ctx.sampleRate };
    const cur = vocalRec;
    node.onaudioprocess = (ev) => { if (cur.done) return; const d = ev.inputBuffer.getChannelData(0); cur.chunks.push(new Float32Array(d)); cur.frames += d.length; };
    src.connect(node); node.connect(sink); sink.connect(ctx.destination);
    return reply("done", 'בדיקת קול התחילה. שיר או זמזם בקול נוח — מהצליל הכי נמוך שנוח לך ועד הכי גבוה, או כמה שורות משיר שאתה אוהב. ההקלטה הזו לא נכנסת לשיר. בסוף לחץ שוב על הכפתור או כתוב "סיים בדיקת קול"');
  }
  function vocalStop() {
    if (!vocalRec) return reply("error", 'אין בדיקת קול פעילה. כתוב "בדיקת קול" כדי להתחיל');
    const r = vocalRec; vocalRec = null; r.done = true; r.node.onaudioprocess = null;
    try { r.src.disconnect(); r.node.disconnect(); r.sink.disconnect(); } catch { /* ignore */ }
    r.stream.getTracks().forEach((x) => x.stop());
    const pcm = new Float32Array(r.frames);
    let o = 0; for (const c of r.chunks) { pcm.set(c, o); o += c.length; }
    return vocalFromSamples(pcm, r.sr); // the samples are dropped after analysis
  }
  async function vocalFile(file) {
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    let buf;
    try { buf = await tr.ctx.decodeAudioData(await file.arrayBuffer()); } catch { return reply("error", "לא הצלחתי לקרוא את הקובץ. נסה הקלטה בפורמט ‎m4a, mp3 או wav"); }
    if (buf.duration > 180) return reply("error", "ההקלטה ארוכה מ-3 דקות");
    const pcm = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) pcm[i] += d[i] / buf.numberOfChannels; }
    return vocalFromSamples(pcm, buf.sampleRate);
  }
  function recommendKey() {
    const r = currentRange();
    if (!r) return reply("ask", 'כדי להמליץ על סולם אני צריך לשמוע את הקול שלך. כתוב "בדיקת קול", שיר כחצי דקה, ואז "סיים בדיקת קול"');
    if (!session) return reply("error", "אין פרויקט פתוח");
    const a = root.HUMTranscribe.recommendTransposition(melodyNotes(), r);
    if (!a) return reply("error", `${describeRange(r)} אבל עדיין אין מנגינה בתווים להשוות אליה. קודם זמזם והפוך לתווים, או הלחן שיר`);
    let msg = adviceText(a);
    if (a.semitones === 0) return reply("done", msg);
    if (r.confidence < 0.4) msg += " שים לב: מדידת הקול הייתה קצרה, אז ההמלצה משוערת.";
    const alts = a.alternatives.filter((x) => x[0] !== 0).slice(0, 2).map((x) => `${shiftHe(x[0])} (${(x[1] * 100).toFixed(0)}%)`);
    if (alts.length) msg += ` אפשרויות נוספות: ${alts.join(", ")}.`;
    msg += " התופים וההקלטות לא משתנים. לבצע?";
    pending = { proposed: { type: "action", action: { a: "transpose", semitones: a.semitones } }, utterance: "סולם שנוח לי", revision: session.project.revision };
    return reply("confirm", msg);
  }
  function vocalSave() {
    if (!vocalRange) return reply("error", 'עדיין לא נמדד טווח קול בפעם הזו. כתוב "בדיקת קול" כדי להתחיל');
    try { localStorage.setItem(PROFILE_KEY, JSON.stringify(vocalRange)); } catch { return reply("error", "הדפדפן הזה לא מאפשר לשמור. הטווח יישמר רק עד סגירת הדף"); }
    return reply("done", 'טווח הקול נשמר בדפדפן הזה בלבד. אפשר למחוק אותו בכל זמן: "מחק את טווח הקול שלי"');
  }
  function vocalDelete() {
    vocalRange = null;
    try { localStorage.removeItem(PROFILE_KEY); } catch { /* nothing stored */ }
    return reply("done", "טווח הקול נמחק. אפשר למדוד שוב בכל זמן");
  }

  // --- humming → notes (port of apps/core/src/notes.rs) ---
  function transcribe(fromSec) {
    if (!session) return reply("error", "אין פרויקט פתוח");
    const p = session.project;
    const clip = (p.clips || [])[(p.clips || []).length - 1];
    if (!clip) return reply("error", 'עדיין אין הקלטה. לחץ על המיקרופון, זמזם, ואז "הפוך לתווים"');
    const a = audioStore.get(clip.media_id);
    if (!a) return reply("error", "קובץ ההקלטה לא נמצא בדפדפן הזה (הקלטות נשמרות רק בדפדפן שבו הוקלטו)");
    let r;
    try { r = root.HUMTranscribe.transcribe(a.pcm, 1, a.sr); } catch (e) { return reply("error", String(e)); }
    // spoken words before the humming (one take: "a Mediterranean song, tra-la-la…") are not melody
    if (fromSec > 0) { r.notes = r.notes.filter((n) => n.start_seconds >= fromSec - 0.05); if (!r.notes.length) return reply("error", "לא שמעתי מנגינה אחרי הדיבור. לחץ ודבר שוב, וזמזם את המנגינה"); }
    const actions = [];
    let bpm = p.bpm, tempoNote = "";
    if (!(p.midi_clips || []).length && r.tempo.bpm !== null && r.tempo.confidence >= 0.6 && Math.abs(r.tempo.bpm - p.bpm) >= 2 && r.tempo.bpm >= 40 && r.tempo.bpm <= 240) {
      actions.push({ a: "set_tempo", bpm: r.tempo.bpm }); bpm = r.tempo.bpm;
      tempoNote = ` קצב השיר עודכן ל-${r.tempo.bpm.toFixed(0)} לפי הזמזום.`;
    }
    const tk = (sec) => Math.round((sec * bpm * PPQ) / 60);
    const t0 = (clip.start_tick * 60) / (bpm * PPQ);
    const notes = r.notes.map((n) => { const s = tk(t0 + n.start_seconds) - clip.start_tick, e = tk(t0 + n.start_seconds + n.duration_seconds) - clip.start_tick; return { pitch: n.pitch, start_tick: Math.max(0, s), dur_ticks: Math.max(1, e - s), velocity: Math.max(1, n.velocity), confidence: n.confidence }; });
    const lead = p.tracks.find((t) => t.role === "lead");
    const leadId = lead ? lead.id : newId("trk");
    if (!lead) actions.push({ a: "add_track", track_id: leadId, role: "lead" });
    const existing = (p.midi_clips || []).find((m) => m.source_clip === clip.id);
    actions.push(existing ? { a: "set_midi_notes", clip_id: existing.id, notes } : { a: "add_midi_clip", clip: { id: newId("mid"), track_id: leadId, start_tick: clip.start_tick, notes, source_clip: clip.id } });
    try { run({ a: "batch", label: "הפיכה לתווים", actions }); } catch (e) { return reply("error", String(e)); }
    let msg = `זוהו ${r.notes.length} תווים מהזמזום.${tempoNote} ההקלטה המקורית נשארה כמו שהיא.`;
    const low = r.notes.filter((n) => n.confidence < 0.5).length;
    if (low) msg += ` ${low} תווים בביטחון נמוך מסומנים בצבע חלש — כדאי לבדוק אותם.`;
    for (const w of r.warnings) msg += ` ${w}.`;
    if (r.key && r.key.confidence >= 0.6) {
      const key = { tonic: r.key.tonic, mode: r.key.major ? "major" : "minor" };
      if (key.tonic !== session.project.key.tonic || key.mode !== session.project.key.mode) {
        pending = { proposed: { type: "action", action: { a: "set_key", key } }, utterance: "סולם משוער מהזמזום", revision: session.project.revision };
        return reply("confirm", `${msg} הסולם המשוער של המנגינה: ${keyLabel(key)} (ביטחון ${(r.key.confidence * 100).toFixed(0)}%). לעדכן את סולם השיר?`);
      }
    }
    if (!r.key || r.key.confidence < 0.6) msg += " הסולם לא ברור מספיק מהקטע הזה, אז סולם השיר לא שונה.";
    return reply("done", msg);
  }
  function quantize() {
    if (!session) return reply("error", "אין פרויקט פתוח");
    const p = session.project, SIX = PPQ / 4;
    const lead = p.tracks.find((t) => t.role === "lead");
    const all = p.midi_clips || [];
    const c = (lead && all.filter((m) => m.track_id === lead.id).pop()) || all[all.length - 1];
    if (!c) return reply("error", 'אין עדיין תווים ליישר. קודם "הפוך לתווים"');
    const snap = (t) => (t < 0 ? -Math.round(-t / SIX) : Math.round(t / SIX)) * SIX;
    // a note whose grid point lies before the clip start moves the clip back onto the grid
    const ns = Math.max(0, Math.min(c.start_tick, ...c.notes.map((n) => snap(c.start_tick + n.start_tick))));
    const out = c.notes.map((n) => { const s = Math.max(ns, snap(c.start_tick + n.start_tick)), e = Math.max(snap(c.start_tick + n.start_tick + n.dur_ticks), s + SIX); return { ...n, start_tick: s - ns, dur_ticks: e - s }; }).sort((x, y) => x.start_tick - y.start_tick);
    for (let i = 0; i + 1 < out.length; i++) { const nx = out[i + 1].start_tick; if (out[i].start_tick < nx && out[i].start_tick + out[i].dur_ticks > nx) out[i].dur_ticks = nx - out[i].start_tick; }
    const moved = out.filter((n, i) => n.start_tick + ns !== c.notes[i].start_tick + c.start_tick || n.dur_ticks !== c.notes[i].dur_ticks).length;
    const set = { a: "set_midi_notes", clip_id: c.id, notes: out };
    try { run(ns !== c.start_tick ? { a: "batch", label: "יישור לקצב", actions: [{ a: "set_midi_clip_start", clip_id: c.id, start_tick: ns }, set] } : set); } catch (e) { return reply("error", String(e)); }
    return reply("done", `התווים יושרו לקצב (רבעי פעמה). ${moved} תווים זזו מעט. אפשר לבטל.`);
  }

  // ---------------- JSON views (same shape as Rust state_json) ----------------
  // model.rs track_audible: groups and effect channels
  function audible(p) {
    const anySolo = p.tracks.some((t) => t.solo);
    return p.tracks.map((t) => {
      if (t.kind === "bus") return !t.mute;
      const g = t.output ? p.tracks.find((x) => x.id === t.output) : null;
      return !t.mute && !(g && g.mute) && (!anySolo || t.solo || !!(g && g.solo));
    });
  }
  // --- arrangement and composition (port of apps/core/src/arrangement.rs) ---
  const ARR_ROLES = ["drums", "bass", "guitar", "piano", "strings", "winds"];
  function harmonyView(p) {
    const h = p.harmony || { style: null, chords: [], sections: [] };
    const r3 = (x) => Math.round(x * 1000) / 1000;
    const C = root.HUMCompose;
    return { style: h.style, style_name: h.style ? C.STYLE_HE[h.style] : null,
      chords: h.chords.map((c) => [r3(t2s(c.start_tick)), r3(t2s(c.dur_ticks)), C.chordName({ root: c.root, quality: c.quality })]),
      sections: h.sections.map((x) => [C.SECTION_HE[x.kind] || "", r3(t2s(x.start_tick)), r3(t2s(x.dur_ticks))]) };
  }
  function melodyOf(p) {
    const v = [];
    for (const t of p.tracks.filter((t) => t.role === "lead")) for (const c of (p.midi_clips || []).filter((c) => c.track_id === t.id)) for (const n of c.notes) v.push({ pitch: n.pitch, start: c.start_tick + n.start_tick, dur: n.dur_ticks, velocity: n.velocity });
    return v.sort((a, b) => a.start - b.start);
  }
  const listHe = (a) => (a.length <= 1 ? a.join("") : `${a.slice(0, -1).join(", ")} ו${a[a.length - 1]}`);
  const noteClip = (tid, notes, generator) => ({ id: newId("mid"), track_id: tid, start_tick: 0, notes: notes.map((n) => ({ pitch: n.pitch, start_tick: n.start, dur_ticks: n.dur, velocity: n.velocity, confidence: 1 })), source_clip: null, generator });
  const toHarmony = (style, chords, sections) => ({ style, chords: chords.map((c) => ({ start_tick: c.start, dur_ticks: c.dur, root: c.root, quality: c.quality })), sections: sections.map((x) => ({ kind: x.kind, start_tick: x.start, dur_ticks: x.dur })) });
  function arrange(style, roles, only, newHarmony) {
    if (!session) return reply("error", "אין פרויקט פתוח");
    const C = root.HUMCompose, BAR = C.BAR;
    let pre = "";
    if (!melodyOf(session.project).length) {
      if (!(session.project.clips || []).length) return reply("error", 'עדיין אין מנגינה לעבד. זמזם (כפתור המיקרופון) או בקש "תלחין שיר"');
      const r = transcribe();
      if (r.status === "error") return r;
      pending = null;
      pre = "קודם הפכתי את הזמזום לתווים. ";
    }
    let shiftMsg = "";
    const shift = [];
    const h0 = session.project.harmony || { chords: [] };
    if (!h0.chords.length) {
      const m0 = melodyOf(session.project);
      if (m0.length) {
        const off = ((m0[0].start % BAR) + BAR) % BAR;
        if (off > PPQ / 16) { shift.push({ a: "shift_timeline", delta: BAR - off }); shiftMsg = "המנגינה הוזזה מעט כך שתתחיל בתחילת תיבה. "; }
      }
    }
    const p = JSON.parse(JSON.stringify(session.project)), h = p.harmony || { style: null, chords: [], sections: [] };
    if (shift.length) { for (const x of [...(p.clips || []), ...(p.midi_clips || [])]) x.start_tick += shift[0].delta; }
    const melody = melodyOf(p);
    const st = style || h.style || "pop";
    const actions = shift.slice(), notesMsg = [];
    let key = { tonic: p.key.tonic, major: p.key.mode === "major" };
    if (!h.chords.length) {
      const e = C.estimateKey(melody);
      if (e && (e[0].tonic !== key.tonic || e[0].major !== key.major)) {
        key = e[0];
        const pk = { tonic: key.tonic, mode: key.major ? "major" : "minor" };
        actions.push({ a: "set_key", key: pk });
        notesMsg.push(`הסולם עודכן ל-${keyLabel(pk)} לפי המנגינה`);
      }
    }
    const start = melody[0].start, end = melody.reduce((m, n) => Math.max(m, n.start + n.dur), BAR);
    const reuse = h.chords.length > 0 && !newHarmony && (only !== null || h.style === st);
    let chords, sections;
    if (reuse) {
      chords = h.chords.map((c) => ({ start: c.start_tick, dur: c.dur_ticks, root: c.root, quality: c.quality }));
      sections = h.sections.map((x) => ({ kind: x.kind, start: x.start_tick, dur: x.dur_ticks }));
    } else {
      sections = C.sectionsAround(start, end, Math.min(Math.trunc(start / BAR), 2), 2);
      const variation = newHarmony ? Math.max(p.revision, 1) : 0;
      const core = C.harmonize(melody, key, start, end, { style: st, chords_per_bar: newHarmony && variation % 2 === 0 ? 2 : 1, variation });
      chords = C.extendOver(core, sections, key);
      actions.push({ a: "set_harmony", harmony: toHarmony(st, chords, sections) });
    }
    const arranged = p.tracks.filter((t) => ARR_ROLES.includes(t.role) && (p.midi_clips || []).some((c) => c.track_id === t.id)).map((t) => t.role);
    const wanted = only ? [only] : roles.length ? roles : arranged.length ? arranged : ["drums", "bass", "guitar", "piano"];
    const written = [], kept = [];
    for (const role of wanted) {
      if (!ARR_ROLES.includes(role)) continue;
      const track = p.tracks.find((t) => t.role === role);
      if (track && track.locked) { kept.push(ROLE_HE[role]); continue; } // a locked track keeps what it has
      const tid = track ? track.id : newId("trk");
      if (!track) actions.push({ a: "add_track", track_id: tid, role }, ...startMix(st, tid, role));
      const clips = (p.midi_clips || []).filter((c) => c.track_id === tid);
      if (!only && clips.some((c) => !c.generator)) { kept.push(ROLE_HE[role]); continue; }
      const vars = clips.map((c) => c.generator).filter(Boolean).map((g) => parseInt(g.split(":").pop(), 10)).filter((x) => !Number.isNaN(x));
      const variation = only ? (vars.length ? Math.max(...vars) + 1 : 1) : 0;
      for (const c of clips) actions.push({ a: "remove_midi_clip", clip_id: c.id });
      actions.push({ a: "add_midi_clip", clip: noteClip(tid, C.arrangePart(role, chords, sections, st, melody, variation), `${st}:${variation}`) });
      written.push(ROLE_HE[role]);
    }
    if (!written.length && reuse) return reply("done", `לא שיניתי דבר — הערוצים ${listHe(kept)} נערכו על ידך ונשמרו. כדי להחליף אחד מהם, בקש למשל "גיטרה אחרת"`);
    try { run({ a: "batch", label: only ? "חלק חדש בעיבוד" : "עיבוד", actions }); } catch (e) { return reply("error", String(e)); }
    const names = [];
    for (const c of chords) { const n = C.chordName(c); if (names[names.length - 1] !== n) names.push(n); }
    names.length = Math.min(names.length, 8);
    let msg = only ? `${pre}נכתב ${listHe(written)} חדש. שאר העיבוד, המנגינה וההקלטה לא השתנו.`
      : newHarmony ? `${pre}הרמוניה חלופית: ${names.join(" ")}. המנגינה שלך לא השתנתה; ${listHe(written)} נכתבו מחדש לפי האקורדים.`
      : !written.length ? `${pre}${shiftMsg}נוצרה הרמוניה בסגנון ${C.STYLE_HE[st]}. אקורדים: ${names.join(" ")}. ההקלטה והמנגינה שלך לא השתנו.`
      : `${pre}${shiftMsg}נוצר עיבוד בסגנון ${C.STYLE_HE[st]}: ${listHe(written)}. אקורדים: ${names.join(" ")}. ההקלטה והמנגינה שלך לא השתנו.`;
    if (kept.length) msg += ` שמרתי את ${listHe(kept)} כפי שערכת.`;
    for (const n of notesMsg) msg += ` ${n}.`;
    if (!only) msg += ' אפשר לבקש: "תן לי בס אחר", "תעשה את זה רוק", "אקורדים אחרים".';
    return reply("done", msg);
  }
  // Starting mix of a track the arranger adds (as apps/core start_mix; CRR-13 step 1).
  function startMix(st, tid, role) {
    const m = root.HUMCompose.mixPreset(st, role);
    return [{ a: "gain_set", track_id: tid, db: m.gain_db }, { a: "set_pan", track_id: tid, pan: m.pan }, { a: "set_fx", track_id: tid, fx: { eq_low_db: m.eq_low_db, eq_mid_db: m.eq_mid_db, eq_high_db: m.eq_high_db, comp: m.comp, reverb: m.reverb } }];
  }
  function compose(style, major, bpm, roles, solo, lift = 0) {
    const C = root.HUMCompose;
    const st = style || "pop";
    const parts = roles.length ? roles.filter((r) => ARR_ROLES.includes(r)) : ["drums", "bass", "guitar", "piano"];
    // an instrumental solo (port of arrangement.rs compose_with)
    const soloPart = solo && solo.role && ARR_ROLES.includes(solo.role) ? solo.role : null;
    if (soloPart && !parts.includes(soloPart)) parts.push(soloPart);
    if (solo && !parts.some((r) => ["guitar", "piano", "winds", "strings"].includes(r))) parts.push("guitar");
    const form = solo ? (solo.at === "end" ? [["intro", 4], ["verse", 8], ["chorus", 8], ["verse", 8], ["chorus", 8], ["solo", 8], ["outro", 4]] : [["intro", 4], ["verse", 8], ["chorus", 8], ["verse", 8], ["chorus", 8], ["solo", 8], ["chorus", 8], ["outro", 4]]) : null;
    const seed = (Date.now() >>> 0) | 1;
    const song = C.composeSong({ style: st, key: null, bpm, major, roles: parts, seed, form, solo: soloPart, lift });
    api.new_project({ name: `שיר ${C.STYLE_HE[st]} חדש` });
    const key = { tonic: song.key.tonic, mode: song.key.major ? "major" : "minor" };
    const lead = newId("trk");
    const actions = [{ a: "set_tempo", bpm: song.bpm }, { a: "set_key", key }, { a: "set_harmony", harmony: toHarmony(st, song.chords, song.sections) }, { a: "add_track", track_id: lead, role: "lead" }, { a: "add_midi_clip", clip: noteClip(lead, song.melody, `compose:${seed}`) }, ...startMix(st, lead, "lead")];
    const names = [];
    for (const [r, notes] of song.parts) { const tid = newId("trk"); actions.push({ a: "add_track", track_id: tid, role: r }, ...startMix(st, tid, r)); actions.push({ a: "add_midi_clip", clip: noteClip(tid, notes, `${st}:0`) }); names.push(ROLE_HE[r]); }
    try { run({ a: "batch", label: "הלחנה", actions }); } catch (e) { return reply("error", String(e)); }
    const total = song.sections.reduce((m, x) => Math.max(m, x.start + x.dur), 0);
    const secs = song.sections.map((x) => C.SECTION_HE[x.kind]).join(" ← ");
    const liftHe = lift ? ` (הפזמון האחרון עולה ב${Math.abs(lift) === 1 ? "חצי טון" : `־${Math.abs(lift)} חצאי טונים`})` : "";
    return reply("done", `הולחן שיר מקורי בסגנון ${C.STYLE_HE[st]}: סולם ${keyLabel(key)}${liftHe}, קצב ${song.bpm.toFixed(0)}, ${secs} (${t2s(total).toFixed(0)} שניות). מנגינה בכלי + ${listHe(names)}. הכול ניתן לעריכה: אפשר ללחוץ על כל קליפ ולשנות תווים, לבקש גרסה אחרת לכל כלי, או "אקורדים אחרים".`);
  }

  // --- export the song to an audio file (port of apps/core/src/export.rs) ---
  const EXPORT_TAIL = 1.5, EXPORT_MAX = 15 * 60, CEILING = 0.98;
  let lastExport = null;
  function wav16(L, R, sr) {
    const n = L.length, b = new ArrayBuffer(44 + n * 4), v = new DataView(b);
    const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); str(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, n * 4, true);
    const q = (x) => Math.round(Math.max(-1, Math.min(1, x)) * 32767);
    for (let i = 0, o = 44; i < n; i++, o += 4) { v.setInt16(o, q(L[i]), true); v.setInt16(o + 2, q(R[i]), true); }
    return new Uint8Array(b);
  }
  /** WAV in three resolutions: 16-bit PCM (plays everywhere), 24-bit PCM (studio), 32-bit float (archive). */
  function wavEncode(L, R, sr, fmt) {
    if (fmt !== "wav24" && fmt !== "wav32f") return wav16(L, R, sr);
    const n = L.length, bps = fmt === "wav24" ? 3 : 4, b = new ArrayBuffer(44 + n * 2 * bps), v = new DataView(b);
    const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + n * 2 * bps, true); str(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, fmt === "wav32f" ? 3 : 1, true); v.setUint16(22, 2, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2 * bps, true); v.setUint16(32, 2 * bps, true); v.setUint16(34, bps * 8, true); str(36, "data"); v.setUint32(40, n * 2 * bps, true);
    let o = 44;
    const put = (x) => {
      if (fmt === "wav32f") { v.setFloat32(o, x, true); o += 4; return; }
      const q = Math.round(Math.max(-1, Math.min(1, x)) * 8388607);
      v.setUint8(o, q & 0xff); v.setUint8(o + 1, (q >> 8) & 0xff); v.setUint8(o + 2, (q >> 16) & 0xff); o += 3;
    };
    for (let i = 0; i < n; i++) { put(L[i]); put(R[i]); }
    return new Uint8Array(b);
  }
  const FMT_EXT = { wav16: "", wav24: " 24bit", wav32f: " 32bit-float" };
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(u8) { let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
  /** A one-file ZIP (stored, UTF-8 name) — the artifact viewer accepts .zip downloads but not .wav. */
  function zip1(name, data) {
    const nm = new TextEncoder().encode(name), crc = crc32(data), lh = new DataView(new ArrayBuffer(30)), ch = new DataView(new ArrayBuffer(46)), end = new DataView(new ArrayBuffer(22));
    [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [14, crc, 4], [18, data.length, 4], [22, data.length, 4], [26, nm.length, 2]].forEach(([o, x, w]) => (w === 4 ? lh.setUint32(o, x, true) : lh.setUint16(o, x, true)));
    [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [16, crc, 4], [20, data.length, 4], [24, data.length, 4], [28, nm.length, 2]].forEach(([o, x, w]) => (w === 4 ? ch.setUint32(o, x, true) : ch.setUint16(o, x, true)));
    const cdOff = 30 + nm.length + data.length;
    [[0, 0x06054b50, 4], [8, 1, 2], [10, 1, 2], [12, 46 + nm.length, 4], [16, cdOff, 4]].forEach(([o, x, w]) => (w === 4 ? end.setUint32(o, x, true) : end.setUint16(o, x, true)));
    return new Blob([lh.buffer, nm, data, ch.buffer, nm, end.buffer], { type: "application/zip" });
  }
  /** A many-file ZIP (stored, UTF-8 names): the stems download. */
  function zipN(files) {
    const parts = [], central = []; let off = 0;
    for (const [name, data] of files) {
      const nm = new TextEncoder().encode(name), crc = crc32(data), lh = new DataView(new ArrayBuffer(30)), ch = new DataView(new ArrayBuffer(46));
      [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [14, crc, 4], [18, data.length, 4], [22, data.length, 4], [26, nm.length, 2]].forEach(([o, x, w]) => (w === 4 ? lh.setUint32(o, x, true) : lh.setUint16(o, x, true)));
      [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [16, crc, 4], [20, data.length, 4], [24, data.length, 4], [28, nm.length, 2], [42, off, 4]].forEach(([o, x, w]) => (w === 4 ? ch.setUint32(o, x, true) : ch.setUint16(o, x, true)));
      parts.push(lh.buffer, nm, data); central.push(ch.buffer, nm); off += 30 + nm.length + data.length;
    }
    const cdLen = central.reduce((a, x) => a + (x.byteLength || x.length), 0), end = new DataView(new ArrayBuffer(22));
    [[0, 0x06054b50, 4], [8, files.length, 2], [10, files.length, 2], [12, cdLen, 4], [16, off, 4]].forEach(([o, x, w]) => (w === 4 ? end.setUint32(o, x, true) : end.setUint16(o, x, true)));
    return new Blob([...parts, ...central, end.buffer], { type: "application/zip" });
  }
  async function deliverBlob(fileName, blob) {
    const use = root.claude && typeof root.claude.use === "function" ? root.claude.use : null;
    if (use) {
      const dl = await use("downloads").catch(() => null);
      if (!dl) return { ok: false, msg: "ההורדה לא זמינה בדף הזה. באתר עצמו היא תעבוד" };
      try { await dl.save({ filename: fileName, data: blob }); return { ok: true }; } catch (e) { return { ok: false, msg: e && e.code === "declined" ? "ההורדה בוטלה" : "ההורדה לא הצליחה בדף הזה" }; }
    }
    const url = URL.createObjectURL(blob), el = document.createElement("a"); el.href = url; el.download = fileName; document.body.appendChild(el); el.click(); el.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    return { ok: true };
  }
  async function deliver(fileName, wavBytes) {
    // inside a claude.ai page: the downloads capability (allows .zip); on the real site: a normal download
    const use = root.claude && typeof root.claude.use === "function" ? root.claude.use : null;
    if (use) {
      const dl = await use("downloads").catch(() => null);
      if (!dl) return { ok: false, msg: "ההורדה לא זמינה בדף הזה. באתר עצמו היא תעבוד" };
      try { await dl.save({ filename: fileName.replace(/\.wav$/, ".zip"), data: zip1(fileName, wavBytes) }); return { ok: true, zipped: true }; }
      catch (e) { return { ok: false, msg: e && e.code === "declined" ? "ההורדה בוטלה" : "ההורדה לא הצליחה בדף הזה" }; }
    }
    const url = URL.createObjectURL(new Blob([wavBytes], { type: "audio/wav" }));
    const a = document.createElement("a"); a.href = url; a.download = fileName; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    return { ok: true, zipped: false };
  }
  // render the song (or one track: a stem) offline through the same chain playback uses
  async function renderMix(p, onlyTrack) {
    // an export waits for every recorded instrument the song uses, so nothing is exported with the stand-in synth
    const S = root.HUMSamples;
    if (S && S.state === "ready" && S.ensure) {
      const need = p.tracks.filter((t) => t.kind !== "audio" && (p.midi_clips || []).some((c) => c.track_id === t.id)).map((t) => (sampledFor(t), resolveInst(t)));
      await Promise.all(need.map((id) => S.ensure(id)));
      rendered.clear();
    }
    const sr = tr.ctx.sampleRate, aud = audible(p), items = [];
    let end = 0, any = false;
    p.tracks.forEach((t, i) => {
      const bufs = [];
      for (const c of (p.clips || []).filter((c) => c.track_id === t.id && !(c.edit && c.edit.mute))) bufs.push([clipBuffer(p, c), t2s(c.start_tick)]);
      for (const c of (p.midi_clips || []).filter((c) => c.track_id === t.id)) bufs.push([midiBuffer(t, c), t2s(c.start_tick)]);
      for (const [b, st] of bufs) if (b) { any = true; end = Math.max(end, st + b.duration); if (onlyTrack ? t.id === onlyTrack : aud[i]) items.push({ t, b, st }); }
    });
    if (!any) return { error: "עדיין אין מה לייצא: מקליטים, מוסיפים כלי או מייבאים קובץ" };
    if (!items.length) return { error: onlyTrack ? "בערוץ הזה אין עדיין צליל" : "השיר יוצא שקט לגמרי — אולי כל הערוצים מושתקים? בטל השתקה ונסה שוב" };
    const secs = Math.min(EXPORT_MAX, end + EXPORT_TAIL), oc = new OfflineAudioContext(2, Math.ceil(secs * sr), sr);
    const master = oc.createGain(); master.gain.value = 0.8 * Math.pow(10, (p.master_db || 0) / 20); master.connect(oc.destination);
    const nodes = new Map(), rbus = items.some(({ t }) => t.fx && t.fx.reverb > 0) ? reverbBusFor(oc, master) : null;
    const dbus = items.some(({ t }) => t.delay_send > 0) ? delayBusFor(oc, master) : null;
    if (root.HUMEffects && items.some(({ t }) => (t.inserts || []).some((x) => x.kind === "gate" && x.on !== false))) await root.HUMEffects.prepare(oc);
    const busIn = new Map();
    if (!onlyTrack) for (const t of p.tracks) if (t.kind === "bus") { // effect and group channels, as in playback
      const g = oc.createGain(); g.gain.value = t.mute ? 0 : Math.pow(10, t.gain_db / 20); if (!t.mute) automate(g.gain, t, 0, 0);
      let out = g; if (oc.createStereoPanner) { const pn = oc.createStereoPanner(); pn.pan.value = t.pan; g.connect(pn); out = pn; }
      out.connect(master); busIn.set(t.id, trackInput(oc, t, g, out, rbus, dbus));
    }
    for (const { t, b, st } of items) {
      if (!nodes.has(t.id)) {
        const g = oc.createGain(); g.gain.value = Math.pow(10, t.gain_db / 20); automate(g.gain, t, 0, 0);
        let out = g; if (oc.createStereoPanner) { const pn = oc.createStereoPanner(); pn.pan.value = t.pan; g.connect(pn); out = pn; }
        out.connect((t.output && busIn.get(t.output)) || master); nodes.set(t.id, trackInput(oc, t, g, out, rbus, dbus)); // the same effects and routing as playback
        for (const [bid, lv] of t.sends || []) { const bi = busIn.get(bid); if (bi && lv > 0) { const sg = oc.createGain(); sg.gain.value = lv; out.connect(sg); sg.connect(bi); } }
      }
      const src = oc.createBufferSource(); src.buffer = b; src.connect(nodes.get(t.id)); src.start(st);
    }
    let out;
    try { out = await oc.startRendering(); } catch { return { error: "הרינדור נכשל בדפדפן הזה" }; }
    return { L: out.getChannelData(0), R: out.getChannelData(1), sr, secs };
  }
  const interleave = (L, R) => { const v = new Float32Array(L.length * 2); for (let i = 0; i < L.length; i++) { v[2 * i] = L[i]; v[2 * i + 1] = R[i]; } return v; };
  const split = (v) => { const n = v.length >> 1, L = new Float32Array(n), R = new Float32Array(n); for (let i = 0; i < n; i++) { L[i] = v[2 * i]; R[i] = v[2 * i + 1]; } return [L, R]; };
  const fmtLu = (x) => (isFinite(x) ? x.toFixed(1) : "שקט");
  function safeName(n) { return (Array.from(n).map((c) => ('<>:"/\\|?*'.includes(c) || c.codePointAt(0) < 32 ? "_" : c)).join("").trim().replace(/^\.+|\.+$/g, "") || "שיר").slice(0, 80); }
  async function exportSong(confirmed, presetId, stems, format) {
    const fmt = FMT_EXT[format] !== undefined ? format : "wav16";
    if (!session) return reply("error", "אין פרויקט פתוח");
    const p = session.project;
    const check = (p.rights || []).filter((r) => r.user_statement === "licensed" || r.user_statement === "unknown");
    if (check.length && !confirmed) {
      pending = { proposed: { type: "export", confirmed: true, master: presetId || null, stems: !!stems, format: format || null }, utterance: "ייצוא", revision: p.revision };
      return reply("confirm", `בשיר יש חומר מיובא שהזכויות עליו לא אומתו (${check.map((r) => r.title || r.source).join(", ")}). אפשר לייצא לשימוש אישי ולתרגול; לפני פרסום, מכירה או שיתוף ציבורי צריך לוודא שמותר. לייצא לשימוש אישי?`);
    }
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    const safe = safeName(p.name);
    if (stems) {
      // each audible-or-not track with sound, alone, same level and pan as in the mix (no mastering)
      const files = [];
      for (const t of p.tracks) {
        const r = await renderMix(p, t.id);
        if (r.error) continue;
        let pk = 0; for (let i = 0; i < r.L.length; i++) pk = Math.max(pk, Math.abs(r.L[i]), Math.abs(r.R[i]));
        if (pk > CEILING) { const g = CEILING / pk; for (let i = 0; i < r.L.length; i++) { r.L[i] *= g; r.R[i] *= g; } }
        files.push([`${safe} - ${safeName(t.name)}${FMT_EXT[fmt]}.wav`, wavEncode(r.L, r.R, r.sr, fmt)]);
      }
      if (!files.length) return reply("error", "עדיין אין ערוצים עם צליל לייצא");
      const zipName = `${safe} - ערוצים נפרדים.zip`, blob = zipN(files);
      lastExport = { name: zipName, stems: files.map((f) => f[0]), bytes: blob.size };
      const d = await deliverBlob(zipName, blob);
      if (!d.ok) return reply("error", d.msg);
      return reply("done", `יוצאו ${files.length} ערוצים נפרדים בקובץ אחד: ${zipName}. כל כלי בקובץ שמע משלו, באותו אורך, כדי שיתיישרו בכל תוכנה.`);
    }
    const r = await renderMix(p, null);
    if (r.error) return reply("error", r.error);
    const { sr, secs } = r; let L = r.L, R = r.R;
    let note = "", report = null;
    const P = presetId && root.HUMLoudness && root.HUMLoudness.preset(presetId);
    if (P) {
      const v = interleave(L, R);
      report = root.HUMLoudness.master(v, sr, P);
      [L, R] = split(v);
      note = ` מאסטרינג "${P.name_he}": עוצמה ${fmtLu(report.after.integrated)} LUFS (יעד ${P.target_lufs}), שיא ${fmtLu(report.after.true_peak)} dBTP.` + (report.max_reduction_db > 6 ? ` הוגבל ב-${report.max_reduction_db.toFixed(1)}dB — כדאי לשקול יעד שקט יותר.` : "");
    } else {
      let peak = 0; for (let i = 0; i < L.length; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
      if (peak > CEILING) { const g = CEILING / peak; for (let i = 0; i < L.length; i++) { L[i] *= g; R[i] *= g; } note = ` העוצמה הונמכה ב-${(20 * Math.log10(peak / CEILING)).toFixed(1)}dB כדי למנוע עיוות.`; }
    }
    const bytes = wavEncode(L, R, sr, fmt);
    const fileName = `${safe}${P ? " (מאסטר)" : ""}${FMT_EXT[fmt]}.wav`;
    let peak = 0; for (let i = 0; i < L.length; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
    // quality check of exactly what is exported (master spec J, part 4 §42.3); the report can be downloaded
    const qc = root.HUMLoudness.qcReport(root.HUMLoudness.analyse(interleave(L, R), sr), P || null, sr, fmt);
    lastQc = { name: `${fileName.replace(/\.wav$/, "")} - דוח בדיקה.txt`, text: root.HUMLoudness.qcText(qc), report: qc };
    lastExport = { name: fileName, bytes: bytes.length, seconds: secs, sampleRate: sr, peak, report, qc };
    const d = await deliver(fileName, bytes);
    if (!d.ok) return reply("error", d.msg);
    return reply("done", `השיר יוצא לקובץ שמע: ${fileName} (${secs.toFixed(0)} שניות, ${(bytes.length / 1048576).toFixed(1)}MB)${d.zipped ? " — בתוך קובץ ZIP, כי דף הבדיקה מאפשר רק סוגי קבצים מסוימים" : ""}.${note} בדיקת איכות: ${qc.summary}. את הדוח המלא אפשר להוריד בלשונית מאסטר. אפשר לשמוע את השיר בכל נגן ולשלוח אותו.`);
  }
  let lastQc = null;
  async function qcCheck(presetId, format) {
    if (!session) return reply("error", "אין פרויקט פתוח");
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    const fmt = FMT_EXT[format] !== undefined ? format : "wav16";
    const r = await renderMix(session.project, null);
    if (r.error) return reply("error", "עדיין אין מה לבדוק. זמזם מנגינה או הלחן שיר");
    const v = interleave(r.L, r.R), P = presetId && root.HUMLoudness.preset(presetId);
    if (P) root.HUMLoudness.master(v, r.sr, P);
    else { let pk = 0; for (let i = 0; i < v.length; i++) pk = Math.max(pk, Math.abs(v[i])); if (pk > CEILING) { const g = CEILING / pk; for (let i = 0; i < v.length; i++) v[i] *= g; } }
    const qc = root.HUMLoudness.qcReport(root.HUMLoudness.analyse(v, r.sr), P || null, r.sr, fmt);
    lastQc = { name: `${safeName(session.project.name)}${P ? " (מאסטר)" : ""} - דוח בדיקה.txt`, text: root.HUMLoudness.qcText(qc), report: qc };
    return reply("done", lastQc.text);
  }
  // before/after listening at matched loudness (spec 41.5): the louder version never wins by being louder
  let preview = null, previewSrc = null;
  async function masterPreview(presetId) {
    if (!session) return reply("error", "אין פרויקט פתוח");
    if (!ensureCtx()) return reply("error", "הדפדפן לא מאפשר שמע");
    const P = root.HUMLoudness && root.HUMLoudness.preset(presetId);
    if (!P) return reply("error", "פריסט מאסטרינג לא מוכר");
    const r = await renderMix(session.project, null);
    if (r.error) return reply("error", r.error);
    const v = interleave(r.L, r.R), orig = v.slice();
    const rep = root.HUMLoudness.master(v, r.sr, P);
    const mk = (x) => { const [L, R] = split(x), b = tr.ctx.createBuffer(2, L.length, r.sr); b.copyToChannel(L, 0); b.copyToChannel(R, 1); return b; };
    const match = isFinite(rep.before.integrated) && isFinite(rep.after.integrated) ? Math.pow(10, (rep.before.integrated - rep.after.integrated) / 20) : 1;
    preview = { before: mk(orig), after: mk(v), match, rep };
    return { ...reply("done", `מאסטרינג "${P.name_he}" מוכן להשוואה: לפני ${fmtLu(rep.before.integrated)} LUFS, אחרי ${fmtLu(rep.after.integrated)} LUFS, שיא ${fmtLu(rep.after.true_peak)} dBTP${rep.max_reduction_db > 0.1 ? `, הגבלה עד ${rep.max_reduction_db.toFixed(1)}dB` : ""}. ההשוואה מושמעת באותה עוצמה, כדי שהחזק לא ינצח רק כי הוא חזק.`), report: rep };
  }
  function previewPlay(which) {
    if (previewSrc) { try { previewSrc.stop(); } catch { /* ended */ } previewSrc = null; }
    if (which === "stop" || !preview) return reply("done", "stop");
    stop();
    const src = tr.ctx.createBufferSource(), g = tr.ctx.createGain();
    src.buffer = which === "after" ? preview.after : preview.before;
    g.gain.value = which === "after" ? preview.match : 1;
    src.connect(g).connect(tr.ctx.destination); src.start(); previewSrc = src;
    return reply("done", which === "after" ? "משמיע אחרי מאסטרינג (באותה עוצמה)" : "משמיע לפני מאסטרינג");
  }


  // --- lawful backing for existing songs (port of apps/core/src/import.rs) ---
  const STATEMENT_HE = { own_work: "יצירה שלך", licensed: "יש לך רישיון", public_domain: "נחלת הכלל", unknown: "מקור לא ידוע" };
  const cut = (x, n) => Array.from(String(x)).slice(0, n).join("");
  function importMidiBytes(fileName, bytes, rights) {
    const M = root.HUMMidi;
    if (!rights || !["own_work", "licensed", "public_domain", "unknown"].includes(rights.statement)) return reply("ask", "מאיפה הקובץ? יצרת אותו, יש לך רישיון, זה בנחלת הכלל, או שהמקור לא ידוע?");
    let s;
    try { s = M.parse(bytes); } catch (e) { return reply("error", String(e)); }
    const parts = M.planParts(s);
    const chans = s.tracks.reduce((n, t) => n + new Set(t.notes.map((x) => x.channel)).size, 0), dropped = Math.max(0, chans - parts.length);
    const base = cut(Array.from(String(fileName).split(/[\\/]/).pop()).filter((c) => c.codePointAt(0) >= 32 && !(c.codePointAt(0) >= 127 && c.codePointAt(0) < 160)).join(""), 120);
    const dot = base.lastIndexOf("."), stem = dot >= 0 ? base.slice(0, dot) : base;
    const title = rights.title && rights.title.trim() ? rights.title : null;
    api.new_project({ name: title || stem || "שיר מיובא" });
    const actions = [];
    const bpm = s.tempo_us ? Math.round((60000000 / s.tempo_us) * 10) / 10 : null;
    if (bpm !== null && bpm >= 40 && bpm <= 240) actions.push({ a: "set_tempo", bpm });
    const key = M.keyOf(s);
    if (key) actions.push({ a: "set_key", key });
    const clipIds = []; let count = 0;
    for (const pt of parts) {
      const tid = newId("trk");
      actions.push({ a: "add_track", track_id: tid, role: pt.role, name: cut(pt.name, 80) });
      const notes = pt.notes.slice(0, 20000).map(([pitch, st, dur, vel]) => ({ pitch, start_tick: st, dur_ticks: dur, velocity: vel, confidence: 1 }));
      count += notes.length;
      const id = newId("mid"); clipIds.push(id);
      actions.push({ a: "add_midi_clip", clip: { id, track_id: tid, start_tick: 0, notes, source_clip: null, generator: null } });
    }
    const rid = newId("rgt"), own = rights.statement === "own_work";
    actions.push({ a: "set_rights", id: rid, record: { id: rid, asset_type: "midi_file", assets: clipIds, source: cut(base, 200), title: title ? cut(title, 200) : null,
      artist: rights.artist && rights.artist.trim() ? cut(rights.artist, 200) : null, user_statement: rights.statement,
      license_identifier: rights.license && rights.license.trim() ? cut(rights.license, 200) : "TO_VERIFY",
      commercial_use_allowed: own ? true : null, redistribution_allowed: own ? true : null, evidence_uri: null, verified_at: null, recorded_at: new Date().toISOString() } });
    try { run({ a: "batch", label: "ייבוא קובץ תווים", actions }); } catch (e) { return reply("error", String(e)); }
    const p = session.project;
    let msg = `יובא הקובץ "${base}": ${parts.length} ערוצים, ${count} תווים — ${parts.map((x) => `${x.name} (${ROLE_HE[x.role]})`).join(", ")}. קצב ${p.bpm.toFixed(0)}, סולם ${keyLabel(p.key)}.`;
    if (!key) msg += " הסולם לא מצוין בקובץ — הוא ייקבע לפי המנגינה בעיבוד הראשון.";
    if (s.time_signature && !(s.time_signature[0] === 4 && s.time_signature[1] === 4)) msg += ` שים לב: השיר במשקל ${s.time_signature[0]}/${s.time_signature[1]}; העיבוד האוטומטי עדיין בנוי ל-4/4.`;
    if (dropped > 0) msg += ` ${dropped} ערוצים קטנים לא יובאו (מקסימום ${M.MAX_IMPORT_TRACKS}).`;
    if (!parts.some((x) => x.role === "lead")) msg += " לא זוהתה מנגינה ראשית, אז עיבוד אוטומטי לא יופעל עד שתסמן ערוץ כמנגינה.";
    msg += ` זכויות: ${STATEMENT_HE[rights.statement]}.`;
    if (!["own_work", "public_domain"].includes(rights.statement)) msg += " אפשר לנגן, לערוך ולתרגל; לפני שיתוף, הפצה או שימוש מסחרי צריך לוודא שמותר.";
    msg += ' אפשר עכשיו: "תעבד את זה ים תיכוני", להחליף כלים, או לערוך כל ערוץ.';
    return reply("done", msg);
  }
  async function importMidi(file, rights) {
    let bytes;
    try { bytes = new Uint8Array(await file.arrayBuffer()); } catch { return reply("error", "לא הצלחתי לקרוא את הקובץ"); }
    return importMidiBytes(file.name, bytes, rights);
  }
  function existingSong(title, artist) {
    const what = title && artist ? `לשיר "${title}" של ${artist}` : title ? `לשיר "${title}"` : "לשיר קיים";
    return reply("ask", `אשמח להכין פלייבק ${what}. אני לא מוריד שירים מהאינטרנט ולא עוקף הגנות של אתרי האזנה — זה מוגן בזכויות יוצרים. יש שלוש דרכים חוקיות: (1) לזמזם או לשיר את המנגינה, ואני אבנה עליה עיבוד; (2) להעלות קובץ תווים (MIDI) שיש לך זכות להשתמש בו — כפתור "ייבוא קובץ תווים"; (3) להעלות הקלטה שלך או קובץ שמע שיש לך רישיון עליו. בכל מקרה העיבוד שייווצר הוא ביצוע חדש, והמקור נרשם בפרויקט.`);
  }

  // the recordings on c's track whose time overlaps c's (a take folder)
  function clipSpan(p, c) { const m = (p.media || []).find((x) => x.id === c.media_id) || {}, e = c.edit || {}, sr = m.sample_rate || 48000, len = Math.min(Math.max(0, (m.frames || 0) - (e.trim_start || 0)), e.trim_len || Infinity) / sr, s = t2s(c.start_tick); return [s, s + len]; }
  function takeGroup(p, c) {
    const [a, b] = clipSpan(p, c);
    return (p.clips || []).filter((x) => x.track_id === c.track_id && (x.id === c.id || (() => { const [s, e] = clipSpan(p, x); return s < b - 0.05 && e > a + 0.05; })()));
  }
  function clipJson(p, c, order) {
    const m = (p.media || []).find((x) => x.id === c.media_id) || {};
    const a = audioStore.get(c.media_id);
    if (a && !a.peaks) a.peaks = peaksOf(a.pcm, a.sr);
    const e = c.edit || { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0 }, sr = m.sample_rate || 48000, total = m.frames || 0;
    const frames = Math.min(Math.max(0, total - e.trim_start), e.trim_len || Infinity);
    const pk = a ? a.peaks : [], bps = pk.length / Math.max(1e-9, total / sr), skip = Math.floor((e.trim_start / sr) * bps);
    const takes = takeGroup(p, c).length;
    return { id: c.id, order, takes, muted: !!e.mute, start: Math.round(t2s(c.start_tick) * 1000) / 1000, start_tick: c.start_tick, length: Math.round((frames / sr) * 1000) / 1000, total: Math.round((total / sr) * 1000) / 1000, edit: { ...e, mute: !!e.mute, gain_db: (e.gain_cdb || 0) / 100 }, sr,
      peaks: (() => { const g = Math.pow(10, (e.gain_cdb || 0) / 2000), v = pk.slice(skip, skip + Math.ceil((frames / sr) * bps)); return g === 1 ? v : v.map((x) => x * g); })() };
  }
  function midiJson(c) {
    const r3 = (x) => Math.round(x * 1000) / 1000;
    const end = c.notes.reduce((m, n) => Math.max(m, n.start_tick + n.dur_ticks), 0);
    return { id: c.id, start: r3(t2s(c.start_tick)), start_tick: c.start_tick, length: r3(t2s(end)), source: c.source_clip || null,
      notes: c.notes.map((n) => [n.pitch, r3(t2s(n.start_tick)), r3(t2s(n.dur_ticks)), n.velocity, Math.round(n.confidence * 100) / 100, n.start_tick, n.dur_ticks]) };
  }
  function state() {
    const p = session && session.project;
    const secs = Math.round(nowSeconds() * 1000) / 1000;
    const bpm = p ? p.bpm : 120;
    const beats = Math.floor(secs * bpm / 60 + 1e-9);
    let master = 0;
    const tracks = p ? p.tracks.map((t, i) => {
      const meter = meterOf(t.id); master = Math.max(master, meter);
      return { ...t, sound: t.kind === "midi" && sampledFor(t) ? "recorded" : "synth", instrument: t.instrument || "auto", instrument_name: INSTRUMENT_HE[resolveInst(t)], audible: audible(p)[i],
        clips: (p.clips || []).map((c, k) => [c, k]).filter(([c]) => c.track_id === t.id).map(([c, k]) => clipJson(p, c, k)),
        midi: (p.midi_clips || []).filter((c) => c.track_id === t.id).map(midiJson), meter };
    }) : [];
    const recSecs = rec ? Math.round((rec.frames / rec.sr) * 10) / 10 : 0;
    const recPeak = rec ? rec.peak : 0;
    if (rec) rec.peak = 0;
    return {
      project: p ? { id: p.id, name: p.name, revision: p.revision, bpm: p.bpm, meter: p.meter || [4, 4], master_db: p.master_db || 0, key: keyLabel(p.key), key_tonic: p.key.tonic, key_mode: p.key.mode, tracks, harmony: harmonyView(p), markers: (p.markers || []).map((m) => ({ id: m.id, tick: m.tick, start: Math.round(t2s(m.tick) * 1000) / 1000, name: m.name })), meta: p.meta || {} } : null,
      can_undo: !!(session && session.undo.length), can_redo: !!(session && session.redo.length),
      transport: { playing: tr.playing, metronome: tr.metronome, click: tr.clickPlay, seconds: secs, loop: tr.loop ? { start: tr.loop.start, end: tr.loop.end } : null, bar: Math.floor(beats / 4) + 1, beat: (beats % 4) + 1 },
      pending: pending ? pending.utterance : null,
      master_meter: master,
      recording: rec ? { active: true, counting: !rec.started, track_id: rec.trackId, start: Math.round((rec.startSec || 0) * 1000) / 1000, seconds: recSecs, input_peak: recPeak, device: rec.device, synthetic: false } : { active: false },
      input_available: !!(typeof navigator !== "undefined" && navigator.mediaDevices),
      monitor: monitorOn, monitor_available: true,
      instrument_credits: root.HUMSamples && root.HUMSamples.state === "ready" ? Object.values(root.HUMSamples.sets).map((x) => x.credit).sort() : [],
      vocal: { testing: !!vocalRec, saved: !!savedProfile(), range: currentRange() },
      notes: [],
    };
  }
  const reply = (status, message) => ({ status, message, state: state() });

  function run(action) {
    if (!session) throw "אין פרויקט פתוח. צור פרויקט חדש או פתח פרויקט קיים";
    const snap = JSON.stringify(session);
    const inv = apply(session.project, action);
    session.project.revision++;
    session.redo = []; session.undo.push({ action, inverse: inv });
    if (!persist()) { session = JSON.parse(snap); throw "השינוי לא נשמר ולכן בוטל: אין מקום באחסון הדפדפן"; }
    afterChange();
    refreshPlayback();
  }
  // ---- versions (port of apps/core versions) ----
  let changes = 0;
  const AUTO_EVERY = 20, AUTO_KEEP = 10;
  function snapList(pid) { return (db.snapshots[pid] = db.snapshots[pid] || []); }
  function makeSnapshot(name, kind) {
    const p = session.project;
    const n = Array.from(String(name).trim()).slice(0, 80).join("");
    const info = { id: newId("snp"), project_id: p.id, name: n || "גרסה ללא שם", kind, revision: p.revision, created_at: new Date().toISOString() };
    snapList(p.id).unshift({ info, state: JSON.parse(JSON.stringify(p)) });
    if (!memoryOnly) saveAll(db);
    return info;
  }
  function afterChange() {
    if (++changes < AUTO_EVERY) return;
    changes = 0;
    makeSnapshot("נקודת שחזור אוטומטית", "auto");
    const list = snapList(session.project.id);
    let autos = 0;
    db.snapshots[session.project.id] = list.filter((s) => s.info.kind !== "auto" || ++autos <= AUTO_KEEP);
    if (!memoryOnly) saveAll(db);
  }
  const fmtDb = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
  function diff(o, n) {
    const d = [];
    if (o.name !== n.name) d.push(`שם הפרויקט: עכשיו "${o.name}" · בגרסה "${n.name}"`);
    if (o.bpm !== n.bpm) d.push(`קצב: עכשיו ${o.bpm.toFixed(0)} · בגרסה ${n.bpm.toFixed(0)}`);
    if (o.key.tonic !== n.key.tonic || o.key.mode !== n.key.mode) d.push(`סולם: עכשיו ${keyLabel(o.key)} · בגרסה ${keyLabel(n.key)}`);
    for (const t of o.tracks) {
      const m = n.tracks.find((x) => x.id === t.id);
      if (!m) { d.push(`ערוץ ${t.name} — קיים עכשיו, לא קיים בגרסה`); continue; }
      const ch = [];
      if (m.name !== t.name) ch.push(`בגרסה נקרא "${m.name}"`);
      if (m.gain_db !== t.gain_db) ch.push(`עוצמה עכשיו ${fmtDb(t.gain_db)} · בגרסה ${fmtDb(m.gain_db)} דציבל`);
      if (m.pan !== t.pan) ch.push("מיקום סטריאו שונה");
      if (m.mute !== t.mute) ch.push(m.mute ? "מושתק בגרסה" : "מושתק עכשיו");
      if (m.solo !== t.solo) ch.push(m.solo ? "סולו בגרסה" : "סולו עכשיו");
      if ((m.instrument || "auto") !== (t.instrument || "auto")) ch.push(`כלי עכשיו: ${INSTRUMENT_HE[t.instrument || "auto"]} · בגרסה: ${INSTRUMENT_HE[m.instrument || "auto"]}`);
      if (JSON.stringify(m.fx || FX_OFF) !== JSON.stringify(t.fx || FX_OFF)) ch.push(`אפקטים עכשיו: ${fxHebrew(t.fx || FX_OFF)} · בגרסה: ${fxHebrew(m.fx || FX_OFF)}`);
      if (ch.length) d.push(`${t.name}: ${ch.join(", ")}`);
    }
    for (const m of n.tracks) if (!o.tracks.some((t) => t.id === m.id)) d.push(`ערוץ ${m.name} — קיים בגרסה, לא קיים עכשיו`);
    if (JSON.stringify(o.harmony || null) !== JSON.stringify(n.harmony || null)) {
      const nm = (h) => ((h && h.chords) || []).slice(0, 8).map((c) => root.HUMCompose.chordName(c)).join(" ");
      d.push(`עיבוד: עכשיו ${(o.harmony && o.harmony.style) || "ללא"} [${nm(o.harmony)}] · בגרסה ${(n.harmony && n.harmony.style) || "ללא"} [${nm(n.harmony)}]`);
    }
    return d.length ? d : ["אין הבדלים"];
  }
  function findSnap(id) {
    if (!session) throw "אין פרויקט פתוח";
    const s = snapList(session.project.id).find((x) => x.info.id === id);
    if (!s) throw "הגרסה לא נמצאה";
    return s;
  }
  function history(isUndo) {
    if (!session) return reply("error", "אין פרויקט פתוח");
    const from = isUndo ? session.undo : session.redo;
    if (!from.length) return reply("error", isUndo ? "אין פעולה לבטל" : "אין פעולה לבצע שוב");
    const e = from.pop();
    const inv = apply(session.project, isUndo ? e.inverse : e.action);
    session.project.revision++;
    (isUndo ? session.redo : session.undo).push(isUndo ? e : { action: e.action, inverse: inv });
    persist();
    afterChange();
    refreshPlayback();
    return reply("done", isUndo ? "הפעולה האחרונה בוטלה" : "הפעולה בוצעה שוב");
  }
  function execProposed(pr, summary) {
    if (pr.type === "undo") return history(true);
    if (pr.type === "redo") return history(false);
    if (pr.type === "play") { play(); return reply("done", summary); }
    if (pr.type === "stop") { if (rec) return recordStop(); if (vocalRec) return vocalStop(); stop(); return reply("done", summary); }
    if (pr.type === "rec") return recordStart(pr.role);
    if (pr.type === "recstop") return recordStop();
    if (pr.type === "transcribe") return transcribe();
    if (pr.type === "quantize") return quantize();
    if (pr.type === "arrange") return arrange(pr.style, pr.roles, pr.only, pr.new_harmony);
    if (pr.type === "compose") return compose(pr.style, pr.major, pr.bpm, pr.roles, pr.solo || null, pr.lift || 0);
    if (pr.type === "existing") return existingSong(pr.title, pr.artist);
    if (pr.type === "export") return exportSong(pr.confirmed, pr.master || null, !!pr.stems, pr.format || null);
    if (pr.type === "vocal_start") return vocalStart();
    if (pr.type === "vocal_stop") return vocalStop();
    if (pr.type === "recommend_key") return recommendKey();
    if (pr.type === "vocal_save") return vocalSave();
    if (pr.type === "vocal_delete") return vocalDelete();
    try {
      if (pr.type === "add") run({ a: "add_track", track_id: newId("trk"), role: pr.role });
      else run(pr.action);
      return reply("done", summary);
    } catch (e) { return reply("error", String(e)); }
  }
  // every setting of a project, in plain Hebrew, for its card in the project list (spec addendum י.3)
  function projectDetails(p, updated) {
    const m = p.meta || {}, out = [];
    const st = (p.harmony && STYLE_HE[p.harmony.style]) || STYLE_HE[m.style];
    out.push(["סגנון", st || "עוד לא נבחר"]);
    const inst = [...new Set(p.tracks.filter((t) => !["melody", "lead", "vocals"].includes(t.role)).map((t) => ROLE_HE[t.role] || t.name))];
    out.push(["כלים", inst.length ? inst.join(", ") : "עוד לא נבחרו"]);
    const voc = p.tracks.find((t) => t.role === "vocals");
    out.push(["שירה", voc ? "יש ערוץ שירה" : "בלי"]);
    if (m.source) out.push(["הקלטה", m.source]);
    if (m.headphones) out.push(["אוזניות", m.headphones === "true" ? "כן" : "לא"]);
    const fx = voc && voc.fx ? [voc.fx.reverb > 0 && "הד", voc.fx.comp > 0 && "דחיסה", voc.fx.eq_high_db > 0 && "צליל בהיר"].filter(Boolean) : [];
    out.push(["אפקטים", fx.length ? fx.join(", ") : "בלי"]);
    out.push(["קצב", `${p.bpm} פעימות לדקה`]);
    out.push(["סולם", keyLabel(p.key)]);
    const lead = p.tracks.find((t) => t.role === "lead");
    out.push(["מנגינה", lead && (lead.midi || []).some((c) => (c.notes || []).length) ? "יש" : "עוד לא"]);
    const d = new Date(updated); if (!isNaN(d.getTime())) out.push(["עודכן", d.toLocaleString("he-IL", { dateStyle: "short", timeStyle: "short" })]);
    return out;
  }
  function openSession(p) { session = { project: p, undo: [], redo: [] }; pending = null; changes = 0; stop(); tr.offset = 0; }

  const api = {
    state,
    list_projects() {
      return Object.values(db.projects).sort((a, b) => b.updated_at.localeCompare(a.updated_at)).map((r) => ({ id: r.project.id, title: r.project.name, updated_at: r.updated_at, tracks: r.project.tracks.length,
        style: (r.project.harmony && STYLE_HE[r.project.harmony.style]) || STYLE_HE[(r.project.meta || {}).style] || "", roles: [...new Set(r.project.tracks.filter((t) => !["melody", "lead"].includes(t.role)).map((t) => ROLE_HE[t.role] || t.name))].slice(0, 4),
        details: projectDetails(r.project, r.updated_at) }));
    },
    // the answers of the opening questions are kept in the project at once (spec addendum י.2)
    set_meta(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const m = (session.project.meta = session.project.meta || {});
      for (const k of ["style", "headphones", "source"]) if (a[k] !== undefined) m[k] = String(a[k]);
      return persist() ? reply("done", "נשמר") : reply("error", "השינוי לא נשמר: אין מקום באחסון הדפדפן");
    },
    new_project(a) {
      const name = (a.name || "").trim() || "שיר חדש";
      openSession({ id: newId("prj"), name: Array.from(name).slice(0, 80).join(""), revision: 0, bpm: 120, key: { tonic: 9, mode: "minor" }, tracks: [] });
      persist();
      return reply("done", `נוצר פרויקט חדש: ${name}`);
    },
    open_project(a) {
      const r = db.projects[a.id];
      if (!r) return reply("error", "הפרויקט לא נמצא");
      openSession(JSON.parse(JSON.stringify(r.project)));
      return loadMedia(session.project).then(() => reply("done", `נפתח הפרויקט: ${r.project.name}`));
    },
    command_text(a) {
      if (!session) return reply("error", "אין פרויקט פתוח. צור פרויקט חדש כדי להתחיל");
      const r = interpretWith(a.text || "", session.project, a.track_id || null);
      if (r.kind === "ready") { pending = null; return execProposed(r.proposed, r.msg); }
      if (r.kind === "confirm") { pending = { proposed: r.proposed, utterance: a.text, revision: session.project.revision }; return reply("confirm", r.msg); }
      return reply(r.kind === "ask" ? "ask" : "unsupported", r.msg);
    },
    confirm(a) {
      if (!pending) return reply("error", "אין פעולה שממתינה לאישור");
      const p = pending; pending = null;
      if (a.yes !== "true") return reply("done", "בוטל. לא בוצע שינוי");
      if (!session || session.project.revision !== p.revision) return reply("error", "הפרויקט השתנה מאז ההצעה. כתוב את הבקשה שוב");
      return execProposed(p.proposed, "בוצע על כל השיר");
    },
    undo: () => history(true),
    redo: () => history(false),
    transport(a) {
      if (a.op === "play") play(); else if (a.op === "stop") stop(); else if (a.op === "rewind") { const was = tr.playing; stop(); tr.offset = 0; if (was) play(); }
      return reply("done", a.op);
    },
    seek(a) {
      const sec = Number(a.seconds);
      if (!(sec >= 0)) return reply("error", "מקום לא תקין");
      const was = tr.playing; stop(); tr.offset = sec; if (was) play();
      return reply("done", "seek");
    },
    set_loop(a) {
      const s0 = Number(a.start), s1 = Number(a.end);
      if (a.start === undefined || a.end === undefined || a.start === "" || a.end === "") { tr.loop = null; tr.prev = null; return reply("done", "הלולאה כובתה"); }
      if (!(s0 >= 0 && s1 - s0 >= 0.05)) return reply("error", "אזור הלולאה קצר מדי");
      tr.loop = { start: Math.round(s0 * 1000) / 1000, end: Math.round(s1 * 1000) / 1000 };
      return reply("done", "לולאה פעילה");
    },
    add_track(a) { try { run({ a: "add_track", track_id: newId("trk"), role: a.role }); return reply("done", "add_track"); } catch (e) { return reply("error", String(e)); } },
    set_gain(a) { try { run({ a: "gain_set", track_id: a.track_id, db: Number(a.db) }); return reply("done", "set_track_gain"); } catch (e) { return reply("error", String(e)); } },
    set_mute(a) { try { run({ a: "set_mute", track_id: a.track_id, mute: a.on === "true" }); return reply("done", "set_mute"); } catch (e) { return reply("error", String(e)); } },
    set_fx(a) {
      try {
        const t = session && session.project.tracks.find((x) => x.id === a.track_id);
        if (!t) return reply("error", "ערוץ לא נמצא");
        const cur = t.fx || FX_OFF;
        const g = (k, d) => (a[k] !== undefined && a[k] !== "" ? Number(a[k]) : d);
        run({ a: "set_fx", track_id: a.track_id, fx: { eq_low_db: g("low", cur.eq_low_db), eq_mid_db: g("mid", cur.eq_mid_db), eq_high_db: g("high", cur.eq_high_db), comp: g("comp", cur.comp), reverb: g("reverb", cur.reverb) } });
        return reply("done", "set_fx");
      } catch (e) { return reply("error", String(e)); }
    },
    set_pan(a) { try { run({ a: "set_pan", track_id: a.track_id, pan: Number(a.pan) }); return reply("done", "set_track_pan"); } catch (e) { return reply("error", String(e)); } },
    remove_track(a) {
      if (!(a.approved === true || a.approved === "true")) return reply("needs_approval", "מחיקת ערוץ דורשת אישור");
      // a folder or an effect channel first frees what points at it (one undo step; router.rs remove_track)
      const acts = [];
      for (const t of (session && session.project.tracks) || []) {
        if (t.output === a.track_id) acts.push({ a: "set_track_output", track_id: t.id, bus_id: null });
        if ((t.sends || []).some((x) => x[0] === a.track_id)) acts.push({ a: "set_track_send", track_id: t.id, bus_id: a.track_id, level: 0 });
      }
      acts.push({ a: "remove_track", track_id: a.track_id });
      try { run(acts.length === 1 ? acts[0] : { a: "batch", label: "מחיקת ערוץ", actions: acts }); return reply("done", "remove_track"); } catch (e) { return reply("error", String(e)); }
    },
    rename_track(a) { try { run({ a: "rename_track", track_id: a.track_id, name: String(a.name ?? "") }); return reply("done", "rename_track"); } catch (e) { return reply("error", String(e)); } },
    set_track_color(a) { try { run({ a: "set_track_color", track_id: a.track_id, color: a.color && a.color !== "null" ? a.color : null }); return reply("done", "set_track_color"); } catch (e) { return reply("error", String(e)); } },
    // the studio's "new track" dialog: one undo step that makes the track exactly as chosen (router.rs create_track)
    create_track(a) {
      if (!session) api.new_project({ name: "שיר חדש" });
      if (!ROLE_HE[a.role]) return reply("error", "סוג ערוץ לא מוכר");
      const id = newId("trk"), acts = [{ a: "add_track", track_id: id, role: a.role, name: a.name ? String(a.name) : undefined }];
      if (a.kind && a.kind !== defaultKind(a.role)) acts.push({ a: "set_track_kind", track_id: id, kind: a.kind });
      if (a.instrument && a.instrument !== "auto") acts.push({ a: "set_instrument", track_id: id, instrument: a.instrument });
      if (a.color) acts.push({ a: "set_track_color", track_id: id, color: a.color });
      if (a.group === true || a.group === "true") acts.push({ a: "set_track_group", track_id: id, group: true });
      if (a.armed === true || a.armed === "true") acts.push({ a: "set_track_arm", track_id: id, armed: true });
      try { run({ a: "batch", label: "ערוץ חדש", actions: acts }); const r = reply("done", "create_track"); r.track_id = id; return r; } catch (e) { return reply("error", String(e)); }
    },
    // light, frequent state for the studio's playhead and meters (no clip data)
    pulse() {
      const p = session && session.project, meters = {};
      let master = 0;
      if (p) for (const t of p.tracks) { const m = meterOf(t.id); meters[t.id] = m; master = Math.max(master, m); }
      const recPeak = rec ? rec.peak : 0; if (rec) rec.peak = 0;
      return { seconds: Math.round(nowSeconds() * 1000) / 1000, playing: tr.playing, revision: p ? p.revision : -1, project_id: p ? p.id : null, meters, master,
        recording: rec ? { active: true, midi: !!rec.midi, counting: !rec.started, track_id: rec.trackId, start: rec.startSec || 0, seconds: rec.midi ? Math.max(0, nowSeconds() - rec.startSec) : rec.frames / rec.sr, input_peak: recPeak, notes: rec.midi ? rec.notes.length + rec.open.size : 0 } : { active: false } };
    },
    set_track_inserts(a) {
      try { const ins = typeof a.inserts === "string" ? JSON.parse(a.inserts) : a.inserts; run({ a: "set_track_inserts", track_id: a.track_id, inserts: ins }); return reply("done", "set_track_inserts"); } catch (e) { return reply("error", String(e)); }
    },
    set_delay_send(a) { try { run({ a: "set_track_delay_send", track_id: a.track_id, level: Number(a.level) }); return reply("done", "set_delay_send"); } catch (e) { return reply("error", String(e)); } },
    live_note(a) { return liveNote(a.track_id, Number(a.pitch), Number(a.velocity || 96), a.on === true || a.on === "true"); },
    // vocal tuning (tune.js): a tuned copy of the recording becomes what the clip plays; undo brings back the original
    tune_clip(a) {
      if (!session || !root.HUMTune) return reply("error", "כוונון קול לא זמין");
      const p = session.project, c = (p.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "ההקלטה לא נמצאה");
      const d = audioStore.get(c.media_id); if (!d) return reply("error", "קובץ ההקלטה לא נמצא");
      const mode = a.mode === "chromatic" ? "chromatic" : p.key.mode;
      let r; try { r = root.HUMTune.tune(d.pcm, d.sr, { tonic: p.key.tonic, mode, strength: Number(a.strength ?? 0.8), speed_ms: Number(a.speed_ms ?? 40) }); } catch (e) { return reply("error", "הכוונון נכשל: " + e); }
      if (!r.frames_corrected) return reply("error", "לא נמצאה שירה בהקלטה הזאת");
      const mediaId = newId("rec"); audioStore.set(mediaId, { pcm: r.pcm, sr: d.sr }); idbPut(mediaId, { pcm: r.pcm, sr: d.sr });
      const m0 = (p.media || []).find((x) => x.id === c.media_id) || {};
      try { run({ a: "set_clip_media", clip_id: c.id, media_id: mediaId, media: { id: mediaId, frames: r.pcm.length, sample_rate: d.sr, origin: "tuned", source: m0.id || null } }); }
      catch (e) { return reply("error", String(e)); }
      return reply("done", `השירה כוונה לסולם ${keyLabel(p.key)} (תיקון ממוצע ${Math.round(r.mean_cents)} סנט). Ctrl+Z מחזיר את המקור`);
    },
    // drag a region to another track of the same kind (one undo step)
    move_region(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = session.project, t = p.tracks.find((x) => x.id === a.track_id), st = Math.max(0, Math.round(Number(a.start_tick) || 0));
      const c = (p.clips || []).find((x) => x.id === a.clip_id), m = (p.midi_clips || []).find((x) => x.id === a.clip_id);
      if (!t || (!c && !m)) return reply("error", "הקטע או הערוץ לא נמצאו");
      if (c && t.kind !== "audio") return reply("error", "הקלטה מעבירים לערוץ שמע");
      if (m && t.kind !== "midi") return reply("error", "קטע תווים מעבירים לערוץ של כלי");
      const nid = newId(c ? "clp" : "mcl");
      const acts = c ? [{ a: "remove_clip", clip_id: c.id }, { a: "add_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: nid, track_id: t.id, start_tick: st } }]
        : [{ a: "remove_midi_clip", clip_id: m.id }, { a: "add_midi_clip", clip: { ...JSON.parse(JSON.stringify(m)), id: nid, track_id: t.id, start_tick: st } }];
      try { run({ a: "batch", label: "העברת קטע", actions: acts }); const r = reply("done", "move_region"); r.clip_id = nid; return r; } catch (e) { return reply("error", String(e)); }
    },
    set_track_output(a) { try { run({ a: "set_track_output", track_id: a.track_id, bus_id: a.bus_id && a.bus_id !== "null" ? a.bus_id : null }); return reply("done", "set_track_output"); } catch (e) { return reply("error", String(e)); } },
    set_track_send(a) { try { run({ a: "set_track_send", track_id: a.track_id, bus_id: a.bus_id, level: Number(a.level) }); return reply("done", "set_track_send"); } catch (e) { return reply("error", String(e)); } },
    set_master_gain(a) { try { run({ a: "set_master_gain", db: Number(a.db) }); return reply("done", "set_master_gain"); } catch (e) { return reply("error", String(e)); } },
    paste_clip(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const c = (session.project.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "ההקלטה לא נמצאה");
      const tid = a.track_id || c.track_id, t = session.project.tracks.find((x) => x.id === tid);
      if (!t || t.kind !== "audio") return reply("error", "מדביקים הקלטה בערוץ שמע");
      try { run({ a: "add_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("clp"), track_id: tid, start_tick: s2t(Math.max(0, Number(a.at_sec) || 0)) } }); return reply("done", "paste_clip"); } catch (e) { return reply("error", String(e)); }
    },
    // the undo list, newest first, in Hebrew (Edit → history)
    history_list() {
      if (!session) return [];
      const HE = { add_track: "ערוץ חדש", remove_track: "מחיקת ערוץ", rename_track: "שינוי שם ערוץ", gain_set: "עוצמת ערוץ", gain_delta: "עוצמת ערוץ", set_pan: "פאן", set_mute: "השתקה", set_solo: "סולו", set_fx: "אפקטים מהירים", set_track_inserts: "אפקטים", set_track_delay_send: "שליחה לדיליי",
        add_clip: "הקלטה / הדבקה", remove_clip: "מחיקת הקלטה", set_clip_edit: "עריכת הקלטה", set_clip_start: "הזזת הקלטה", add_midi_clip: "קטע תווים חדש", remove_midi_clip: "מחיקת קטע תווים", set_midi_notes: "עריכת תווים", set_midi_clip_start: "הזזת קטע תווים",
        set_tempo: "קצב", set_key: "סולם", set_meter: "משקל", set_marker: "סמן", remove_marker: "מחיקת סמן", set_track_automation: "אוטומציה", set_track_arm: "דריכה להקלטה", set_track_color: "צבע", set_track_lock: "נעילה", move_track: "הזזת ערוץ", set_instrument: "החלפת כלי", set_master_gain: "עוצמה ראשית", transpose: "טרנספוזיציה", rename_project: "שם הפרויקט" };
      return session.undo.slice().reverse().map((e, i) => ({ steps: i + 1, label: e.action.a === "batch" ? e.action.label : HE[e.action.a] || "שינוי" }));
    },
    set_meter(a) { try { run({ a: "set_meter", numerator: Math.round(Number(a.numerator)), denominator: Math.round(Number(a.denominator)) }); return reply("done", "set_meter"); } catch (e) { return reply("error", String(e)); } },
    set_marker(a) { try { run({ a: "set_marker", marker: { id: a.id || newId("mrk"), tick: Math.round(Number(a.tick)), name: String(a.name ?? "") } }); return reply("done", "set_marker"); } catch (e) { return reply("error", String(e)); } },
    remove_marker(a) { try { run({ a: "remove_marker", id: a.id }); return reply("done", "remove_marker"); } catch (e) { return reply("error", String(e)); } },
    set_automation(a) {
      try {
        const pts = String(a.points || "").split(";").filter((x) => x.trim()).map((x) => { const [t, d] = x.split(":").map(Number); if (!Number.isFinite(t) || !Number.isFinite(d)) throw "נקודות אוטומציה לא תקינות"; return [Math.round(t), d]; });
        run({ a: "set_track_automation", track_id: a.track_id, points: pts }); return reply("done", "set_automation");
      } catch (e) { return reply("error", String(e)); }
    },
    set_track_arm(a) { try { run({ a: "set_track_arm", track_id: a.track_id, armed: a.on === true || a.on === "true" }); return reply("done", "set_track_arm"); } catch (e) { return reply("error", String(e)); } },
    set_track_kind(a) { try { run({ a: "set_track_kind", track_id: a.track_id, kind: a.kind }); return reply("done", "set_track_kind"); } catch (e) { return reply("error", String(e)); } },
    set_track_lock(a) { try { run({ a: "set_track_lock", track_id: a.track_id, locked: a.on === true || a.on === "true" }); return reply("done", "set_track_lock"); } catch (e) { return reply("error", String(e)); } },
    move_track(a) { const to = Number(a.to); if (!(to >= 0)) return reply("error", "מקום חסר"); try { run({ a: "move_track", track_id: a.track_id, to_index: Math.floor(to) }); return reply("done", "move_track"); } catch (e) { return reply("error", String(e)); } },
    duplicate_track(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = session.project, i = p.tracks.findIndex((t) => t.id === a.track_id);
      if (i < 0) return reply("error", "ערוץ לא נמצא");
      const t = p.tracks[i], nid = newId("trk"), name = Array.from(`${t.name} (עותק)`).slice(0, 80).join("");
      const actions = [{ a: "add_track", track_id: nid, role: t.role, name }, { a: "move_track", track_id: nid, to_index: i + 1 }, { a: "gain_set", track_id: nid, db: t.gain_db }, { a: "set_pan", track_id: nid, pan: t.pan }, { a: "set_fx", track_id: nid, fx: { ...(t.fx || FX_OFF) } }, { a: "set_instrument", track_id: nid, instrument: t.instrument || "auto" }, { a: "set_track_color", track_id: nid, color: t.color ?? null }];
      if (t.mute) actions.push({ a: "set_mute", track_id: nid, mute: true });
      if (t.kind !== defaultKind(t.role)) actions.push({ a: "set_track_kind", track_id: nid, kind: t.kind });
      if ((t.inserts || []).length) actions.push({ a: "set_track_inserts", track_id: nid, inserts: JSON.parse(JSON.stringify(t.inserts)) });
      if (t.delay_send > 0) actions.push({ a: "set_track_delay_send", track_id: nid, level: t.delay_send });
      if ((t.automation || []).length) actions.push({ a: "set_track_automation", track_id: nid, points: t.automation.map((x) => [...x]) });
      for (const [bid, lv] of t.sends || []) actions.push({ a: "set_track_send", track_id: nid, bus_id: bid, level: lv });
      if (t.output) actions.push({ a: "set_track_output", track_id: nid, bus_id: t.output });
      if (t.group) actions.push({ a: "set_track_group", track_id: nid, group: true });
      for (const c of (p.midi_clips || []).filter((c) => c.track_id === t.id)) actions.push({ a: "add_midi_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("mcl"), track_id: nid } });
      for (const c of (p.clips || []).filter((c) => c.track_id === t.id)) actions.push({ a: "add_clip", clip: { ...c, id: newId("clp"), track_id: nid } });
      try { run({ a: "batch", label: `שכפול ${t.name}`, actions }); return reply("done", "duplicate_track"); } catch (e) { return reply("error", String(e)); }
    },
    set_midi_clip_start(a) { try { run({ a: "set_midi_clip_start", clip_id: a.clip_id, start_tick: Math.round(Number(a.start_tick)) }); return reply("done", "set_midi_clip_start"); } catch (e) { return reply("error", String(e)); } },
    set_clip_edit(a) {
      const c = session && (session.project.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "ההקלטה לא נמצאה");
      const cur = c.edit || { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0 };
      const u = (k, d) => (a[k] === undefined || a[k] === "" ? d : Math.max(0, Math.round(Number(a[k]))));
      const edit = { trim_start: u("trim_start", cur.trim_start), trim_len: a.trim_len === "null" ? null : a.trim_len === undefined ? cur.trim_len : Math.round(Number(a.trim_len)), fade_in_ms: u("fade_in_ms", cur.fade_in_ms), fade_out_ms: u("fade_out_ms", cur.fade_out_ms), mute: a.mute === undefined ? !!cur.mute : a.mute === true || a.mute === "true",
        gain_cdb: a.gain_db === undefined || a.gain_db === "" ? cur.gain_cdb | 0 : Math.round(Number(a.gain_db) * 100) };
      try { run({ a: "set_clip_edit", clip_id: c.id, start_tick: a.start_tick === undefined ? c.start_tick : Math.round(Number(a.start_tick)), edit }); return reply("done", "set_clip_edit"); } catch (e) { return reply("error", String(e)); }
    },
    set_clip_start(a) { try { run({ a: "set_clip_start", clip_id: a.clip_id, start_tick: Math.round(Number(a.start_tick)) }); return reply("done", "set_clip_start"); } catch (e) { return reply("error", String(e)); } },
    // copies one after another, the clip's length rounded up to whole bars (as apps/core duplicate_midi_clip)
    duplicate_midi_clip(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const c = (session.project.midi_clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "קליפ התווים לא נמצא");
      const bar = 960 * 4, end = c.notes.reduce((m, n) => Math.max(m, n.start_tick + n.dur_ticks), 0) || bar;
      const len = Math.max(1, Math.ceil(end / bar)) * bar, times = Math.min(16, Math.max(1, Math.floor(Number(a.times) || 1)));
      const actions = []; for (let k = 1; k <= times; k++) actions.push({ a: "add_midi_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("mcl"), start_tick: c.start_tick + k * len } });
      try { run({ a: "batch", label: times === 1 ? "שכפול קטע" : `חזרה על הקטע ${times} פעמים`, actions }); return reply("done", "duplicate_midi_clip"); } catch (e) { return reply("error", String(e)); }
    },
    // cut a note clip in two at a song time; notes crossing the cut continue in the second part (as apps/core split_midi_clip)
    split_midi_clip(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const c = (session.project.midi_clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "קליפ התווים לא נמצא");
      const at = Math.round(Number(a.at_tick)), cut = at - c.start_tick, end = c.notes.reduce((m, n) => Math.max(m, n.start_tick + n.dur_ticks), 0);
      if (!(cut > 0 && cut < end)) return reply("error", "סמן הנגינה צריך להיות בתוך הקטע כדי לחתוך אותו");
      const A = [], B = [];
      for (const n of c.notes) {
        if (n.start_tick >= cut) B.push({ ...n, start_tick: n.start_tick - cut });
        else if (n.start_tick + n.dur_ticks > cut) { A.push({ ...n, dur_ticks: cut - n.start_tick }); B.push({ ...n, start_tick: 0, dur_ticks: n.start_tick + n.dur_ticks - cut }); }
        else A.push({ ...n });
      }
      try { run({ a: "batch", label: "חיתוך קטע", actions: [{ a: "set_midi_notes", clip_id: c.id, notes: A }, { a: "add_midi_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("mcl"), start_tick: at, notes: B } }] }); return reply("done", "split_midi_clip"); } catch (e) { return reply("error", String(e)); }
    },
    set_solo(a) { try { run({ a: "set_solo", track_id: a.track_id, solo: a.on === "true" }); return reply("done", "set_solo"); } catch (e) { return reply("error", String(e)); } },
    rename_project(a) { try { run({ a: "rename_project", name: a.name || "" }); persist(); return reply("done", "rename_project"); } catch (e) { return reply("error", String(e)); } },
    set_tempo(a) { try { run({ a: "set_tempo", bpm: Number(a.bpm) }); return reply("done", "set_tempo"); } catch (e) { return reply("error", String(e)); } },
    snapshot_create(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const i = makeSnapshot(a.name || "", "manual");
      return reply("done", `נשמרה גרסה: ${i.name}`);
    },
    snapshots() { return session ? snapList(session.project.id).map((s) => s.info) : []; },
    snapshot_diff(a) {
      try { const s = findSnap(a.id); return reply("done", diff(session.project, s.state).join("\n")); } catch (e) { return reply("error", String(e)); }
    },
    snapshot_restore(a) {
      try {
        const s = findSnap(a.id);
        if (a.approved !== "true") return reply("confirm", `שחזור הגרסה ישנה את:\n${diff(session.project, s.state).join("\n")}\nאפשר לבטל אחר כך. לשחזר?`);
        run({ a: "restore_snapshot", label: s.info.name, state: s.state });
        return reply("done", `שוחזרה הגרסה "${s.info.name}". אפשר לבטל עם "ביטול"`);
      } catch (e) { return reply("error", String(e)); }
    },
    duplicate_project() {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = JSON.parse(JSON.stringify(session.project));
      p.id = newId("prj"); p.revision = 0; p.name = Array.from(`${session.project.name} (עותק)`).slice(0, 80).join("");
      const map = {};
      p.tracks.forEach((t) => { const id = newId("trk"); map[t.id] = id; t.id = id; });
      const clipMap = {};
      (p.clips || []).forEach((c) => { const id = newId("clp"); clipMap[c.id] = id; c.id = id; c.track_id = map[c.track_id] || c.track_id; });
      const midiMap = {};
      (p.midi_clips || []).forEach((c) => { const nid = newId("mid"); midiMap[c.id] = nid; c.id = nid; c.track_id = map[c.track_id] || c.track_id; c.source_clip = clipMap[c.source_clip] || null; });
      (p.rights || []).forEach((r) => { r.assets = (r.assets || []).map((x) => midiMap[x] || x); });
      openSession(p); persist();
      return reply("done", `נוצר עותק: ${p.name}`);
    },
    record_start(a) { const pu = a.punch_start !== undefined && a.punch_end !== undefined ? [Number(a.punch_start), Number(a.punch_end)] : null; return recordStart(a.role || null, a.track_id || null, a.count_in, pu); },
    // cut a recording in two at a song time; same media, nothing re-recorded (lib.rs split_clip)
    split_clip(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = session.project, c = (p.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "ההקלטה לא נמצאה");
      const m = (p.media || []).find((x) => x.id === c.media_id) || {}, e = { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0, mute: false, gain_cdb: 0, ...(c.edit || {}) }, sr = m.sample_rate || 48000;
      const avail = Math.max(0, (m.frames || 0) - e.trim_start), frames = e.trim_len ? Math.min(e.trim_len, avail) : avail, st = t2s(c.start_tick), cut = Math.round((Number(a.at_sec) - st) * sr);
      if (!(cut >= 0.05 * sr && cut <= frames - 0.05 * sr)) return reply("error", "סמן הנגינה צריך להיות בתוך ההקלטה כדי לחתוך אותה");
      const actions = [{ a: "set_clip_edit", clip_id: c.id, start_tick: c.start_tick, edit: { ...e, trim_len: cut, fade_out_ms: 0 } },
        { a: "add_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("clp"), start_tick: s2t(st + cut / sr), edit: { ...e, trim_start: e.trim_start + cut, trim_len: frames - cut, fade_in_ms: 0 } } }];
      try { run({ a: "batch", label: "חיתוך הקלטה", actions }); return reply("done", "split_clip"); } catch (err) { return reply("error", String(err)); }
    },
    duplicate_clip(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = session.project, c = (p.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "ההקלטה לא נמצאה");
      const [, end] = clipSpan(p, c);
      try { run({ a: "add_clip", clip: { ...JSON.parse(JSON.stringify(c)), id: newId("clp"), start_tick: s2t(end) } }); return reply("done", "duplicate_clip"); } catch (err) { return reply("error", String(err)); }
    },
    // clip gain so the loudest sample of the kept part reaches -1 dBFS; the recording is not changed (lib.rs normalize_clip)
    normalize_clip(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = session.project, c = (p.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "ההקלטה לא נמצאה");
      const d = audioStore.get(c.media_id); if (!d) return reply("error", "קובץ ההקלטה לא נמצא");
      const e = { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0, mute: false, gain_cdb: 0, ...(c.edit || {}) };
      const off = Math.min(d.pcm.length, e.trim_start), end = e.trim_len ? Math.min(d.pcm.length, off + e.trim_len) : d.pcm.length;
      let pk = 0; for (let i = off; i < end; i++) pk = Math.max(pk, Math.abs(d.pcm[i]));
      if (pk < 1e-5) return reply("error", "ההקלטה שקטה לגמרי");
      const g = Math.max(-4800, Math.min(2400, Math.round((-1 - 20 * Math.log10(pk)) * 100)));
      try { run({ a: "set_clip_edit", clip_id: c.id, start_tick: c.start_tick, edit: { ...e, gain_cdb: g } }); return reply("done", `העוצמה הותאמה (${g > 0 ? "+" : ""}${(g / 100).toFixed(1)} dB)`); } catch (err) { return reply("error", String(err)); }
    },
    // take folders: the takes are the recordings on the track that overlap this one; exactly the chosen one is heard
    choose_take(a) {
      if (!session) return reply("error", "אין פרויקט פתוח");
      const p = session.project, c = (p.clips || []).find((x) => x.id === a.clip_id);
      if (!c) return reply("error", "הטייק לא נמצא");
      const group = takeGroup(p, c), actions = [];
      for (const x of group) { const e = x.edit || { trim_start: 0, trim_len: null, fade_in_ms: 0, fade_out_ms: 0 }, m = x.id !== c.id; if (!!e.mute !== m) actions.push({ a: "set_clip_edit", clip_id: x.id, start_tick: x.start_tick, edit: { ...e, mute: m } }); }
      if (!actions.length) return reply("done", "הטייק הזה כבר נבחר");
      try { run({ a: "batch", label: "בחירת טייק", actions }); return reply("done", "הטייק נבחר"); } catch (e) { return reply("error", String(e)); }
    },
    set_metronome(a) { tr.metronome = tr.clickPlay = a.on === true || a.on === "true"; return reply("done", tr.metronome ? "מטרונום פועל" : "מטרונום כבוי"); },
    record_stop() { return recordStop(); },
    export_audio(a) { return exportSong(a.confirmed === "true", a.master || null, a.stems === "true", a.format || null); },
    master_preview(a) { return masterPreview(a.preset || "streaming"); },
    qc_report(a) { return qcCheck(a.master || null, a.format || null); },
    async qc_download() {
      if (!lastQc) return reply("error", "עדיין אין דוח. לחץ קודם על: בדיקת איכות");
      const d = await deliverBlob(lastQc.name, new Blob(["\ufeff" + lastQc.text], { type: "text/plain;charset=utf-8" }));
      return d.ok ? reply("done", `הדוח ירד: ${lastQc.name}`) : reply("error", d.msg);
    },
    master_play(a) { return previewPlay(a.which || "stop"); },
    vocal_test_start() { return vocalStart(); },
    coach_review(a) { return coachReview(a.level || "easy", a.accuracy); },
    guide_tone(a) { return guideTone(Number(a.midi), Number(a.seconds || 1.2)); },
    guide_result(a) { try { return guideResult(JSON.parse(a.attempts || "[]")); } catch { return reply("error", "תוצאות הבדיקה לא תקינות"); } },
    vocal_test_stop() { return vocalStop(); },
    recommend_key() { return recommendKey(); },
    save_vocal_profile() { return vocalSave(); },
    delete_vocal_profile() { return vocalDelete(); },
    transpose(a) { try { run({ a: "transpose", semitones: Number(a.semitones) }); return reply("done", "transpose"); } catch (e) { return reply("error", String(e)); } },
    transcribe(a) { return transcribe(a && a.from_sec ? Number(a.from_sec) : 0); },
    quantize() { return quantize(); },
    arrange(a) { return arrange(a.style || null, (a.roles || "").split(",").filter(Boolean), a.only || null, a.new_harmony === "true"); },
    compose(a) { return compose(a.style || null, a.mood === "happy" ? true : a.mood === "sad" ? false : null, a.bpm ? Number(a.bpm) : null, (a.roles || "").split(",").filter(Boolean)); },
    remove_clip(a) { try { run({ a: "remove_clip", clip_id: a.clip_id }); return reply("done", "remove_clip"); } catch (e) { return reply("error", String(e)); } },
    remove_midi_clip(a) { try { run({ a: "remove_midi_clip", clip_id: a.clip_id }); return reply("done", "remove_midi_clip"); } catch (e) { return reply("error", String(e)); } },
    set_instrument(a) { try { run({ a: "set_instrument", track_id: a.track_id, instrument: a.instrument }); return reply("done", "set_instrument"); } catch (e) { return reply("error", String(e)); } },
    // an instrument part to draw in (piano roll / step sequencer): a new note clip, possibly empty (router.rs new_midi_clip)
    new_midi_clip(a) {
      try {
        const notes = String(a.notes || "").split(";").filter((x) => x.trim()).map((it) => { const f = it.split(",").map(Number); return { pitch: f[0], start_tick: f[1], dur_ticks: f[2], velocity: f[3], confidence: 1 }; });
        const id = newId("mcl");
        run({ a: "add_midi_clip", clip: { id, track_id: a.track_id, start_tick: Math.max(0, Math.round(Number(a.start_tick) || 0)), notes, source_clip: null, generator: null } });
        const r = reply("done", "new_midi_clip"); r.clip_id = id; return r;
      } catch (e) { return reply("error", String(e)); }
    },
    set_midi_notes(a) {
      try {
        const notes = String(a.notes || "").split(";").filter((x) => x.trim()).map((it) => {
          const f = it.split(",").map((x) => x.trim());
          if (f.length < 4 || f.slice(0, 4).some((x) => !/^-?\d+$/.test(x))) throw "רשימת תווים לא תקינה";
          return { pitch: +f[0], start_tick: +f[1], dur_ticks: +f[2], velocity: +f[3], confidence: f[4] !== undefined && !Number.isNaN(+f[4]) ? +f[4] : 1 };
        });
        const c = session && (session.project.midi_clips || []).find((x) => x.id === a.clip_id);
        if (c && c.generator) run({ a: "batch", label: "עריכת תווים", actions: [{ a: "set_midi_notes", clip_id: a.clip_id, notes }, { a: "set_clip_generator", clip_id: a.clip_id, generator: null }] });
        else run({ a: "set_midi_notes", clip_id: a.clip_id, notes });
        return reply("done", "set_midi_notes");
      } catch (e) { return reply("error", String(e)); }
    },
    set_monitor(a) { monitorOn = a.on === "true"; return reply("done", monitorOn ? "הקשבה עצמית פועלת: בזמן הקלטה תשמע את עצמך. השתמש באוזניות כדי למנוע צפצוף" : "הקשבה עצמית כבויה"); },
    async list_inputs() {
      try {
        const all = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
        return all.map((d, i) => ({ id: d.deviceId, name: d.label || `מיקרופון ${i + 1}` }));
      } catch { return []; }
    },
    set_input(a) { try { if (a.id) localStorage.setItem(INPUT_KEY, a.id); else localStorage.removeItem(INPUT_KEY); } catch { /* private window */ } return reply("done", "התקן הקלט נבחר"); },
    delete_project(a) {
      const r = db.projects[a.id];
      if (!r) return reply("error", "הפרויקט לא נמצא");
      if (a.approved !== "true") return reply("confirm", `למחוק את הפרויקט "${r.project.name}"? אי אפשר לבטל את המחיקה`);
      const name = r.project.name;
      delete db.projects[a.id]; delete db.snapshots[a.id];
      if (session && session.project.id === a.id) {
        stop(); session = null;
        const next = Object.values(db.projects).sort((x, y) => y.updated_at.localeCompare(x.updated_at))[0];
        if (next) openSession(JSON.parse(JSON.stringify(next.project)));
      }
      if (!memoryOnly) saveAll(db);
      return reply("done", `הפרויקט "${name}" נמחק`);
    },
    list_bundles() { return []; },
    export_bundle() { return { status: "error", message: "יצוא לקובץ זמין בתוכנה המותקנת" }; },
    import_bundle() { return { status: "error", message: "ייבוא מקובץ זמין בתוכנה המותקנת" }; },
    set_key(a) { try { run({ a: "set_key", key: { tonic: Number(a.tonic), mode: a.mode === "major" ? "major" : "minor" } }); return reply("done", "set_key"); } catch (e) { return reply("error", String(e)); } },
  };

  const LAB = {
    call(cmd, args) {
      const f = api[cmd];
      if (!f) return { status: "error", message: "פקודה לא מוכרת" };
      try { return f(args || {}); } catch (e) { return reply("error", String(e)); }
    },
    memoryOnly: () => memoryOnly,
    importAudio,
    importPcm,
    importMidi,
    _importMidiBytes: importMidiBytes,
    _lastExport: () => lastExport,
    vocalFile,
    _vocalFromSamples: vocalFromSamples,
    _test: { interpret, canon, keyLabel,
      // a vocal take for the coach tests (as if recorded with the "שיר" button)
      addVocal(pcm, sr) { if (!session) return null; let t = session.project.tracks.find((x) => x.role === "vocals"); if (!t) { run({ a: "add_track", track_id: newId("trk"), role: "vocals" }); t = session.project.tracks.find((x) => x.role === "vocals"); } return addRecording(pcm, sr, t.id, 0, "recording"); } },
  };
  if (typeof window !== "undefined") window.__HUM_LAB__ = LAB;
  if (root.HUMSamples) {
    root.HUMSamples.onChange(() => { rendered.clear(); refreshPlayback(); });
    root.HUMSamples.load();
  }
  if (typeof module !== "undefined") module.exports = LAB;
})();

;
/* HUM studio — the professional workspace (docs/specs/studio-architecture-he.md).
 * One screen with fixed zones, like every recording program: control bar, inspector, tracks + timeline, library,
 * one bottom panel. Every control calls one engine command (window.__HUM_LAB__.call) and shows only what the
 * engine really did. Nothing is added to the song unless the user asks for it. The AI assistant is frozen until
 * the studio base is approved. */
(() => {
  "use strict";
  const LAB = window.__HUM_LAB__;
  const E = window.HUMEffects;
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const TPB = 960; // ticks per beat (domain/project)

  // ---------------- engine access ----------------
  let S = null; // last full state
  const ui = { sel: null, region: null, tool: "ptr", zoom: 18, snap: 4, count: true, cycle: null, bottom: "mixer", libTab: "inst", libCat: "הכל", pulse: null, pr: { clip: null, sel: null } };
  try { const k = JSON.parse(localStorage.getItem("hum-studio") || "{}"); Object.assign(ui, { zoom: k.zoom || ui.zoom, snap: k.snap ?? ui.snap, count: k.count ?? ui.count, bottom: k.bottom || ui.bottom, punch: k.punch || null }); } catch { /* private window */ }
  const saveUi = () => { try { localStorage.setItem("hum-studio", JSON.stringify({ zoom: ui.zoom, snap: ui.snap, count: ui.count, bottom: ui.bottom, punch: ui.punch || null })); } catch { /* fine */ } };

  async function call(cmd, args = {}, quiet = false) {
    let r;
    try { r = await Promise.resolve(LAB.call(cmd, args)); } catch (e) { r = { status: "error", message: String(e) }; }
    if (r && r.state) S = r.state;
    if (r && r.status === "error" && !quiet) toast(r.message, true);
    if (r && r.status === "confirm" && !quiet) {
      const ok = await ask("לאשר?", r.message, null);
      if (ok) return call("confirm", { yes: "true" });
      await call("confirm", { yes: "false" }, true);
    }
    return r || { status: "error" };
  }
  const refresh = () => { S = LAB.call("state", {}); renderAll(); };
  async function act(cmd, args, msg) { const r = await call(cmd, args); if (r.status === "done") { refresh(); if (msg) toast(msg); } return r; }

  // ---------------- small ui helpers ----------------
  function toast(t, bad) { if (!t) return; const e = $("#toast"); e.textContent = t; e.style.borderColor = bad ? "var(--rec)" : "var(--gold2)"; e.classList.add("on"); clearTimeout(toast.t); toast.t = setTimeout(() => e.classList.remove("on"), bad ? 4200 : 2600); }
  function ask(title, msg, value) {
    return new Promise((res) => {
      $("#ask-t").textContent = title; $("#ask-m").textContent = msg || ""; const inp = $("#ask-in");
      inp.style.display = value === null ? "none" : ""; inp.value = value || "";
      const done = (v) => { $("#ask").classList.remove("on"); $("#ask-ok").onclick = $("#ask-no").onclick = null; inp.onkeydown = null; res(v); };
      $("#ask-ok").onclick = () => done(value === null ? true : inp.value.trim());
      $("#ask-no").onclick = () => done(value === null ? false : null);
      inp.onkeydown = (e) => { if (e.key === "Enter") $("#ask-ok").click(); if (e.key === "Escape") $("#ask-no").click(); };
      $("#ask").classList.add("on"); if (value !== null) setTimeout(() => inp.focus(), 30);
    });
  }
  let popEl = null;
  function pop(x, y, html, bind) {
    closePop(); popEl = document.createElement("div"); popEl.className = "pop"; popEl.innerHTML = html; document.body.appendChild(popEl);
    const r = popEl.getBoundingClientRect(); popEl.style.left = Math.min(x, innerWidth - r.width - 8) + "px"; popEl.style.top = Math.min(y, innerHeight - r.height - 8) + "px";
    bind(popEl); setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
  }
  function outside(e) { if (popEl && !popEl.contains(e.target)) closePop(); }
  function closePop() { if (popEl) { popEl.remove(); popEl = null; document.removeEventListener("pointerdown", outside, true); } }

  const IC = {
    mic: '<path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    guitar: '<path d="M19 3l2 2-4.5 4.5"/><path d="M15 7l2 2"/><path d="M11.5 9.5a4 4 0 0 0-5.3.3c-1.2 1.2-.9 2.4-1.7 3.2S2 14 3 17s4 4 7 4 1.8-2.3 3-3.5 2-.5 3.2-1.7a4 4 0 0 0 .3-5.3"/><circle cx="10" cy="15" r="1.5"/>',
    bass: '<path d="M20 2l2 2-6 6"/><path d="M13 9l2 2"/><path d="M10 11a4 4 0 0 0-5 1c-1.5 1.5 0 3-1.5 4.5S5 22 9 21s3-3.5 4.5-5 2-3 1-5"/>',
    drums: '<ellipse cx="12" cy="9" rx="8" ry="3"/><path d="M4 9v6c0 1.7 3.6 3 8 3s8-1.3 8-3V9"/><path d="M6 3l4 5M18 3l-4 5"/>',
    piano: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M8 5v8M12 5v8M16 5v8M3 13h18"/>',
    synth: '<rect x="2" y="7" width="20" height="11" rx="2"/><circle cx="7" cy="11" r="1.4"/><circle cx="12" cy="11" r="1.4"/><path d="M5 15h14"/>',
    strings: '<path d="M14 3c-3 3-4 6-3 9-3 0-5 2-5 4.5S8 21 10 21s4-1.5 4-4c3 1 6 0 9-3"/><path d="M3 21l8-8"/>',
    flute: '<path d="M3 17 17 3l4 4L7 21z"/><path d="M8 14l1 1M11 11l1 1M14 8l1 1"/>',
    oud: '<circle cx="9" cy="15" r="6"/><path d="M13 11 21 3"/><circle cx="9" cy="15" r="1.5"/>',
    fx: '<path d="M4 7h10M4 17h6M18 4v6M15 7h6"/><circle cx="16" cy="17" r="3"/>',
    play: '<path d="M8 5v14l11-7z"/>', plus: '<path d="M12 5v14M5 12h14"/>', more: '<circle cx="12" cy="5" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="12" cy="19" r="1.4"/>',
    ptr: '<path d="M5 3l14 7-6 2-2 6z"/>', pen: '<path d="M4 20l4-1L19 8l-3-3L5 16z"/>', auto: '<path d="M3 17l5-6 4 3 4-7 5 5"/>', cut: '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M8.5 8.5 20 20M8.5 15.5 20 4"/>',
    erase: '<path d="m7 21-4-4 10-10 6 6-8 8z"/><path d="M14 21h7"/>', fade: '<path d="M3 20 21 4"/><path d="M3 20h18"/>', folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  };
  const ico = (k) => `<svg viewBox="0 0 24 24">${IC[k] || IC.fx}</svg>`;
  const ROLE = {
    vocals: ["mic", "#e0567c"], melody: ["mic", "#e0567c"], lead: ["synth", "#e2b14a"], drums: ["drums", "#3c8ff0"], bass: ["bass", "#8b6cf0"], guitar: ["guitar", "#3fbf6f"],
    piano: ["piano", "#e2b14a"], strings: ["strings", "#2fc0c8"], winds: ["flute", "#e07a4f"], middle_eastern: ["oud", "#d9894a"], synth: ["synth", "#a46cf0"], other: ["fx", "#a0a8b8"],
  };
  const tColor = (t) => t.color || (ROLE[t.role] || ROLE.other)[1];
  const tIcon = (t) => (t.kind === "bus" ? "fx" : t.kind === "audio" && t.role !== "guitar" && t.role !== "bass" ? "mic" : (ROLE[t.role] || ROLE.other)[0]);
  const P = () => (S && S.project) || null;
  const tracks = () => (P() ? P().tracks : []);
  const selTrack = () => tracks().find((t) => t.id === ui.sel) || null;
  const bpm = () => (P() ? P().bpm : 120);
  const meter = () => (P() && P().meter) || [4, 4];
  const unitSec = () => (60 / bpm()) * (4 / meter()[1]); // the meter's counting unit (quarter in x/4, eighth in x/8)
  const secPerBar = () => meter()[0] * unitSec();
  const pxs = () => (ui.zoom * bpm()) / 60; // px per second
  const X = (sec) => sec * pxs();
  const s2t = (sec) => Math.round((sec * bpm() / 60) * TPB);
  const fmtT = (s) => { const m = Math.floor(s / 60), r = s - m * 60; return String(m).padStart(2, "0") + ":" + r.toFixed(1).padStart(4, "0"); };
  const gainToSlider = (db) => Math.round(db <= -60 ? 0 : Math.pow((db + 60) / 72, 2) * 1000);
  const sliderToGain = (v) => Math.round((Math.sqrt(v / 1000) * 72 - 60) * 10) / 10;
  const snapQ = () => (ui.snap === 4 ? (secPerBar() * bpm()) / 60 : ui.snap); // "bar" follows the meter; others are quarter-note fractions
  const snapSec = (sec) => { if (!ui.snap) return Math.max(0, sec); const q = (snapQ() * 60) / bpm(); return Math.max(0, Math.round(sec / q) * q); };
  function songSecs() {
    let end = 0;
    for (const t of tracks()) { for (const c of t.clips || []) end = Math.max(end, c.start + c.length); for (const c of t.midi || []) end = Math.max(end, c.start + Math.max(c.length, secPerBar())); }
    return Math.max(end + 4 * secPerBar(), 32 * secPerBar());
  }

  // ---------------- menus ----------------
  const MENUS = [
    ["קובץ", [["פרויקט חדש…", "Alt+N", newProject], ["הפרויקטים שלי…", "Ctrl+O", openProjects], ["שינוי שם…", "", renameProject], ["שכפול הפרויקט", "", () => act("duplicate_project", {}, "נוצר עותק של הפרויקט")], "-",
      ["שמירת גרסה…", "Ctrl+S", saveVersion], ["גרסאות שמורות…", "", openVersions], "-", ["ייבוא קובץ שמע…", "Ctrl+I", () => $("#f-audio").click()], ["ייבוא קובץ תווים (MIDI)…", "", () => $("#f-midi").click()], "-",
      ["ייצוא השיר לקובץ WAV", "Ctrl+E", () => exportSong(false)], ["ייצוא כל ערוץ לקובץ נפרד", "", () => exportSong(true)], "-", ["הגדרות שמע…", "", openSettings]]],
    ["עריכה", [["ביטול פעולה", "Ctrl+Z", () => undo()], ["ביצוע שוב", "Ctrl+Shift+Z", () => redo()], ["היסטוריית פעולות…", "Alt+Z", () => openHistory()], "-", ["העתקת הקטע", "Ctrl+C", () => copyRegion()], ["הדבקה בסמן", "Ctrl+V", () => pasteRegion()], ["שכפול הקטע הנבחר", "Ctrl+D", dupRegion], ["חיתוך במיקום הסמן", "T", splitAtPlayhead], ["מחיקת הקטע הנבחר", "Delete", delRegion], "-",
      ["יישור הקטע הנבחר לקצב", "Q", () => quantizeSel()], ["העלאת כל התווים בחצי טון (לא הקלטות)", "", () => act("transpose", { semitones: 1 }, "התווים והסולם עלו בחצי טון; הקלטות לא משתנות")], ["הורדת כל התווים בחצי טון (לא הקלטות)", "", () => act("transpose", { semitones: -1 }, "התווים והסולם ירדו בחצי טון; הקלטות לא משתנות")]]],
    ["ערוץ", [["ערוץ חדש…", "Ctrl+Alt+N", () => openNT()], ["שכפול הערוץ", "", () => trackCmd("dup")], ["מחיקת הערוץ", "", () => trackCmd("del")], "-", ["העברה למעלה", "", () => trackCmd("up")], ["העברה למטה", "", () => trackCmd("down")], ["נעילה / ביטול נעילה", "", () => trackCmd("lock")]]],
    ["הקלטה", [["הקלטה / עצירה", "R", () => rec()], ["ספירה לפני הקלטה", "", () => $("#b-count").click()], ["מטרונום", "K", () => $("#b-metro").click()], ["לולאה", "C", () => $("#b-cyc").click()], "-", ["הגדרות מיקרופון…", "", openSettings]]],
    ["מיקס", [["מיקסר", "X", () => btab("mixer")], ["אפקטים לערוץ", "", () => btab("fx")], ["מאסטר וייצוא", "", () => btab("master")]]],
    ["תצוגה", [["מפקח", "I", () => toggleView("insp")], ["ספרייה", "Y", () => toggleView("lib")], ["חלון תחתון", "", () => toggleView("bot")], ["עורך תווים", "E", () => btab("piano")], ["עורך שמע", "", () => btab("audio")], ["מכונת תופים", "", () => btab("steps")], "-", ["הגדלה", "+", () => zoomBy(1.25)], ["הקטנה", "-", () => zoomBy(0.8)]]],
    ["עזרה", [["קיצורי מקלדת", "?", () => $("#hp").classList.add("on")]]],
  ];
  function buildMenus() {
    $("#menus").innerHTML = MENUS.map(([n, items], i) => `<div class="menu" data-i="${i}"><button>${n}</button><div class="drop">${items.map((it, j) => (it === "-" ? "<hr>" : `<button data-j="${j}"><span>${it[0]}</span><kbd>${it[1]}</kbd></button>`)).join("")}</div></div>`).join("");
    $$(".menu").forEach((m) => {
      $("button", m).onclick = (e) => { const o = m.classList.contains("open"); $$(".menu").forEach((x) => x.classList.remove("open")); if (!o) m.classList.add("open"); e.stopPropagation(); };
      $$(".drop button", m).forEach((b) => (b.onclick = () => { m.classList.remove("open"); MENUS[+m.dataset.i][1][+b.dataset.j][2](); }));
    });
    document.addEventListener("click", () => $$(".menu").forEach((x) => x.classList.remove("open")));
  }

  // ---------------- projects ----------------
  async function newProject() { const n = await ask("פרויקט חדש", "איך לקרוא לשיר?", "שיר חדש"); if (n === null) return; stopAll(); await act("new_project", { name: n || "שיר חדש" }, "נוצר פרויקט חדש"); ui.sel = null; ui.region = null; }
  async function renameProject() { if (!P()) return; const n = await ask("שם הפרויקט", "", P().name); if (n) act("rename_project", { name: n }, "השם עודכן"); }
  async function saveVersion() { if (!P()) return; const n = await ask("שמירת גרסה", "שם לגרסה (אפשר לחזור אליה מתי שרוצים)", ""); if (n === null) return; const r = await call("snapshot_create", { name: n }); if (r.status === "done") toast(r.message); }
  function openProjects() {
    const list = LAB.call("list_projects", {}) || [];
    $("#pj-list").innerHTML = list.map((p) => `<button class="pj ${P() && P().id === p.id ? "cur" : ""}" data-id="${p.id}"><b>${esc(p.title)}</b><span class="lbl">${p.tracks} ערוצים</span><span class="lbl ltr">${esc((p.updated_at || "").slice(0, 10))}</span></button>`).join("") || '<div class="empty-hint">אין עדיין פרויקטים</div>';
    $$("#pj-list .pj").forEach((b) => (b.onclick = async () => { stopAll(); const r = await call("open_project", { id: b.dataset.id }); if (r.status === "done") { $("#pj").classList.remove("on"); ui.sel = null; ui.region = null; refresh(); toast(r.message); } }));
    $("#pj").classList.add("on");
  }
  $("#pj-new").onclick = () => { $("#pj").classList.remove("on"); newProject(); };
  async function exportSong(stems) {
    toast(stems ? "מכין קובץ לכל ערוץ…" : "מכין את קובץ השיר…");
    const r = await call("export_audio", stems ? { stems: "true" } : {});
    if (r.status === "done") toast(r.message);
  }
  async function openSettings() {
    const ins = await Promise.resolve(LAB.call("list_inputs", {})) || [];
    $("#st-in").innerHTML = '<option value="">ברירת המחדל של המחשב</option>' + ins.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
    $("#st-in").onchange = () => call("set_input", { id: $("#st-in").value });
    $("#st-mon").checked = !!(S && S.monitor);
    $("#st-mon").onchange = async () => { const r = await call("set_monitor", { on: String($("#st-mon").checked) }); toast(r.message); };
    $("#st").classList.add("on");
  }
  $("#f-audio").onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = ""; if (!f) return;
    const t = selTrack(), at = snapBar(LAB.call("pulse", {}).seconds);
    const r = await Promise.resolve(LAB.importAudio(f, t && t.kind === "audio" ? { track_id: t.id, at_sec: at } : { new_track: true, at_sec: at }));
    S = r.state || S; if (r.track_id) ui.sel = r.track_id; refresh(); toast(r.message, r.status === "error");
  };
  $("#f-midi").onchange = async (e) => { const f = e.target.files[0]; e.target.value = ""; if (!f) return; const r = await Promise.resolve(LAB.importMidi(f, "own")); S = r.state || S; refresh(); toast(r.message, r.status === "error"); };

  // ---------------- transport ----------------
  let pulseTimer = 0;
  async function play() { if (S && S.recording && S.recording.active) return stopAll(); const pl = ui.pulse && ui.pulse.playing; await call("transport", { op: pl ? "stop" : "play" }); startPulse(); }
  async function stopAll() {
    const p = LAB.call("pulse", {});
    if (p.recording.active) { const r = await call("record_stop", {}); refresh(); if (r.status === "done") toast(r.message); else if (r.status === "error") { /* shown */ } return; }
    if (p.playing) { await call("transport", { op: "stop" }); }
    else { await call("seek", { seconds: ui.cycle ? ui.cycle.a : 0 }); }
    startPulse();
  }
  async function rec() {
    const p = LAB.call("pulse", {});
    if (p.recording.active) return stopAll();
    if (!P()) await call("new_project", { name: "שיר חדש" });
    let armed = tracks().filter((t) => t.armed);
    if (armed.length > 1 && armed.some((t) => t.id === ui.sel)) armed = [selTrack(), ...armed.filter((t) => t.id !== ui.sel)];
    if (!armed.length) {
      const st = selTrack();
      if (st && !st.locked && st.kind !== "bus") { await call("set_track_arm", { track_id: st.id, on: "true" }); refresh(); armed = [selTrack()]; }
      else { flashArm(); toast('קודם בוחרים איפה להקליט: לוחצים R על ערוץ, או "ערוץ חדש"', true); return; }
    }
    if (armed.length > 1) toast(`בשלב זה מקליטים לערוץ אחד: "${armed[0].name}"`);
    if (armed[0].kind === "midi" && !ui.keys && !midiIn) { ui.keys = true; setKeysUi(); toast("נגינה במקלדת המחשב הופעלה: A עד K הם קלידי הפסנתר, Z/X מחליפים אוקטבה"); }
    const args = { track_id: armed[0].id, count_in: ui.count ? 4 : 0 };
    if (ui.punch) { await call("seek", { seconds: Math.max(0, ui.punch.a - secPerBar()) }); args.punch_start = ui.punch.a; args.punch_end = ui.punch.b; } // one bar of pre-roll
    else if (S.transport.loop) await call("seek", { seconds: S.transport.loop.start });
    const r = await call("record_start", args);
    if (r.status === "done") { refresh(); toast(r.message); }
    startPulse();
  }
  function flashArm() { $$(".th .pill.r").forEach((b) => b.animate([{ boxShadow: "0 0 0 0 rgba(239,75,79,.9)" }, { boxShadow: "0 0 0 9px rgba(239,75,79,0)" }], { duration: 900, iterations: 2 })); }
  $("#b-play").onclick = play; $("#b-stop").onclick = stopAll; $("#b-rec").onclick = rec;
  $("#b-start").onclick = () => call("seek", { seconds: 0 }).then(startPulse);
  $("#b-rew").onclick = () => { const s = LAB.call("pulse", {}).seconds; call("seek", { seconds: Math.max(0, snapBar(s) - secPerBar()) }).then(startPulse); };
  $("#b-ff").onclick = () => { const s = LAB.call("pulse", {}).seconds; call("seek", { seconds: snapBar(s) + secPerBar() }).then(startPulse); };
  const snapBar = (s) => Math.round(s / secPerBar()) * secPerBar();
  $("#b-cyc").onclick = async () => {
    if (ui.cycle && S.transport.loop) { await call("set_loop", {}); }
    else { const a = ui.cycle ? ui.cycle.a : 0, b = ui.cycle ? ui.cycle.b : 4 * secPerBar(); ui.cycle = { a, b }; await call("set_loop", { start: a, end: b }); }
    refresh();
  };
  $("#b-punch").onclick = async () => {
    if (ui.punch) { ui.punch = null; saveUi(); renderRuler(); setTransportUi(); return toast("פאנץ' כבוי — ההקלטה תתחיל ותיגמר בכפתורים"); }
    if (!ui.cycle) return toast("קודם מסמנים קטע: גוררים על סרגל הזמן מהתיבה שבה מתחילים לתיבה שבה מסיימים", true);
    ui.punch = { ...ui.cycle }; saveUi(); if (S.transport.loop) await call("set_loop", {}); refresh();
    toast(`פאנץ': תוקלט רק הקטע בתיבות ${Math.round(ui.punch.a / secPerBar()) + 1}–${Math.round(ui.punch.b / secPerBar())} (הפס האדום). ההקלטה מתחילה תיבה לפני`);
  };
  $("#b-metro").onclick = async () => { const on = !(S && S.transport.click); const r = await call("set_metronome", { on: String(on) }); refresh(); toast(r.message); };
  $("#b-count").onclick = () => { ui.count = !ui.count; saveUi(); setTransportUi(); toast(ui.count ? "ספירה 1-2-3-4 לפני כל הקלטה" : "ההקלטה מתחילה מיד, בלי ספירה"); };
  $("#lcd-bpm").onchange = async (e) => { const v = Math.round(+e.target.value); if (!(v >= 30 && v <= 300)) { toast("קצב בין 30 ל-300", true); e.target.value = bpm(); return; } await act("set_tempo", { bpm: v }, `קצב: ${v}`); };
  const KEYS = [["דו", 0], ["דו#", 1], ["רה", 2], ["מי♭", 3], ["מי", 4], ["פה", 5], ["פה#", 6], ["סול", 7], ["לה♭", 8], ["לה", 9], ["סי♭", 10], ["סי", 11]];
  $("#lcd-key").innerHTML = KEYS.flatMap(([n, t]) => [`<option value="${t}:major">${n} מז'ור</option>`, `<option value="${t}:minor">${n} מינור</option>`]).join("");
  const METERS = [[4, 4], [3, 4], [2, 4], [5, 4], [6, 8], [7, 8], [9, 8], [12, 8]];
  $("#lcd-sig").innerHTML = METERS.map(([n, d]) => `<option value="${n}/${d}">${n}/${d}</option>`).join("");
  $("#lcd-sig").removeAttribute("title");
  $("#lcd-sig").onchange = async (e) => { const [n, d] = e.target.value.split("/"); await act("set_meter", { numerator: n, denominator: d }, `משקל ${e.target.value}`); };
  $("#lcd-key").onchange = async (e) => { const [t, m] = e.target.value.split(":"); await act("set_key", { tonic: t, mode: m }, "הסולם עודכן"); };
  function setTransportUi() {
    const tr = (S && S.transport) || {}, rc = (S && S.recording) || {};
    const pl = ui.pulse ? ui.pulse.playing : tr.playing, recA = ui.pulse ? ui.pulse.recording.active : rc.active, counting = ui.pulse && ui.pulse.recording.counting;
    $("#b-play").classList.toggle("on", !!pl && !recA); $("#b-rec").classList.toggle("on", !!recA);
    $("#b-cyc").classList.toggle("on", !!tr.loop); $("#b-punch").classList.toggle("on", !!ui.punch); $("#b-metro").classList.toggle("on", !!tr.click); $("#b-count").classList.toggle("on", ui.count);
    $("#lcd").classList.toggle("recording", !!recA); $("#arr").classList.toggle("recording", !!recA);
    $("#lcd-state").textContent = counting ? "1-2-3-4…" : recA ? "● מקליט" : pl ? "▶ מנגן" : "עצור";
    if (P()) { if (document.activeElement !== $("#lcd-bpm")) $("#lcd-bpm").value = Math.round(bpm()); $("#lcd-key").value = `${P().key_tonic}:${P().key_mode}`; $("#lcd-sig").value = meter().join("/"); }
    $("#undo").disabled = !(S && S.can_undo); $("#redo").disabled = !(S && S.can_redo);
    $("#undo").style.opacity = $("#undo").disabled ? 0.35 : 1; $("#redo").style.opacity = $("#redo").disabled ? 0.35 : 1;
  }
  // light polling for the playhead, meters and recording; full state only when the project changed
  function startPulse() { if (!pulseTimer) pulseTimer = requestAnimationFrame(pulseLoop); }
  let lastRev = -2, idleFrames = 0, wasRec = false;
  function pulseLoop() {
    pulseTimer = 0;
    const p = LAB.call("pulse", {}); ui.pulse = p;
    if (!ui.dragging && (p.revision !== lastRev || (wasRec && !p.recording.active))) { lastRev = p.revision; refresh(); } // never redraw under a drag
    wasRec = p.recording.active;
    movePH(p.seconds); meters(p); setTransportUi(); liveRegion(p);
    if (p.playing || p.recording.active) { idleFrames = 0; pulseTimer = requestAnimationFrame(pulseLoop); }
    else if (idleFrames++ < 3) pulseTimer = requestAnimationFrame(pulseLoop);
    else { $$("[data-meter]").forEach((m) => (m.style.height = "0")); $("#inmeter").style.width = "0"; }
  }
  function meters(p) {
    $$("[data-meter]").forEach((m) => { const id = m.dataset.meter; const v = id === "master" ? p.master : p.meters[id] || 0; m.style.height = Math.min(100, Math.round(Math.sqrt(v) * 100)) + "%"; });
    $("#inmeter").style.width = p.recording.active ? Math.min(100, Math.round(Math.sqrt(p.recording.input_peak || 0) * 100)) + "%" : "0";
  }
  function liveRegion(p) {
    let el = $("#live-region");
    if (!p.recording.active || p.recording.counting) {
      if (el) el.remove(); const ci = $("#countin");
      if (p.recording.active && p.recording.counting) { ui.countT0 = ui.countT0 || performance.now(); const n = meter()[0], k = Math.min(n, Math.floor((performance.now() - ui.countT0) / 1000 / unitSec()) + 1); ci.classList.add("on"); ci.textContent = String(k); $("#lcd-state").textContent = `ספירה ${k}/${n}`; }
      else { ci.classList.remove("on"); ui.countT0 = 0; }
      return;
    }
    ui.countT0 = 0;
    $("#countin").classList.remove("on");
    const lane = $(`.lane[data-id="${p.recording.track_id}"]`); if (!lane) return;
    if (!el) { el = document.createElement("div"); el.id = "live-region"; el.className = "region live"; el.innerHTML = '<span class="rn">● מקליט…</span>'; lane.appendChild(el); }
    el.style.left = X(p.recording.start) + "px"; el.style.width = Math.max(4, X(p.recording.seconds)) + "px";
  }
  function movePH(sec) {
    const ph = $("#ph"); if (!ph) return; ph.style.left = X(sec) + "px";
    const n = meter()[0], u = sec / unitSec(), bar = Math.floor(u / n) + 1, beat = Math.floor(u % n) + 1, six = Math.floor((u * 4) % 4) + 1;
    $("#lcd-bars").textContent = `${bar}.${beat}.${six}`; $("#lcd-time").textContent = fmtT(sec);
    const L = $("#lanes"), px = X(sec);
    if ((ui.pulse && ui.pulse.playing) && (px > L.scrollLeft + L.clientWidth - 40 || px < L.scrollLeft)) L.scrollLeft = Math.max(0, px - 60);
    syncScroll();
  }
  function syncScroll() { const L = $("#lanes"); $("#rulerwrap").scrollLeft = L.scrollLeft; $("#heads").scrollTop = L.scrollTop; }

  // ---------------- arrange: ruler, tracks, regions ----------------
  const TOOLS = [["ptr", "בחירה והזזה"], ["pen", "עיפרון – קטע תווים חדש בערוץ כלי"], ["cut", "מספריים – חיתוך קטע או הקלטה"], ["erase", "מחק – מחיקת קטע"], ["fade", "דעיכה – כניסה/יציאה רכה להקלטה"], ["auto", "אוטומציה – ציור עוצמה לאורך השיר"]];
  $("#toolgrp").innerHTML = TOOLS.map(([k, t]) => `<button class="tool" data-t="${k}" title="${t}">${ico(k)}</button>`).join("");
  $$(".tool").forEach((b) => (b.onclick = () => { const was = ui.tool; ui.tool = b.dataset.t; setTool(); if ((was === "auto") !== (ui.tool === "auto")) renderTracks(); toast(ui.tool === "auto" ? "אוטומציה: לחיצה על ערוץ מוסיפה נקודת עוצמה; גוררים נקודה; לחיצה כפולה מוחקת" : b.title); }));
  function setTool() { $$(".tool").forEach((x) => x.classList.toggle("on", x.dataset.t === ui.tool)); $$(".lane").forEach((l) => l.classList.toggle("pencil", ui.tool === "pen")); }
  function renderRuler() {
    const total = songSecs(), W = X(total), spb = secPerBar(), bars = Math.ceil(total / spb);
    let h = ""; const step = ui.zoom < 10 ? 4 : ui.zoom < 16 ? 2 : 1;
    for (let b = 0; b < bars; b++) { if (b % step === 0) h += `<span style="left:${X(b * spb)}px">${b + 1}</span>`; if (ui.zoom >= 26) for (let q = 1; q < meter()[0]; q++) h += `<span class="beat" style="left:${X(b * spb + q * unitSec())}px"></span>`; }
    $("#ruler").innerHTML = h + '<div class="cycle" id="cycle"></div><div class="punch" id="punch"></div>';
    const pu = $("#punch"); pu.classList.toggle("on", !!ui.punch); if (ui.punch) { pu.style.left = X(ui.punch.a) + "px"; pu.style.width = X(ui.punch.b - ui.punch.a) + "px"; }
    const hm = P() ? P().harmony : null;
    const mks = (P() && P().markers) || [];
    $("#g-mark").innerHTML = mks.length ? mks.map((m, i) => `<div class="marker user" data-mk="${m.id}" style="left:${X(m.start)}px;width:${Math.max(30, X((mks[i + 1] ? mks[i + 1].start : songSecs()) - m.start) - 3)}px" title="לחיצה: קפיצה · לחיצה כפולה: שינוי שם · קליק ימני: מחיקה">${esc(m.name)}</div>`).join("")
      : hm && hm.sections ? hm.sections.map(([n, a, d]) => `<div class="marker" style="left:${X(a)}px;width:${Math.max(10, X(d) - 3)}px;opacity:.6">${esc(n)}</div>`).join("") : "";
    bindMarkers();
    $("#g-chord").innerHTML = hm && hm.chords ? hm.chords.map(([a, d, n]) => `<div class="chord" style="left:${X(a)}px;max-width:${X(d)}px">${esc(n)}</div>`).join("") : "";
    [$("#ruler"), $("#g-mark"), $("#g-chord")].forEach((e) => (e.style.width = W + "px"));
    const lp = S && S.transport.loop, c = $("#cycle");
    if (lp) { ui.cycle = { a: lp.start, b: lp.end }; }
    c.classList.toggle("on", !!lp); if (ui.cycle) { c.style.left = X(ui.cycle.a) + "px"; c.style.width = X(ui.cycle.b - ui.cycle.a) + "px"; }
  }
  (() => { // ruler: click = move the playhead (to the bar), drag = loop
    const R = $("#ruler");
    R.onpointerdown = (e) => {
      const rb = R.getBoundingClientRect(), toSec = (cx) => Math.max(0, snapBar((cx - rb.left) / pxs())), a = toSec(e.clientX); let dragged = false;
      R.setPointerCapture(e.pointerId);
      R.onpointermove = (ev) => { const b = toSec(ev.clientX); if (Math.abs(b - a) >= secPerBar() * 0.99) { dragged = true; ui.cycle = { a: Math.min(a, b), b: Math.max(a, b) }; const c = $("#cycle"); c.classList.add("on"); c.style.left = X(ui.cycle.a) + "px"; c.style.width = X(ui.cycle.b - ui.cycle.a) + "px"; } };
      R.onpointerup = async (ev) => { R.onpointermove = R.onpointerup = null; if (dragged) { await call("set_loop", { start: ui.cycle.a, end: ui.cycle.b }); refresh(); toast(`לולאה: תיבות ${Math.round(ui.cycle.a / secPerBar()) + 1}–${Math.round(ui.cycle.b / secPerBar())}`); } else { await call("seek", { seconds: toSec(ev.clientX) }); startPulse(); } };
    };
  })();
  function bindMarkers() {
    const row = $("#g-mark");
    row.ondblclick = async (e) => {
      const mk = e.target.closest("[data-mk]"), m = mk && ((P() && P().markers) || []).find((x) => x.id === mk.dataset.mk);
      if (m) { const n = await ask("שם הסמן", "", m.name); if (n) act("set_marker", { id: m.id, tick: m.tick, name: n }); return; }
      const rb = row.getBoundingClientRect(), sec = snapBar((e.clientX - rb.left) / pxs());
      const n = await ask("סמן חדש", `בתיבה ${Math.round(sec / secPerBar()) + 1} — איך לקרוא לו?`, ["פתיחה", "בית", "פזמון", "גשר", "סיום"][Math.min(4, ((P().markers || []).length))]);
      if (n) act("set_marker", { tick: s2t(sec), name: n }, `נוסף סמן "${n}"`);
    };
    $$("[data-mk]", row).forEach((el) => {
      const m = ((P() && P().markers) || []).find((x) => x.id === el.dataset.mk);
      el.onclick = (e) => { e.stopPropagation(); call("seek", { seconds: m.start }).then(startPulse); };
      el.oncontextmenu = (e) => { e.preventDefault(); pop(e.clientX, e.clientY, '<button data-a="go">קפיצה לכאן</button><button data-a="ren">שינוי שם</button><button class="danger" data-a="del">מחיקת הסמן</button>', (pp) => {
        $$("button", pp).forEach((b) => (b.onclick = async () => { closePop(); if (b.dataset.a === "go") { await call("seek", { seconds: m.start }); startPulse(); } else if (b.dataset.a === "del") act("remove_marker", { id: m.id }, "הסמן נמחק"); else { const n = await ask("שם הסמן", "", m.name); if (n) act("set_marker", { id: m.id, tick: m.tick, name: n }); } }));
      }); };
    });
  }
  function grid() { const bw = X(secPerBar()), qw = X(unitSec()); return `repeating-linear-gradient(90deg,#223040 0 1px,transparent 1px ${bw}px)${qw >= 8 ? `,repeating-linear-gradient(90deg,#172230 0 1px,transparent 1px ${qw}px)` : ""}`; }
  function wavePath(peaks, w, h) {
    if (!peaks || !peaks.length) return "";
    const n = peaks.length, step = Math.max(1, Math.floor(n / Math.max(1, w / 2)));
    let up = "", dn = "";
    for (let i = 0; i < n; i += step) { let m = 0; for (let k = i; k < Math.min(n, i + step); k++) m = Math.max(m, Math.abs(peaks[k])); const x = ((i / n) * w).toFixed(1), a = Math.max(0.5, Math.min(1, m) * (h / 2)); up += `${up ? "L" : "M"}${x},${(h / 2 - a).toFixed(1)}`; dn = `L${x},${(h / 2 + a).toFixed(1)}` + dn; }
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${up}${dn}Z"/></svg>`;
  }
  function notesSvg(c) {
    if (!c.notes.length) return "";
    const lo = Math.min(...c.notes.map((n) => n[0])), hi = Math.max(...c.notes.map((n) => n[0])), L = Math.max(c.length, 0.001), span = Math.max(1, hi - lo);
    return `<svg viewBox="0 0 1000 100" preserveAspectRatio="none">${c.notes.map((n) => `<rect x="${((n[1] / L) * 1000).toFixed(1)}" y="${(88 - ((n[0] - lo) / span) * 80).toFixed(1)}" width="${Math.max(4, (n[2] / L) * 1000).toFixed(1)}" height="9" rx="2"/>`).join("")}</svg>`;
  }
  ui.collapsed = new Set();
  const isGroup = (t) => t.kind === "bus" && !!t.group;
  function viewTracks() { // a group's tracks right under it; collapsed groups hide them
    const all = tracks(), out = [];
    for (const t of all) {
      if (t.output && all.some((g) => g.id === t.output)) continue;
      out.push(t);
      if (t.kind === "bus" && !ui.collapsed.has(t.id)) for (const c of all) if (c.output === t.id) out.push(c);
    }
    return out;
  }
  function renderTracks() {
    const T = viewTracks(), W = X(songSecs()), anySolo = T.some((t) => t.solo);
    $("#heads").innerHTML = T.map((t) => `<div class="th ${t.id === ui.sel ? "pk" : ""} ${t.output ? "child" : ""}" data-id="${t.id}" style="--c:${tColor(t)}">
      ${isGroup(t) ? `<button class="fold" data-k="fold" title="פתיחה / קיפול">${ui.collapsed.has(t.id) ? "◂" : "▾"}</button>` : ""}<div class="tic">${ico(isGroup(t) ? "folder" : tIcon(t))}</div>
      <div class="tname" title="לחיצה כפולה לשינוי שם">${esc(t.name)}${t.locked ? ' <span class="lk">🔒</span>' : ""}<small>${t.kind === "audio" ? "שמע" : t.kind === "bus" ? (isGroup(t) ? "תיקייה / קבוצה" : "ערוץ אפקט") : esc(t.instrument_name || "כלי")}</small></div>
      <button class="pill m ${t.mute ? "on" : ""}" data-k="m" title="השתקה (M)">M</button>
      <button class="pill s ${t.solo ? "on" : ""}" data-k="s" title="סולו (S)">S</button>
      ${t.kind === "bus" ? '<span style="width:22px"></span>' : `<button class="pill r ${t.armed ? "on" : ""}" data-k="r" title="מוכן להקלטה">R</button>`}
      <button class="more" data-k="menu" title="אפשרויות">${ico("more")}</button></div>`).join("");
    const lanes = T.map((t) => `<div class="lane ${t.id === ui.sel ? "pk" : ""}" data-id="${t.id}" style="--c:${tColor(t)};width:${W}px;background-image:${grid()};opacity:${(anySolo && !t.solo) || t.mute ? 0.45 : 1}">
      ${(t.clips || []).filter((c) => !c.muted).map((c) => { const w = Math.max(6, X(c.length) - 1), sel = ui.region && ui.region.id === c.id; const fi = c.edit && c.edit.fade_in_ms ? X(c.edit.fade_in_ms / 1000) : 0, fo = c.edit && c.edit.fade_out_ms ? X(c.edit.fade_out_ms / 1000) : 0;
        return `<div class="region ${sel ? "selr" : ""}" data-kind="audio" data-id="${c.id}" data-tid="${t.id}" style="left:${X(c.start)}px;width:${w}px"><span class="rn">${esc(t.name)}</span>${c.takes > 1 ? `<button class="takes" data-takes="${c.id}" title="בחירת טייק">טייקים ${c.takes} ▾</button>` : ""}${wavePath(c.peaks, Math.round(w), 40)}${fi ? `<div class="fin" style="left:0;width:${fi}px"></div>` : ""}${fo ? `<div class="fout" style="right:0;width:${fo}px"></div>` : ""}<div class="trimh l"></div><div class="trimh r"></div></div>`; }).join("")}
      ${(t.midi || []).map((c) => { const len = c.notes.length ? Math.max(c.length, secPerBar()) : 4 * secPerBar(); const sel = ui.region && ui.region.id === c.id;
        return `<div class="region midi ${sel ? "selr" : ""}" data-kind="midi" data-id="${c.id}" data-tid="${t.id}" style="left:${X(c.start)}px;width:${Math.max(6, X(len) - 1)}px"><span class="rn">${esc(t.name)}${c.notes.length ? "" : " · ריק"}</span>${notesSvg({ ...c, length: len })}</div>`; }).join("")}
      ${ui.tool === "auto" || (t.automation || []).length ? autoLayer(t, W) : ""}
      </div>`).join("");
    $("#lanes").innerHTML = (lanes || `<div class="empty-hint" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center">הפרויקט ריק. לחץ "ערוץ חדש" כדי להתחיל — שירה, גיטרה, תופים, פסנתר או כל כלי אחר.</div>`) + '<div class="playhead" id="ph"></div>';
    $$(".th").forEach((h) => {
      const t = T.find((x) => x.id === h.dataset.id);
      h.onclick = (e) => {
        const k = e.target.closest("[data-k]")?.dataset.k;
        if (k === "m") return act("set_mute", { track_id: t.id, on: String(!t.mute) });
        if (k === "s") return act("set_solo", { track_id: t.id, on: String(!t.solo) });
        if (k === "r") { if (t.locked && !t.armed) return toast(`הערוץ ${t.name} נעול — קודם מבטלים נעילה`, true); return act("set_track_arm", { track_id: t.id, on: String(!t.armed) }, !t.armed ? `"${t.name}" מוכן להקלטה. לחץ על כפתור ההקלטה למעלה (או R)` : null); }
        if (k === "menu") return trackMenu(t, e.clientX, e.clientY);
        if (k === "fold") { ui.collapsed.has(t.id) ? ui.collapsed.delete(t.id) : ui.collapsed.add(t.id); return renderTracks(); }
        select(t.id);
      };
      h.oncontextmenu = (e) => { e.preventDefault(); trackMenu(t, e.clientX, e.clientY); };
      $(".tname", h).ondblclick = () => renameInline(t, h);
    });
    $$(".lane").forEach((l) => (l.onpointerdown = (e) => { if (e.target.closest(".region") || e.target.closest(".autodot")) return; laneClick(l.dataset.id, e, l); }));
    $$(".region").forEach(bindRegion);
    bindAuto();
    setTool(); movePH(ui.pulse ? ui.pulse.seconds : (S ? S.transport.seconds : 0));
  }
  // volume automation: a line over the lane; y = the fader's own scale (top = +12 dB, bottom = silence)
  const autoY = (db) => 1 - gainToSlider(db) / 1000, autoDb = (y) => Math.max(-60, Math.min(12, sliderToGain((1 - Math.max(0, Math.min(1, y))) * 1000)));
  function autoLayer(t, W) {
    const pts = (t.automation || []).map(([k, d]) => [k / TPB * 60 / bpm(), d]), live = ui.tool === "auto";
    const line = pts.length ? [[0, pts[0][1]], ...pts, [songSecs(), pts[pts.length - 1][1]]] : [[0, t.gain_db], [songSecs(), t.gain_db]];
    return `<svg class="autoline ${live ? "live" : ""}" viewBox="0 0 ${W} 100" preserveAspectRatio="none"><polyline points="${line.map(([x, d]) => `${X(x).toFixed(1)},${(autoY(d) * 100).toFixed(1)}`).join(" ")}" ${pts.length ? "" : 'stroke-dasharray="4 4"'}/></svg>
      ${live ? pts.map(([x, d], i) => `<div class="autodot" data-ai="${i}" data-tid="${t.id}" style="left:${X(x)}px;top:${autoY(d) * 100}%" title="${d.toFixed(1)} dB · גרירה להזזה · לחיצה כפולה למחיקה"></div>`).join("") : ""}`;
  }
  const savePts = (t, pts, msg) => act("set_automation", { track_id: t.id, points: pts.slice().sort((a, b) => a[0] - b[0]).map(([k, d]) => `${Math.round(k)}:${Math.round(d * 10) / 10}`).join(";") }, msg);
  function bindAuto() {
    $$(".autodot").forEach((d) => {
      const t = tracks().find((x) => x.id === d.dataset.tid), i = +d.dataset.ai, lane = d.parentElement;
      d.ondblclick = (e) => { e.stopPropagation(); const l = t.automation.slice(); l.splice(i, 1); savePts(t, l, "הנקודה נמחקה"); };
      d.onpointerdown = (e) => {
        e.stopPropagation(); d.setPointerCapture(e.pointerId); const lb = lane.getBoundingClientRect(); let moved = false;
        d.onpointermove = (ev) => { moved = true; d.style.left = Math.max(0, ev.clientX - lb.left) + "px"; d.style.top = Math.max(0, Math.min(100, ((ev.clientY - lb.top) / lb.height) * 100)) + "%"; };
        d.onpointerup = (ev) => { d.onpointermove = d.onpointerup = null; if (!moved) return; const l = t.automation.slice(); l[i] = [s2t(snapSec((ev.clientX - lb.left) / pxs())), autoDb((ev.clientY - lb.top) / lb.height)]; savePts(t, l); };
      };
    });
  }
  async function laneClick(tid, e, lane) {
    // read the click position before anything is redrawn
    const lb = lane.getBoundingClientRect(), fx = (e.clientX - lb.left) / pxs(), fy = (e.clientY - lb.top) / lb.height;
    const t = tracks().find((x) => x.id === tid), sec = snapBar(Math.max(0, fx));
    select(tid, true);
    if (ui.tool === "auto") { const s2 = snapSec(fx), db = autoDb(fy); return savePts(t, [...(t.automation || []), [s2t(s2), db]], `נקודת עוצמה: ${db.toFixed(1)} dB`); }
    if (ui.tool === "pen") {
      if (t.kind !== "midi") return toast("בערוץ שמע מקליטים. קטע תווים מציירים בערוץ של כלי וירטואלי", true);
      const r = await call("new_midi_clip", { track_id: t.id, start_tick: s2t(sec) });
      if (r.status === "done") { ui.region = { id: r.clip_id, kind: "midi", tid: t.id }; refresh(); btab(t.role === "drums" ? "steps" : "piano"); toast("נוצר קטע חדש. מציירים בו תווים בחלון התחתון"); }
    } else { ui.region = null; renderAll(); }
  }
  let lastClick = { id: null, t: 0 };
  function bindRegion(el) {
    const id = el.dataset.id, kind = el.dataset.kind, tid = el.dataset.tid;
    const openEditor = () => { ui.region = { id, kind, tid }; select(tid, true); const t = selTrack(); btab(kind === "audio" ? "audio" : t && t.role === "drums" ? "steps" : "piano"); };
    el.oncontextmenu = (e) => { e.preventDefault(); ui.region = { id, kind, tid }; select(tid, true); regionMenu(e.clientX, e.clientY); };
    el.onpointerdown = async (e) => {
      if (e.button !== 0) return;
      if (e.target.closest("[data-takes]")) { e.stopPropagation(); return takesMenu(tid, id, e.clientX, e.clientY); }
      e.stopPropagation(); ui.region = { id, kind, tid };
      const t = tracks().find((x) => x.id === tid);
      if (ui.tool === "auto") { ui.region = null; return laneClick(tid, e, el.parentElement); } // automation is drawn over regions too
      if (ui.tool === "erase") return delRegion();
      if (ui.tool === "cut") { const lb = el.parentElement.getBoundingClientRect(); return splitAt(snapSec((e.clientX - lb.left) / pxs())); }
      if (ui.tool === "fade" && kind === "audio") { const r = el.getBoundingClientRect(), left = e.clientX - r.left < r.width / 2; return act("set_clip_edit", { clip_id: id, [left ? "fade_in_ms" : "fade_out_ms"]: 400 }, left ? "כניסה רכה" : "יציאה רכה"); }
      if (t && t.locked) { select(tid, true); return toast(`הערוץ ${t.name} נעול`, true); }
      const trim = kind === "audio" ? e.target.closest(".trimh") : null, x0 = e.clientX, left0 = parseFloat(el.style.left), w0 = parseFloat(el.style.width);
      const c = kind === "audio" ? (t.clips || []).find((x) => x.id === id) : (t.midi || []).find((x) => x.id === id);
      let moved = false; el.setPointerCapture(e.pointerId); ui.dragging = true;
      const y0 = e.clientY;
      el.onpointermove = (ev) => {
        const dx = ev.clientX - x0, dy = ev.clientY - y0; if (Math.abs(dx) > 3 || Math.abs(dy) > 8) moved = true; if (!moved) return;
        if (trim && trim.classList.contains("r")) el.style.width = Math.max(8, w0 + dx) + "px";
        else if (trim) { const d = Math.min(w0 - 8, Math.max(-left0, dx)); el.style.left = left0 + d + "px"; el.style.width = w0 - d + "px"; }
        else { el.style.left = X(snapSec((left0 + dx) / pxs())) + "px"; el.style.transform = Math.abs(dy) > 8 ? `translateY(${dy}px)` : ""; el.style.zIndex = 6; }
      };
      el.onpointercancel = () => { ui.dragging = false; };
      el.onpointerup = async (ev) => {
        el.onpointermove = el.onpointerup = null; ui.dragging = false;
        if (!moved) { // a second click on the same region opens its editor (the region is redrawn between clicks)
          const now = Date.now(), dbl = lastClick.id === id && now - lastClick.t < 400; lastClick = { id, t: now };
          if (dbl) return openEditor();
          select(tid, true); return;
        }
        const dx = ev.clientX - x0;
        if (trim) { // trims snap to the grid like moves do
          const sr = c.sr || 48000, e0 = c.edit || {};
          if (trim.classList.contains("r")) { const end = snapSec(c.start + c.length + dx / pxs()), len = Math.max(0.05, end - c.start); await act("set_clip_edit", { clip_id: id, trim_len: Math.round(Math.min(c.total - (e0.trim_start || 0) / sr, len) * sr) }); }
          else { const ns = snapSec(c.start + dx / pxs()), d = Math.max(-((e0.trim_start || 0) / sr), Math.min(c.length - 0.05, ns - c.start)); await act("set_clip_edit", { clip_id: id, start_tick: s2t(c.start + d), trim_start: Math.round((e0.trim_start || 0) + d * sr), trim_len: Math.round((c.length - d) * sr) }); }
        } else {
          const ns = snapSec((left0 + dx) / pxs());
          el.style.visibility = "hidden"; const under = document.elementFromPoint(ev.clientX, ev.clientY); el.style.visibility = "";
          const lane = under && under.closest(".lane");
          if (lane && lane.dataset.id !== tid) { const r = await act("move_region", { clip_id: id, track_id: lane.dataset.id, start_tick: s2t(ns) }); if (r.status === "done") { ui.region = { id: r.clip_id, kind, tid: lane.dataset.id }; ui.sel = lane.dataset.id; renderAll(); } else renderTracks(); return; }
          await act(kind === "audio" ? "set_clip_start" : "set_midi_clip_start", { clip_id: id, start_tick: s2t(ns) });
        }
      };
    };
  }
  function takesMenu(tid, id, x, y) {
    const t = tracks().find((q) => q.id === tid), c = (t.clips || []).find((q) => q.id === id);
    const group = (t.clips || []).filter((q) => q.id === id || (q.start < c.start + c.length - 0.05 && q.start + q.length > c.start + 0.05));
    pop(x, y, group.map((q, i) => `<button data-id="${q.id}"><span>טייק ${i + 1}${q.muted ? "" : " ✓"}</span><span class="lbl ltr">${q.length.toFixed(1)}s</span></button>`).join("") + '<button data-play>ניגון מהטייק</button>', (pp) => {
      $$("[data-id]", pp).forEach((b) => (b.onclick = () => { closePop(); act("choose_take", { clip_id: b.dataset.id }, "הטייק נבחר"); }));
      $("[data-play]", pp).onclick = async () => { closePop(); await call("seek", { seconds: c.start }); await call("transport", { op: "play" }); startPulse(); };
    });
  }
  function regionMenu(x, y) {
    const r = ui.region; if (!r) return;
    pop(x, y, `<button data-a="dup">שכפול אחרי הקטע</button>${r.kind === "midi" ? '<button data-a="rep">חזרה 4 פעמים</button><button data-a="split">חיתוך במיקום הסמן</button><button data-a="edit">פתיחה בעורך</button>' : '<button data-a="split">חיתוך במיקום הסמן</button><button data-a="norm">התאמת עוצמה (נרמול)</button><button data-a="fin">כניסה רכה</button><button data-a="fout">יציאה רכה</button><button data-a="nofade">ביטול כניסה/יציאה רכה</button><button data-a="edit">פתיחה בעורך השמע</button>'}<button class="danger" data-a="del">מחיקה</button>`, (p) => {
      $$("button", p).forEach((b) => (b.onclick = () => { closePop(); const a = b.dataset.a;
        if (a === "dup") dupRegion(); else if (a === "rep") act("duplicate_midi_clip", { clip_id: r.id, times: 4 }, "הקטע חוזר 4 פעמים"); else if (a === "split") splitAtPlayhead(); else if (a === "del") delRegion(); else if (a === "norm") act("normalize_clip", { clip_id: r.id }, null).then((x) => x.status === "done" && toast(x.message));
        else if (a === "fin") act("set_clip_edit", { clip_id: r.id, fade_in_ms: 400 }, "כניסה רכה"); else if (a === "fout") act("set_clip_edit", { clip_id: r.id, fade_out_ms: 600 }, "יציאה רכה"); else if (a === "nofade") act("set_clip_edit", { clip_id: r.id, fade_in_ms: 0, fade_out_ms: 0 });
        else if (a === "edit") btab(r.kind === "audio" ? "audio" : selTrack() && selTrack().role === "drums" ? "steps" : "piano"); }));
    });
  }
  // clipboard: one region, pasted at the playhead on the selected track
  let clip = null;
  function copyRegion() {
    const r = ui.region; if (!r) return toast("קודם בוחרים קטע");
    const t = tracks().find((x) => x.id === r.tid), c = r.kind === "audio" ? (t.clips || []).find((x) => x.id === r.id) : (t.midi || []).find((x) => x.id === r.id);
    if (!c) return; clip = { kind: r.kind, id: c.id, notes: r.kind === "midi" ? c.notes.map((n) => `${n[0]},${n[5]},${n[6]},${n[3]}`).join(";") : null };
    toast("הקטע הועתק. בוחרים ערוץ ומקום, ו-Ctrl+V");
  }
  async function pasteRegion() {
    if (!clip) return toast("אין קטע מועתק");
    const t = selTrack(), at = snapSec(LAB.call("pulse", {}).seconds);
    if (!t) return toast("קודם בוחרים ערוץ", true);
    if (clip.kind === "audio") { if (t.kind !== "audio") return toast("הקלטה מדביקים בערוץ שמע", true); return act("paste_clip", { clip_id: clip.id, track_id: t.id, at_sec: at }, "הודבק"); }
    if (t.kind !== "midi") return toast("קטע תווים מדביקים בערוץ של כלי וירטואלי", true);
    return act("new_midi_clip", { track_id: t.id, start_tick: s2t(at), notes: clip.notes }, "הודבק");
  }
  function openHistory() {
    const list = LAB.call("history_list", {}) || [];
    pop(innerWidth / 2 - 150, 120, `<div class="lbl" style="padding:4px 10px">לחיצה על פעולה מחזירה את השיר למצב שלפניה</div>${list.slice(0, 30).map((h) => `<button data-n="${h.steps}"><span>${esc(h.label)}</span><span class="lbl">${h.steps === 1 ? "האחרונה" : `לפני ${h.steps} פעולות`}</span></button>`).join("") || '<div class="lbl" style="padding:8px 10px">עדיין אין פעולות</div>'}`, (pp) => {
      $$("[data-n]", pp).forEach((b) => (b.onclick = async () => { closePop(); for (let k = 0; k < +b.dataset.n; k++) await call("undo", {}, true); refresh(); toast(`בוטלו ${b.dataset.n} פעולות`); }));
    });
  }
  async function quantizeSel() {
    const r = ui.region, t = r && tracks().find((x) => x.id === r.tid), c = r && r.kind === "midi" && t && (t.midi || []).find((x) => x.id === r.id);
    if (!c) return toast("קודם בוחרים קטע תווים (לחיצה על הקטע)", true);
    const g = ui.snap ? Math.max(120, Math.round(snapQ() * TPB)) : 240;
    return act("set_midi_notes", { clip_id: c.id, notes: c.notes.map((n) => `${n[0]},${Math.round(n[5] / g) * g},${n[6]},${n[3]}`).join(";") }, "התווים של הקטע יושרו לקצב");
  }
  function openVersions() {
    const list = LAB.call("snapshots", {}) || [];
    pop(innerWidth / 2 - 170, 120, `<div class="lbl" style="padding:4px 10px">לחיצה על גרסה מחזירה את השיר אליה (אפשר לבטל עם Ctrl+Z)</div>${list.map((v) => `<button data-id="${v.id}"><span>${esc(v.name)}</span><span class="lbl ltr">${esc((v.created_at || "").slice(0, 16).replace("T", " "))}</span></button>`).join("") || '<div class="lbl" style="padding:8px 10px">עדיין אין גרסאות. "שמירת גרסה" בתפריט קובץ</div>'}`, (pp) => {
      $$("[data-id]", pp).forEach((b) => (b.onclick = async () => { closePop(); const r = await call("snapshot_restore", { id: b.dataset.id, approved: "true" }); refresh(); if (r.status === "done") toast(r.message); }));
    });
  }
  async function delRegion() { const r = ui.region; if (!r) return toast("קודם בוחרים קטע"); ui.region = null; await act(r.kind === "audio" ? "remove_clip" : "remove_midi_clip", { clip_id: r.id }, "הקטע נמחק · Ctrl+Z לביטול"); }
  async function dupRegion() {
    const r = ui.region; if (!r) return toast("קודם בוחרים קטע");
    if (r.kind === "midi") return act("duplicate_midi_clip", { clip_id: r.id, times: 1 }, "הקטע שוכפל");
    return act("duplicate_clip", { clip_id: r.id }, "ההקלטה שוכפלה אחריה");
  }
  function splitAtPlayhead() { splitAt(LAB.call("pulse", {}).seconds); }
  async function splitAt(sec) {
    const r = ui.region; if (!r) return toast("קודם בוחרים קטע");
    if (r.kind !== "midi") return act("split_clip", { clip_id: r.id, at_sec: sec }, "ההקלטה נחתכה לשניים");
    await act("split_midi_clip", { clip_id: r.id, at_tick: s2t(sec) }, "הקטע נחתך");
  }
  async function renameInline(t, h) {
    const box = $(".tname", h); box.innerHTML = `<input value="${esc(t.name)}">`; const i = $("input", box); i.focus(); i.select();
    const done = async (ok) => { i.onblur = null; if (ok && i.value.trim() && i.value.trim() !== t.name) await act("rename_track", { track_id: t.id, name: i.value.trim() }); else renderTracks(); };
    i.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); }; i.onblur = () => done(true);
  }
  const COLORS = ["#e0567c", "#3fbf6f", "#8b6cf0", "#3c8ff0", "#e2b14a", "#2fc0c8", "#e07a4f", "#a0a8b8"];
  function trackMenu(t, x, y) {
    select(t.id, true);
    pop(x, y, `<button data-a="rename">שינוי שם</button><div class="colors">${COLORS.map((c) => `<i style="background:${c}" data-c="${c}"></i>`).join("")}</div>
      <button data-a="dup">שכפול הערוץ</button><button data-a="up">העברה למעלה</button><button data-a="down">העברה למטה</button><button data-a="lock">${t.locked ? "ביטול נעילה" : "נעילה"}</button>
      ${t.kind === "midi" ? '<button data-a="clip">קטע תווים חדש בסמן</button>' : ""}
      ${t.kind !== "bus" ? tracks().filter((b) => b.kind === "bus" && b.group).map((b) => `<button data-out="${b.id}" ${t.output === b.id ? "disabled" : ""}>העברה אל "${esc(b.name)}"</button>`).join("") + (t.output ? '<button data-out="">החזרה ליציאה הראשית</button>' : "") : ""}
      <button class="danger" data-a="del">מחיקת הערוץ</button>`, (p) => {
      $$("[data-out]", p).forEach((b) => (b.onclick = () => { closePop(); act("set_track_output", { track_id: t.id, bus_id: b.dataset.out || "null" }, b.dataset.out ? "הערוץ הועבר לתיקייה" : "הערוץ חזר ליציאה הראשית"); }));
      $$("[data-c]", p).forEach((i) => (i.onclick = () => { closePop(); act("set_track_color", { track_id: t.id, color: i.dataset.c }); }));
      $$("button[data-a]", p).forEach((b) => (b.onclick = () => { closePop(); const a = b.dataset.a; if (a === "rename") renameInline(t, $(`.th[data-id="${t.id}"]`)); else if (a === "clip") newClipAtPlayhead(t); else trackCmd(a); }));
    });
  }
  async function newClipAtPlayhead(t) { const s = snapBar(LAB.call("pulse", {}).seconds); const r = await call("new_midi_clip", { track_id: t.id, start_tick: s2t(s) }); if (r.status === "done") { ui.region = { id: r.clip_id, kind: "midi", tid: t.id }; refresh(); btab(t.role === "drums" ? "steps" : "piano"); } }
  async function trackCmd(a) {
    const t = selTrack(); if (!t) return toast("קודם בוחרים ערוץ");
    const i = tracks().findIndex((x) => x.id === t.id);
    if (a === "dup") return act("duplicate_track", { track_id: t.id }, "הערוץ שוכפל");
    if (a === "up") return i > 0 && act("move_track", { track_id: t.id, to: i - 1 });
    if (a === "down") return i < tracks().length - 1 && act("move_track", { track_id: t.id, to: i + 1 });
    if (a === "lock") return act("set_track_lock", { track_id: t.id, on: String(!t.locked) }, t.locked ? "הנעילה בוטלה" : "הערוץ נעול");
    if (a === "del") {
      const has = (t.clips || []).length + (t.midi || []).length;
      if (has && !(await ask("מחיקת ערוץ", `למחוק את "${t.name}" עם כל מה שיש בו? אפשר לבטל אחר כך עם Ctrl+Z`, null))) return;
      const r = await act("remove_track", { track_id: t.id, approved: "true" }, `הערוץ "${t.name}" נמחק · Ctrl+Z לביטול`);
      if (r.status === "done") { ui.sel = tracks()[Math.max(0, i - 1)]?.id || null; renderAll(); }
    }
  }
  function select(id, keepRegion) { ui.sel = id; if (!keepRegion) ui.region = null; renderAll(); }
  $("#lanes").addEventListener("scroll", syncScroll);
  $("#lanes").addEventListener("wheel", (e) => {
    const L = $("#lanes");
    if (e.ctrlKey) { e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.15 : 0.87); return; }
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { L.scrollLeft += e.deltaX || e.deltaY; e.preventDefault(); } else { L.scrollTop += e.deltaY; e.preventDefault(); }
    syncScroll();
  }, { passive: false });
  $("#lanes").style.overflow = "auto"; // real scrollbars, like every DAW
  $("#heads").addEventListener("wheel", (e) => { $("#lanes").scrollTop += e.deltaY; syncScroll(); e.preventDefault(); }, { passive: false });
  function zoomBy(f) { ui.zoom = Math.max(4, Math.min(120, ui.zoom * f)); $("#zoom").value = ui.zoom; saveUi(); renderRuler(); renderTracks(); if (ui.bottom === "piano") renderPiano(); }
  $("#zoom").value = ui.zoom; $("#zoom").oninput = (e) => { ui.zoom = +e.target.value; saveUi(); renderRuler(); renderTracks(); };
  $("#snap").value = String(ui.snap); $("#snap").onchange = (e) => { ui.snap = +e.target.value; saveUi(); };
  $("#th").oninput = (e) => document.documentElement.style.setProperty("--th", e.target.value + "px");

  // ---------------- inspector ----------------
  // every instrument the studio offers: [id, Hebrew name, library categories (|), icon, track role]. Same ids as the
  // engine (lab.js INSTRUMENT_HE / domain Instrument); each plays a real recorded sample set, the synth only stands in
  const INSTRUMENTS = [
    ["piano", "פסנתר", "פסנתרים", "piano", "piano"], ["epiano", "פסנתר חשמלי", "קלידים", "piano", "piano"], ["organ", "עוגב", "קלידים", "piano", "piano"],
    ["accordion", "אקורדיון", "קלידים|ים תיכוני", "piano", "piano"],
    ["guitar", "גיטרה אקוסטית", "גיטרות", "guitar", "guitar"], ["nylon_guitar", "גיטרה קלאסית", "גיטרות|ים תיכוני", "guitar", "guitar"], ["electric_guitar", "גיטרה חשמלית", "גיטרות", "guitar", "guitar"],
    ["bass", "בס חשמלי", "בסים", "bass", "bass"], ["double_bass", "קונטרבס", "בסים|מיתרים", "bass", "bass"],
    ["strings", "כלי קשת (כינורות)", "מיתרים", "strings", "strings"], ["violin", "כינור סולו", "מיתרים|ים תיכוני", "strings", "strings"], ["cello", "צ'לו", "מיתרים", "strings", "strings"], ["harp", "נבל", "מיתרים", "strings", "strings"],
    ["flute", "חליל", "נשיפה", "flute", "winds"], ["clarinet", "קלרינט", "נשיפה|ים תיכוני", "flute", "winds"], ["oboe", "אבובה", "נשיפה", "flute", "winds"], ["sax", "סקסופון", "נשיפה", "flute", "winds"], ["harmonica", "מפוחית", "נשיפה", "flute", "winds"],
    ["trumpet", "חצוצרה", "נשיפה ממתכת", "flute", "winds"], ["trombone", "טרומבון", "נשיפה ממתכת", "flute", "winds"], ["horn", "קרן יער", "נשיפה ממתכת", "flute", "winds"], ["tuba", "טובה", "נשיפה ממתכת|בסים", "flute", "bass"],
    ["marimba", "מרימבה", "הקשה מלודית", "piano", "piano"], ["vibraphone", "ויברפון", "הקשה מלודית", "piano", "piano"], ["glockenspiel", "פעמונים (גלוקנשפיל)", "הקשה מלודית", "piano", "piano"],
    ["drums", "מערכת תופים", "תופים", "drums", "drums"], ["world_percussion", "דרבוקה, קחון ובונגו", "תופים|ים תיכוני", "drums", "drums"],
    ["soft_lead", "סינתיסייזר רך", "סינתיסייזרים", "synth", "synth"],
  ];
  const INSTR = INSTRUMENTS.map(([id, he, , , role]) => [id, he, role]);
  const fxOn = (t) => { const f = t.fx || {}; return { eq: !!(f.eq_low_db || f.eq_mid_db || f.eq_high_db), comp: f.comp > 0, reverb: f.reverb > 0 }; };
  function renderInsp() {
    const t = selTrack(), box = $("#insp");
    if (!t) { box.innerHTML = '<div class="ptitle">מפקח</div><div class="empty-hint">בחר ערוץ כדי לראות את ההגדרות שלו</div>'; return; }
    const on = fxOn(t), r = ui.region && ui.region.tid === t.id ? (ui.region.kind === "audio" ? (t.clips || []).find((c) => c.id === ui.region.id) : (t.midi || []).find((c) => c.id === ui.region.id)) : null;
    box.innerHTML = `<div class="ptitle">מפקח <span class="muted">הערוץ הנבחר</span></div><div class="insp-body">
      <div class="trk-id"><div class="ic" style="color:${tColor(t)}">${ico(tIcon(t))}</div><div><b>${esc(t.name)}</b><small>${t.kind === "audio" ? "ערוץ שמע (מיקרופון / הקלטה)" : t.kind === "bus" ? "ערוץ אפקט — ערוצים אחרים שולחים אליו" : "כלי וירטואלי"}</small></div></div>
      ${r ? `<div class="sect">קטע נבחר</div><div class="row"><span class="lbl">התחלה</span><span class="ltr">${fmtT(r.start)}</span></div><div class="row"><span class="lbl">אורך</span><span class="ltr">${r.length.toFixed(2)} s</span></div>${ui.region.kind === "midi" ? `<div class="row"><span class="lbl">תווים</span><span>${r.notes.length}</span></div>` : `<div class="row"><span class="lbl">כניסה / יציאה רכה</span><span class="ltr">${(r.edit || {}).fade_in_ms || 0} / ${(r.edit || {}).fade_out_ms || 0} ms</span></div>`}` : ""}
      ${t.kind === "bus" ? "" : '<div class="sect">צליל</div>'}
      ${t.kind === "bus" ? "" : t.kind === "midi" ? `<div class="row"><span class="lbl">כלי</span><select class="sel" id="in-inst">${[["auto", "אוטומטי לפי התפקיד"], ...INSTR.map((x) => [x[0], x[1]])].map(([k, n]) => `<option value="${k}" ${t.instrument === k ? "selected" : ""}>${n}</option>`).join("")}</select></div><div class="lbl">${t.sound === "recorded" ? "כלי מוקלט אמיתי" : "צליל מסונתז"} · ${esc(t.instrument_name || "")}</div>` : '<div class="row"><span class="lbl">כניסה</span><button class="slot" id="in-mic" style="width:auto">בחירת מיקרופון…</button></div>'}
      <div class="sect">אפקטים בערוץ (לפי הסדר)</div>
      ${(t.inserts || []).map((x, i) => `<button class="slot ${x.on ? "" : "off"}" data-ins="${i}"><i data-tog="${i}" title="הפעלה / כיבוי"></i>${esc(E.DEFS[x.kind].he)}</button>`).join("")}
      ${(t.inserts || []).length < E.MAX_INSERTS ? '<button class="slot empty" id="in-add">+ הוספת אפקט</button>' : ""}
      ${on.eq || on.comp ? `<div class="sect">הגדרות צליל מהעיבוד האוטומטי</div>
      <div class="row" style="gap:4px">${on.eq ? '<button class="slot" data-fx="eq" style="flex:1"><i></i>אקולייזר בסיסי</button>' : ""}${on.comp ? '<button class="slot" data-fx="comp" style="flex:1"><i></i>קומפרסור בסיסי</button>' : ""}<button class="slot empty" id="in-qreset" style="flex:none;width:auto">איפוס</button></div>` : ""}
      ${t.kind === "bus" ? "" : `<div class="sect">שליחות לחלל משותף</div>
      ${tracks().filter((b) => b.kind === "bus" && !b.group).map((b) => { const v = Math.round(((t.sends || []).find((x) => x[0] === b.id) || [0, 0])[1] * 100); return `<div class="row"><span class="lbl">${esc(b.name)}</span><span class="sendv"><input type="range" dir="ltr" min="0" max="100" value="${v}" data-send="${b.id}"><b class="ltr">${v}%</b></span></div>`; }).join("")}
      <div class="row"><span class="lbl">ריוורב (חדר)</span><input type="range" min="0" max="100" value="${Math.round((t.fx.reverb || 0) * 100)}" id="in-rev" style="width:110px;accent-color:var(--gold)"></div>
      <div class="row"><span class="lbl">דיליי (הד)</span><input type="range" min="0" max="100" value="${Math.round((t.delay_send || 0) * 100)}" id="in-del" style="width:110px;accent-color:var(--gold)"></div>`}
      <div class="sect">יציאה: יציאה ראשית</div>
      <div class="strip-mini">
        <div class="kwrap"><div class="knob" id="in-pan" style="--a:${t.pan * 135}deg" title="פאן — גרירה למעלה/למטה, לחיצה כפולה לאמצע"></div><span id="in-panv">${panLabel(t.pan)}</span></div>
        <input class="vfader" type="range" min="0" max="1000" value="${gainToSlider(t.gain_db)}" id="in-vol" title="עוצמה">
        <div class="meter"><i data-meter="${t.id}"></i></div>
      </div>
      <div class="row" style="justify-content:center"><span class="ltr lbl" id="in-db">${t.gain_db.toFixed(1)} dB</span></div>
      <div class="msri" style="justify-content:center"><button class="pill m ${t.mute ? "on" : ""}" data-k="m">M</button><button class="pill s ${t.solo ? "on" : ""}" data-k="s">S</button>${t.kind === "bus" ? "" : `<button class="pill r ${t.armed ? "on" : ""}" data-k="r">R</button>`}${t.kind === "audio" ? `<button class="pill i ${S.monitor ? "on" : ""}" data-k="i" title="לשמוע את עצמי בזמן הקלטה (אוזניות)">I</button>` : ""}</div>
    </div>`;
    $$(".msri .pill", box).forEach((b) => (b.onclick = () => { const k = b.dataset.k; if (k === "m") act("set_mute", { track_id: t.id, on: String(!t.mute) }); else if (k === "s") act("set_solo", { track_id: t.id, on: String(!t.solo) }); else if (k === "r") act("set_track_arm", { track_id: t.id, on: String(!t.armed) }); else call("set_monitor", { on: String(!S.monitor) }).then((r) => { refresh(); toast(r.message); }); }));
    $$("[data-fx]", box).forEach((b) => (b.onclick = () => openPlug(t, b.dataset.fx)));
    $$("[data-ins]", box).forEach((b) => (b.onclick = (e) => { const i = +b.dataset.ins; if (e.target.closest("[data-tog]")) return toggleIns(t, i); openIns(t, i); }));
    if ($("#in-add", box)) $("#in-add", box).onclick = (e) => addInsMenu(t, e.clientX, e.clientY);
    if ($("#in-qreset", box)) $("#in-qreset", box).onclick = () => act("set_fx", { track_id: t.id, low: 0, mid: 0, high: 0, comp: 0 }, "ההגדרות הבסיסיות אופסו");
    $$("[data-send]", box).forEach((r) => (r.oninput = () => (r.nextElementSibling.textContent = r.value + "%")));
    if ($("#in-rev", box)) { $("#in-rev", box).onchange = (e) => act("set_fx", { track_id: t.id, reverb: +e.target.value / 100 }); $("#in-del", box).onchange = (e) => act("set_delay_send", { track_id: t.id, level: +e.target.value / 100 }); }
    $$("[data-send]", box).forEach((r) => (r.onchange = () => act("set_track_send", { track_id: t.id, bus_id: r.dataset.send, level: +r.value / 100 })));
    if ($("#in-inst", box)) $("#in-inst", box).onchange = (e) => act("set_instrument", { track_id: t.id, instrument: e.target.value }, "הכלי הוחלף");
    if ($("#in-mic", box)) $("#in-mic", box).onclick = openSettings;
    const vol = $("#in-vol", box); vol.oninput = () => ($("#in-db").textContent = sliderToGain(+vol.value).toFixed(1) + " dB"); vol.onchange = () => act("set_gain", { track_id: t.id, db: sliderToGain(+vol.value) });
    knob($("#in-pan", box), t.pan, (v) => ($("#in-panv").textContent = panLabel(v)), (v) => act("set_pan", { track_id: t.id, pan: v }));
  }
  const panLabel = (p) => (Math.abs(p) < 0.02 ? "אמצע" : (p < 0 ? "שמאל " : "ימין ") + Math.round(Math.abs(p) * 100));
  function knob(el, v0, live, done) {
    if (!el) return;
    el.ondblclick = () => { el.style.setProperty("--a", "0deg"); done(0); };
    el.onpointerdown = (e) => {
      const y0 = e.clientY; let v = v0; el.setPointerCapture(e.pointerId);
      el.onpointermove = (ev) => { v = Math.max(-1, Math.min(1, v0 + (y0 - ev.clientY) / 100)); el.style.setProperty("--a", v * 135 + "deg"); live && live(v); };
      el.onpointerup = () => { el.onpointermove = el.onpointerup = null; if (Math.abs(v - v0) > 0.004) done(Math.round(v * 100) / 100); };
    };
  }

  // ---------------- library ----------------
  const LIB = INSTRUMENTS.map(([inst, n, c, i, role]) => [n, c, i, role, inst]);
  const SOON = [["בוזוקי", "ים תיכוני", "guitar"], ["עוד", "ים תיכוני", "oud"], ["פד סינתי", "סינתיסייזרים", "synth"]];
  const CATS = { inst: ["הכל", "פסנתרים", "קלידים", "גיטרות", "בסים", "תופים", "מיתרים", "נשיפה", "נשיפה ממתכת", "הקשה מלודית", "סינתיסייזרים", "ים תיכוני"], fx: [] };
  function renderLib() {
    $$("#libtabs .tab").forEach((b) => b.classList.toggle("on", b.dataset.l === ui.libTab));
    const q = $("#libq").value.trim(); $("#chips").innerHTML = (CATS[ui.libTab] || []).map((c) => `<button class="chip ${c === ui.libCat ? "on" : ""}">${c}</button>`).join("");
    $$("#chips .chip").forEach((c) => (c.onclick = () => { ui.libCat = c.textContent; renderLib(); }));
    let h = "";
    if (ui.libTab === "inst") {
      const ok = (n, c) => (ui.libCat === "הכל" || c.split("|").includes(ui.libCat)) && (!q || n.includes(q) || c.includes(q));
      h = LIB.filter(([n, c]) => ok(n, c)).map(([n, c, i, role, inst]) => `<div class="item" style="--c:${(ROLE[role] || ROLE.other)[1]}"><div class="ic">${ico(i)}</div><div class="t"><b>${n}</b><small>${c.split("|").join(" · ")}</small></div><span class="lic">חינם</span><button class="mini" title="הוספה כערוץ חדש" data-add="${role}:${inst}:${n}">${ico("plus")}</button></div>`).join("")
        + SOON.filter(([n, c]) => ok(n, c)).map(([n, c, i]) => `<div class="item" style="--c:#6c7787;opacity:.55"><div class="ic">${ico(i)}</div><div class="t"><b>${n}</b><small>${c}</small></div><span class="soon">בקרוב</span></div>`).join("");
    } else if (ui.libTab === "fx") {
      h = Object.entries(E.DEFS).filter(([, d]) => !q || d.he.includes(q) || d.desc.includes(q)).map(([k, d]) => `<div class="item" style="--c:#a0a8b8"><div class="ic">${ico("fx")}</div><div class="t"><b>${esc(d.he)}</b><small>${esc(d.desc)}</small></div><span class="lic">חינם</span><button class="mini" title="הוספה לערוץ הנבחר" data-ins="${k}">${ico("plus")}</button></div>`).join("");
    } else if (ui.libTab === "files") {
      h = `<button class="item" id="lib-imp-a" style="cursor:pointer"><div class="ic" style="--c:#e0567c">${ico("mic")}</div><div class="t"><b>ייבוא קובץ שמע</b><small>הקלטה מהטלפון, WAV, MP3, M4A</small></div></button><button class="item" id="lib-imp-m" style="cursor:pointer"><div class="ic" style="--c:#e2b14a">${ico("piano")}</div><div class="t"><b>ייבוא קובץ תווים (MIDI)</b><small>קובץ שיש לך זכות להשתמש בו</small></div></button>`;
    } else {
      h = '<div class="assist-off"><b>העוזר החכם יגיע אחרי שהאולפן יושלם.</b><br>קודם אולפן מלא שעובד בעכבר ובמקלדת. אחר כך העוזר ילחץ בשבילך על אותם כפתורים בדיוק — "תתחיל הקלטה", "תשתיק את התופים", "תוסיף ערוץ גיטרה" — ולא יוסיף שום דבר בלי שתבקש ותאשר.</div>';
    }
    $("#items").innerHTML = h || '<div class="assist-off">לא נמצא. נסה מילה אחרת.</div>';
    $$("#items [data-add]").forEach((b) => (b.onclick = async () => { const [role, inst, n] = b.dataset.add.split(":"); const r = await call("create_track", { role, instrument: inst, name: n }); if (r.status === "done") { ui.sel = r.track_id; refresh(); toast(`נוסף ערוץ "${n}"`); } }));
    $$("#items [data-ins]").forEach((b) => (b.onclick = () => { const t = selTrack(); if (!t) return toast("קודם בוחרים ערוץ", true); addIns(t, b.dataset.ins); }));
    if ($("#lib-imp-a")) { $("#lib-imp-a").onclick = () => $("#f-audio").click(); $("#lib-imp-m").onclick = () => $("#f-midi").click(); }
  }
  $$("#libtabs .tab").forEach((b) => (b.onclick = () => { ui.libTab = b.dataset.l; ui.libCat = "הכל"; renderLib(); }));
  $("#libq").oninput = renderLib;
  // the library tab "AI" stays frozen: keep its label honest
  $('#libtabs .tab[data-l="ai"]').textContent = "עוזר · בהמשך";

  // ---------------- new track dialog ----------------
  const TYPES = [["audio", "mic", "שמע – מיקרופון", "שירה, גיטרה אקוסטית, כל מה שמקליטים במיקרופון"], ["inst", "piano", "כלי וירטואלי", "פסנתר, בס, מיתרים, סינתיסייזר — מציירים תווים או מנגנים"], ["drums", "drums", "תופים", "ערכת תופים עם מכונת תופים"],
    ["line", "guitar", "גיטרה / בס בכבל", "הקלטת כלי שמחובר בכבל לכרטיס קול"], ["aux", "fx", "ערוץ אפקט משותף", "ריוורב או דיליי שכמה ערוצים שולחים אליו"], ["folder", "folder", "תיקייה", "קיבוץ כמה ערוצים יחד: עוצמה, השתקה וסולו לכולם"]];
  const AUDIO_KIND = [["vocals", "שירה"], ["guitar", "גיטרה אקוסטית"], ["other", "כלי אחר / מיקרופון"]];
  let ntType = "audio";
  function openNT(type) {
    if (type) ntType = type;
    $("#types").innerHTML = TYPES.map(([k, i, n, d, soon]) => `<button class="type ${k === ntType ? "on" : ""}" data-k="${k}" ${soon ? "disabled style='opacity:.45;cursor:not-allowed'" : ""}><div class="ic">${ico(i)}</div><div><b>${n}${soon ? ' <span class="soon">בשלב הבא</span>' : ""}</b><small>${d}</small></div></button>`).join("");
    $$(".type").forEach((b) => (b.onclick = () => { if (!b.disabled) openNT(b.dataset.k); }));
    const opts = ntType === "audio" ? AUDIO_KIND : ntType === "inst" ? INSTR.filter((x) => x[2] !== "drums").map((x) => [x[0], x[1]]) : ntType === "drums" ? [["drums", "מערכת תופים"], ["world_percussion", "דרבוקה, קחון ובונגו"]] : ntType === "aux" ? [["reverb", "ריוורב (חלל)"], ["delay", "דיליי (הד)"], ["chorus", "כורוס"]] : ntType === "folder" ? [["sel", "עם הערוץ הנבחר"], ["empty", "ריקה"]] : [["guitar", "גיטרה חשמלית"], ["bass", "בס חשמלי"]];
    $("#nt-inst").innerHTML = opts.map(([k, n]) => `<option value="${k}">${n}</option>`).join("");
    $("#nt-in").closest(".field").style.opacity = ntType === "audio" || ntType === "line" ? 1 : 0.4; $("#nt-in").disabled = !(ntType === "audio" || ntType === "line");
    $("#nt-arm").closest("label").style.display = ntType === "audio" || ntType === "line" ? "" : "none";
    $("#nt-mon").closest("label").style.display = ntType === "audio" || ntType === "line" ? "" : "none";
    $("#nt-in").innerHTML = '<option value="">המיקרופון שנבחר בהגדרות</option>';
    $("#nt-color").innerHTML = COLORS.map((c, i) => `<option value="${c}">${["ורוד", "ירוק", "סגול", "כחול", "זהב", "טורקיז", "כתום", "אפור"][i]}</option>`).join("");
    $("#nt-color").value = ntType === "audio" ? COLORS[0] : ntType === "drums" ? COLORS[3] : ntType === "line" ? COLORS[1] : COLORS[4];
    $("#nt-lib").closest("label").style.display = "none";
    $("#nt").classList.add("on");
  }
  $("#addtrack").onclick = () => openNT();
  $("#nt-ok").onclick = async () => {
    const n = Math.max(1, Math.min(16, +$("#nt-n").value || 1)), v = $("#nt-inst").value, name0 = $("#nt-name").value.trim(), label = $("#nt-inst").selectedOptions[0].textContent;
    let spec;
    if (ntType === "audio") spec = { role: v, kind: "audio", name: name0 || label };
    else if (ntType === "line") spec = { role: v, kind: "audio", name: name0 || label };
    else if (ntType === "drums") spec = v === "world_percussion" ? { role: "drums", instrument: v, name: name0 || "כלי הקשה" } : { role: "drums", name: name0 || "תופים" };
    else if (ntType === "aux") spec = { role: "other", kind: "bus", name: name0 || `ערוץ אפקט: ${label.split(" ")[0]}`, fxKind: v };
    else if (ntType === "folder") spec = { role: "other", kind: "bus", group: "true", name: name0 || "תיקייה", withSel: v === "sel" && selTrack() && selTrack().kind !== "bus" ? selTrack().id : null };
    else spec = { role: (INSTR.find((x) => x[0] === v) || [])[2] || "other", instrument: v, name: name0 || label };
    spec.color = $("#nt-color").value; if (spec.kind === "audio") spec.armed = $("#nt-arm").checked;
    let last = null;
    for (let k = 0; k < n; k++) {
      const { fxKind, withSel, ...args } = spec;
      const r = await call("create_track", { ...args, name: n > 1 ? `${spec.name} ${k + 1}` : spec.name }); if (r.status !== "done") break; last = r.track_id;
      if (withSel && k === 0) await call("set_track_output", { track_id: withSel, bus_id: last });
      if (fxKind) await call("set_track_inserts", { track_id: last, inserts: JSON.stringify([{ kind: fxKind, on: true, p: { ...E.defaults(fxKind), mix: 1 } }]) });
    }
    if (spec.kind === "audio" && $("#nt-mon").checked) await call("set_monitor", { on: "true" });
    $("#nt").classList.remove("on"); $("#nt-name").value = ""; $("#nt-n").value = 1;
    if (last) { ui.sel = last; refresh(); toast(spec.kind === "audio" && spec.armed ? `נוצר ערוץ "${spec.name}" — מוכן להקלטה. לחץ על כפתור ההקלטה (R)` : spec.kind === "bus" && spec.group ? `נוצרה "${spec.name}". מעבירים אליה ערוצים מתפריט ⋮ של כל ערוץ` : spec.kind === "bus" ? `נוצר "${spec.name}". בכל ערוץ אחר, במפקח, מכוונים כמה לשלוח אליו` : `נוצר ערוץ "${spec.name}"`); }
  };
  $$("[data-close]").forEach((b) => (b.onclick = () => b.closest(".scrim").classList.remove("on")));
  $$(".scrim").forEach((s) => (s.onpointerdown = (e) => { if (e.target === s && s.id !== "ask") s.classList.remove("on"); }));

  // ---------------- plugin windows (the effects the engine has today) ----------------
  let plugT = null, plugK = null;
  function openPlug(t, k) {
    plugT = t; plugK = k; const f = { ...(t.fx || {}) };
    $("#pw-title").textContent = { eq: "אקולייזר", comp: "קומפרסור", reverb: "ריוורב" }[k] + " · " + t.name;
    $("#pw-preset").innerHTML = '<option value="">הגדרות מוכנות…</option>' + (k === "eq" ? ["ללא שינוי", "שירה בהירה", "פחות בוץ", "בס עגול"] : ["כבוי", "עדין", "בינוני", "חזק"]).map((x, i) => `<option value="${i}">${x}</option>`).join("");
    $("#pw-bypass").style.display = "none";
    const b = $("#pw-body");
    const commit = (patch) => act("set_fx", { track_id: t.id, ...patch });
    if (k === "eq") {
      const bands = [["low", "בס (נמוך)", f.eq_low_db || 0], ["mid", "אמצע", f.eq_mid_db || 0], ["high", "גבוה (בהירות)", f.eq_high_db || 0]];
      const draw = () => {
        const W = 600, H = 190, fx = (fr) => (Math.log10(fr / 20) / 3) * W, fy = (g) => H / 2 - g * 6, C = [120, 1000, 8000]; let d = "";
        for (let i = 0; i <= 120; i++) { const fr = 20 * Math.pow(1000, i / 120); let g = 0; bands.forEach((bd, j) => { g += bd[2] * (j === 0 ? 1 / (1 + Math.pow(fr / C[0], 2)) : j === 2 ? 1 / (1 + Math.pow(C[2] / fr, 2)) : Math.exp(-Math.pow(Math.log2(fr / C[1]), 2) * 0.8)); }); d += (i ? "L" : "M") + fx(fr).toFixed(1) + "," + fy(g).toFixed(1); }
        $("#eqc").setAttribute("d", d);
      };
      b.innerHTML = `<svg class="eqsvg" viewBox="0 0 600 190" preserveAspectRatio="none">${[100, 1000, 10000].map((fr) => `<line x1="${(Math.log10(fr / 20) / 3) * 600}" x2="${(Math.log10(fr / 20) / 3) * 600}" y1="0" y2="190" stroke="#1d2733"/>`).join("")}<line x1="0" x2="600" y1="95" y2="95" stroke="#263241"/><path id="eqc" stroke="#e2b47e" stroke-width="2.2" fill="none"/></svg>
        <div class="knobs">${bands.map(([key, n, v]) => `<div class="kwrap"><div class="knob" data-b="${key}" style="--a:${(v / 12) * 135}deg"></div><b data-v="${key}">${v > 0 ? "+" : ""}${v.toFixed(1)} dB</b>${n}</div>`).join("")}</div><div class="lbl">גוררים כפתור למעלה כדי להגביר ולמטה כדי להנמיך. לחיצה כפולה מחזירה לאפס.</div>`;
      draw();
      $$("[data-b]", b).forEach((kn) => { const i = bands.findIndex((x) => x[0] === kn.dataset.b); knob(kn, bands[i][2] / 12, (v) => { bands[i][2] = Math.round(v * 120) / 10; $(`[data-v="${kn.dataset.b}"]`).textContent = (bands[i][2] > 0 ? "+" : "") + bands[i][2].toFixed(1) + " dB"; draw(); }, (v) => commit({ [kn.dataset.b]: Math.round(v * 120) / 10 })); });
      $("#pw-preset").onchange = (e) => { const P3 = [[0, 0, 0], [-2, 1, 4], [-3, -2, 1], [3, -1, -2]][+e.target.value]; commit({ low: P3[0], mid: P3[1], high: P3[2] }).then(() => openPlug(selTrack(), "eq")); };
    } else {
      const key = k, v = f[key] || 0;
      b.innerHTML = `<div class="knobs"><div class="kwrap"><div class="knob" id="pk" style="--a:${(v * 2 - 1) * 135}deg"></div><b id="pkv">${Math.round(v * 100)}%</b>${k === "comp" ? "עוצמת דחיסה" : "כמות חלל"}</div></div><div class="lbl">${k === "comp" ? "הקומפרסור מחליש את החלקים החזקים, כך שהקול נשמע יציב וקדימה במיקס." : "הריוורב מוסיף חלל — כמו לשיר בחדר או באולם. בשלב הבא: גודל חדר, זמן דעיכה ועוד."}</div>`;
      knob($("#pk", b), v * 2 - 1, (x) => ($("#pkv").textContent = Math.round(((x + 1) / 2) * 100) + "%"), (x) => commit({ [key]: Math.round(((x + 1) / 2) * 100) / 100 }));
      $("#pw-preset").onchange = (e) => commit({ [key]: [0, 0.25, 0.5, 0.8][+e.target.value] }).then(() => openPlug(selTrack(), key));
    }
    $("#pw").classList.add("on");
  }

  // ---------------- bottom panel ----------------
  function btab(k) { if ($("#app").classList.contains("no-bot")) toggleView("bot", true); ui.bottom = k; saveUi(); $$(".btab").forEach((b) => b.classList.toggle("on", b.dataset.b === k)); $$(".bpane").forEach((p) => p.classList.toggle("on", p.id === "p-" + k)); renderBottom(); }
  $$(".btab").forEach((b) => (b.onclick = () => btab(b.dataset.b)));
  function renderBottom() {
    const t = selTrack(); $("#bctx").textContent = ["mixer", "master"].includes(ui.bottom) || !t ? "" : `ערוץ: ${t.name}`;
    ({ mixer: renderMixer, piano: renderPiano, audio: renderAudio, steps: renderSteps, fx: renderFx, master: renderMaster, keys: renderKeys })[ui.bottom]();
  }
  function renderMixer() {
    const T = tracks();
    const strip = (t) => { const on = fxOn(t); return `<div class="strip ${t.id === ui.sel ? "pk" : ""}" data-id="${t.id}" style="--c:${tColor(t)}">
      <div class="nm" title="${esc(t.name)}">${esc(t.name)}</div>
      <div class="ins">${(t.inserts || []).slice(0, 2).map((x, i) => `<button data-ins="${i}" class="${x.on ? "" : "empty"}">${esc(E.DEFS[x.kind].he)}</button>`).join("")}${(t.inserts || []).length > 2 ? `<button data-more>+${t.inserts.length - 2} נוספים</button>` : ""}<button class="empty" data-addins>+ אפקט</button></div>
      <div class="kwrap"><div class="knob" data-pan style="--a:${t.pan * 135}deg" title="פאן"></div></div>
      <div class="fm"><input class="vfader" type="range" min="0" max="1000" value="${gainToSlider(t.gain_db)}" data-vol><div class="meter"><i data-meter="${t.id}"></i></div></div>
      <div class="db">${t.gain_db.toFixed(1)} dB${t.output ? ` · ← ${esc((tracks().find((b) => b.id === t.output) || {}).name || "")}` : ""}</div>
      <div class="msri"><button class="pill m ${t.mute ? "on" : ""}" data-k="m">M</button><button class="pill s ${t.solo ? "on" : ""}" data-k="s">S</button>${t.kind === "bus" ? "" : `<button class="pill r ${t.armed ? "on" : ""}" data-k="r">R</button>`}</div></div>`; };
    const buses = T.filter((t) => t.kind === "bus");
    $("#p-mixer").innerHTML = (T.length ? T.filter((t) => t.kind !== "bus").map(strip).join("") : '<div class="empty-hint">אין עדיין ערוצים</div>') + (buses.length ? '<div class="mixsep"></div>' + buses.map((t) => strip(t).replace('class="strip ', 'class="strip aux ')).join("") : "") + `<div class="mixsep"></div><div class="strip master" style="--c:var(--gold)"><div class="nm">יציאה ראשית</div><div class="ins"><button data-master>מאסטר וייצוא…</button></div><div class="fm"><input class="vfader" type="range" min="0" max="${gainToSlider(6)}" value="${gainToSlider(P() ? P().master_db || 0 : 0)}" id="mx-master"><div class="meter" style="width:8px"><i data-meter="master"></i></div><div class="meter" style="width:8px"><i data-meter="master"></i></div></div><div class="db" id="mx-mdb">${(P() ? P().master_db || 0 : 0).toFixed(1)} dB</div></div>`;
    const mf = $("#mx-master"); mf.oninput = () => ($("#mx-mdb").textContent = Math.min(6, sliderToGain(+mf.value)).toFixed(1) + " dB"); mf.onchange = () => act("set_master_gain", { db: Math.min(6, sliderToGain(+mf.value)) });
    $$("#p-mixer .strip[data-id]").forEach((s) => {
      const t = T.find((x) => x.id === s.dataset.id);
      s.onclick = (e) => { const k = e.target.closest("[data-k]")?.dataset.k, fx = e.target.closest("[data-fx]")?.dataset.fx, ii = e.target.closest("[data-ins]"); if (fx) return openPlug(t, fx); if (ii) { select(t.id, true); return openIns(t, +ii.dataset.ins); } if (e.target.closest("[data-more]")) { select(t.id, true); return btab("fx"); } if (e.target.closest("[data-addins]")) { select(t.id, true); return addInsMenu(t, e.clientX, e.clientY); } if (k === "m") return act("set_mute", { track_id: t.id, on: String(!t.mute) }); if (k === "s") return act("set_solo", { track_id: t.id, on: String(!t.solo) }); if (k === "r") return act("set_track_arm", { track_id: t.id, on: String(!t.armed) }); if (!e.target.matches("input") && !e.target.closest(".knob")) select(t.id, true); };
      const v = $("[data-vol]", s); v.oninput = () => ($(".db", s).textContent = sliderToGain(+v.value).toFixed(1) + " dB"); v.onchange = () => act("set_gain", { track_id: t.id, db: sliderToGain(+v.value) });
      knob($("[data-pan]", s), t.pan, null, (p) => act("set_pan", { track_id: t.id, pan: p }));
    });
    const m = $("[data-master]"); if (m) m.onclick = () => btab("master");
  }
  // piano roll: the selected note clip of the selected instrument track
  const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  function clipOf(t, kind) {
    if (!t) return null;
    const list = kind === "audio" ? t.clips || [] : t.midi || [];
    return (ui.region && list.find((c) => c.id === ui.region.id)) || list[0] || null;
  }
  // piano roll: click empty = new note; click a note = select (Shift adds); drag = move; drag its right edge = longer/shorter;
  // double-click or Delete = remove; the velocity row sets the loudness of the note above the click
  function renderPiano() {
    const t = selTrack(), p = $("#p-piano");
    if (!t || t.kind !== "midi") { p.innerHTML = `<div class="empty-hint">${t ? `"${esc(t.name)}" הוא ערוץ שמע — בו מקליטים. עורך התווים מיועד לכלים וירטואליים.` : "בחר ערוץ של כלי וירטואלי"}</div>`; return; }
    const c = clipOf(t, "midi");
    if (!c) { p.innerHTML = `<div class="empty-hint">אין עדיין קטע תווים בערוץ "${esc(t.name)}".<br><button class="btn gold" id="pr-new">קטע חדש של 4 תיבות בסמן</button></div>`; $("#pr-new").onclick = () => newClipAtPlayhead(t); return; }
    if (ui.pr.clip !== c.id) { ui.pr.clip = c.id; ui.pr.sel = new Set(); }
    const notes = c.notes.map((n) => ({ pitch: n[0], st: n[5], dt: n[6], vel: n[3] }));
    const key = (n) => `${n.pitch}:${n.st}`;
    const lo0 = notes.length ? Math.min(...notes.map((n) => n.pitch)) : 55, hi0 = notes.length ? Math.max(...notes.map((n) => n.pitch)) : 72;
    const lo = Math.max(0, Math.min(lo0 - 5, 48)), hi = Math.min(127, Math.max(hi0 + 5, lo + 24)), RH = 13;
    const qpb = (secPerBar() * bpm()) / 60; // quarter notes per bar
    const endQ = Math.max(...notes.map((n) => (n.st + n.dt) / TPB), 0.01), bars = Math.max(4, Math.ceil(endQ / qpb));
    const avail = Math.max(300, (p.clientWidth || 900) - 80), pxBeat = Math.max(18, ui.zoom * 1.6, avail / (bars * qpb)), W = bars * qpb * pxBeat;
    const tx = (tick) => (tick / TPB) * pxBeat;
    let keys = ""; for (let n = hi; n >= lo; n--) keys += `<div class="key ${NAMES[n % 12].includes("#") ? "b" : ""}" style="height:${RH}px">${n % 12 === 0 ? "C" + (Math.floor(n / 12) - 1) : ""}</div>`;
    p.innerHTML = `<div class="prwrap"><div class="prtools"><b>${esc(t.name)}</b><span class="lbl">לחיצה במקום ריק: תו חדש · לחיצה על תו: בחירה · גרירה: הזזה · קצה ימני: אורך · לחיצה כפולה או Delete: מחיקה</span><span style="flex:1"></span>
      <label class="lbl">אורך תו חדש <select class="sel" id="pr-len"><option value="240">1/16</option><option value="480" selected>1/8</option><option value="960">רבע</option><option value="1920">חצי</option></select></label>
      <button class="btn" id="pr-q">יישור לקצב</button><button class="btn" id="pr-up">+12</button><button class="btn" id="pr-dn">−12</button></div>
      <div class="prscroll"><div class="pr" style="grid-template-rows:${(hi - lo + 1) * RH}px 44px;width:${W + 54}px"><div class="keys">${keys}</div>
      <div class="prgrid" id="prgrid" style="width:${W}px;background-image:repeating-linear-gradient(0deg,#1b2430 0 1px,transparent 1px ${RH}px),repeating-linear-gradient(90deg,#33424f 0 1px,transparent 1px ${pxBeat * qpb}px),repeating-linear-gradient(90deg,#1c2631 0 1px,transparent 1px ${pxBeat}px)">
      ${notes.map((n, i) => `<div class="note ${ui.pr.sel.has(key(n)) ? "selnote" : ""}" data-i="${i}" style="--c:${tColor(t)};left:${tx(n.st)}px;width:${Math.max(4, tx(n.dt) - 1)}px;top:${(hi - n.pitch) * RH + 1}px;height:${RH - 2}px;opacity:${0.45 + (n.vel / 127) * 0.55}"><i class="nres"></i></div>`).join("")}</div>
      <div class="prside">עוצמה</div><div class="vel" id="prvel" title="לחיצה קובעת את עוצמת התו שמעליה">${notes.map((n) => `<i style="left:${tx(n.st) + 1}px;height:${(n.vel / 127) * 36}px"></i>`).join("")}</div></div></div></div>`;
    const save = (list, msg) => act("set_midi_notes", { clip_id: c.id, notes: list.map((n) => `${n.pitch},${Math.max(0, Math.round(n.st))},${Math.max(1, Math.round(n.dt))},${n.vel}`).join(";") }, msg);
    ui.pr.remove = () => { if (!ui.pr.sel.size) return false; const l = notes.filter((n) => !ui.pr.sel.has(key(n))); ui.pr.sel = new Set(); save(l, "התווים נמחקו"); return true; };
    const g = $("#prgrid"), q = () => (ui.snap ? Math.max(120, Math.round(snapQ() * TPB)) : 120); // grid in ticks: the snap setting, finest 1/32
    g.ondblclick = (e) => { const ne = e.target.closest(".note"); if (!ne) return; const l = notes.slice(); l.splice(+ne.dataset.i, 1); ui.pr.sel = new Set(); save(l, "התו נמחק"); };
    g.onpointerdown = (e) => {
      const r = g.getBoundingClientRect(), tick = ((e.clientX - r.left) / pxBeat) * TPB, pitch = hi - Math.floor((e.clientY - r.top) / RH);
      const ne = e.target.closest(".note");
      if (ne) {
        const i = +ne.dataset.i, n0 = notes[i], x0 = e.clientX, y0 = e.clientY, resize = !!e.target.closest(".nres"); let moved = false; ne.setPointerCapture(e.pointerId);
        if (!ui.pr.sel.has(key(n0))) { if (!e.shiftKey) ui.pr.sel = new Set(); ui.pr.sel.add(key(n0)); $$("#prgrid .note").forEach((x) => x.classList.toggle("selnote", ui.pr.sel.has(key(notes[+x.dataset.i])))); }
        ne.onpointermove = (ev) => { if (Math.abs(ev.clientX - x0) + Math.abs(ev.clientY - y0) > 4) moved = true; if (!moved) return; if (resize) ne.style.width = Math.max(4, tx(n0.dt) + ev.clientX - x0) + "px"; else ne.style.transform = `translate(${ev.clientX - x0}px,${Math.round((ev.clientY - y0) / RH) * RH}px)`; };
        ne.onpointerup = (ev) => {
          ne.onpointermove = ne.onpointerup = null; if (!moved) return;
          const dT = Math.round((((ev.clientX - x0) / pxBeat) * TPB) / q()) * q(), dp = -Math.round((ev.clientY - y0) / RH);
          const l = notes.map((n) => {
            if (!ui.pr.sel.has(key(n))) return n;
            if (resize) return { ...n, dt: Math.max(q(), n.dt + dT) };
            return { ...n, st: Math.max(0, n.st + dT), pitch: Math.max(0, Math.min(127, n.pitch + dp)) };
          });
          if (!resize) ui.pr.sel = new Set(l.filter((n, k) => ui.pr.sel.has(key(notes[k]))).map(key));
          save(l);
        };
        return;
      }
      if (pitch < 0 || pitch > 127) return;
      const st = Math.floor(tick / q()) * q(), n = { pitch, st, dt: +$("#pr-len").value, vel: 96 };
      ui.pr.sel = new Set([key(n)]);
      save([...notes, n], `נוסף תו ${NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`);
    };
    $("#prvel").onpointerdown = (e) => {
      const r = e.currentTarget.getBoundingClientRect(), tick = ((e.clientX - r.left) / pxBeat) * TPB, v = Math.max(1, Math.min(127, Math.round(((r.bottom - 4 - e.clientY) / 36) * 127)));
      const hit = notes.filter((n) => tick >= n.st - 60 && tick <= n.st + n.dt);
      if (!hit.length) return;
      save(notes.map((n) => (hit.includes(n) || ui.pr.sel.has(key(n)) && ui.pr.sel.has(key(hit[0])) ? { ...n, vel: v } : n)), `עוצמת תו: ${v}`);
    };
    $("#pr-q").onclick = () => save(notes.map((n) => ({ ...n, st: Math.round(n.st / q()) * q() })), "התווים יושרו לקצב");
    $("#pr-up").onclick = () => save(notes.map((n) => ({ ...n, pitch: Math.min(127, n.pitch + 12) })), "אוקטבה למעלה");
    $("#pr-dn").onclick = () => save(notes.map((n) => ({ ...n, pitch: Math.max(0, n.pitch - 12) })), "אוקטבה למטה");
  }
  function renderAudio() {
    const t = selTrack(), p = $("#p-audio");
    if (!t || t.kind !== "audio") { p.innerHTML = `<div class="empty-hint">${t ? `"${esc(t.name)}" הוא כלי וירטואלי. עורך השמע מיועד להקלטות.` : "בחר ערוץ שמע"}</div>`; return; }
    const c = clipOf(t, "audio");
    if (!c) { p.innerHTML = `<div class="empty-hint">אין עדיין הקלטה בערוץ "${esc(t.name)}". לחץ R בערוץ ואז על כפתור ההקלטה.</div>`; return; }
    const e = c.edit || {};
    p.innerHTML = `<div class="ae"><div class="bar"><b>${esc(t.name)}</b><span class="lbl ltr">${fmtT(c.start)} · ${c.length.toFixed(2)}s מתוך ${c.total.toFixed(2)}s</span><span style="flex:1"></span>
      <button class="btn" data-a="fin">כניסה רכה</button><button class="btn" data-a="fout">יציאה רכה</button><button class="btn" data-a="nofade">בלי כניסה/יציאה</button><button class="btn" data-a="full">החזרת ההקלטה המלאה</button><button class="btn" data-a="norm">נרמול עוצמה</button><button class="btn" data-a="split">חיתוך בסמן</button><button class="btn" data-a="dup">שכפול</button><button class="btn" data-a="play">ניגון מתחילת הקטע</button><button class="btn" data-a="del">מחיקה</button>
      <span class="tunebox"><b>כוונון קול</b> <select class="sel" id="tu-s"><option value="0.5">עדין</option><option value="0.8" selected>רגיל</option><option value="1">מלא</option></select> <select class="sel" id="tu-v"><option value="60">טבעי</option><option value="25">מהיר</option><option value="5">אפקט רובוטי</option></select> <select class="sel" id="tu-m"><option value="key">לפי סולם השיר</option><option value="chromatic">לחצי הטון הקרוב</option></select> <button class="btn gold" data-a="tune">כוונון</button></span>
      <label class="lbl">עוצמת הקטע <input type="range" min="-24" max="12" step="0.5" value="${e.gain_db || 0}" id="ae-g" style="width:110px;accent-color:var(--gold)"> <span class="ltr" id="ae-gv">${(e.gain_db || 0).toFixed(1)} dB</span></label></div>
      <div class="wave" style="--c:${tColor(t)}"><svg viewBox="0 0 1000 100" preserveAspectRatio="none" style="fill:${tColor(t)}">${(wavePath(c.peaks, 1000, 100).match(/<path[^>]*>/) || [""])[0]}</svg>
      ${e.fade_in_ms ? `<div class="fin" style="left:0;width:${(e.fade_in_ms / 1000 / c.length) * 100}%"></div>` : ""}${e.fade_out_ms ? `<div class="fout" style="right:0;width:${(e.fade_out_ms / 1000 / c.length) * 100}%"></div>` : ""}</div></div>`;
    $$("#p-audio [data-a]").forEach((b) => (b.onclick = async () => { const a = b.dataset.a; ui.region = { id: c.id, kind: "audio", tid: t.id };
      if (a === "fin") act("set_clip_edit", { clip_id: c.id, fade_in_ms: 400 }, "כניסה רכה"); else if (a === "fout") act("set_clip_edit", { clip_id: c.id, fade_out_ms: 600 }, "יציאה רכה"); else if (a === "nofade") act("set_clip_edit", { clip_id: c.id, fade_in_ms: 0, fade_out_ms: 0 });
      else if (a === "full") act("set_clip_edit", { clip_id: c.id, start_tick: s2t(Math.max(0, c.start - (e.trim_start || 0) / (c.sr || 48000))), trim_start: 0, trim_len: "null" }, "ההקלטה המלאה הוחזרה");
      else if (a === "play") { await call("seek", { seconds: c.start }); await call("transport", { op: "play" }); startPulse(); } else if (a === "del") delRegion();
      else if (a === "norm") { const r = await act("normalize_clip", { clip_id: c.id }); if (r.status === "done") toast(r.message); }
      else if (a === "tune") { toast("מכוונן את השירה…"); await new Promise((res) => setTimeout(res, 30)); const r = await act("tune_clip", { clip_id: c.id, strength: $("#tu-s").value, speed_ms: $("#tu-v").value, mode: $("#tu-m").value === "chromatic" ? "chromatic" : "" }); if (r.status === "done") toast(r.message); } else if (a === "split") splitAtPlayhead(); else if (a === "dup") dupRegion(); }));
    const g = $("#ae-g"); g.oninput = () => ($("#ae-gv").textContent = (+g.value).toFixed(1) + " dB"); g.onchange = () => { ui.region = { id: c.id, kind: "audio", tid: t.id }; act("set_clip_edit", { clip_id: c.id, gain_db: +g.value }); };
  }
  // step sequencer: one bar of 16 steps on the selected drum clip, repeated over the clip (GM drum notes)
  const DRUMS = [["קיק", 36], ["סנר", 38], ["היי-האט סגור", 42], ["היי-האט פתוח", 46], ["קלאפ", 39], ["טום נמוך", 45], ["טום גבוה", 50], ["מצילה", 49]];
  // the world-percussion kit (FreePats key map): darbuka strokes first, as in Mediterranean rhythms
  const PERC = [["דרבוקה דום", 69], ["דרבוקה טק", 70], ["דרבוקה פה", 71], ["קחון", 48], ["קחון חד", 50], ["בונגו גבוה", 52], ["בונגו נמוך", 53], ["קונגה", 62], ["טמבורין", 57], ["שייקר", 55], ["כפיים", 59], ["קסטנייטות", 61]];
  function renderSteps() {
    const t = selTrack(), p = $("#p-steps");
    if (!t || t.role !== "drums") {
      const ex = tracks().find((x) => x.role === "drums");
      p.innerHTML = `<div class="empty-hint">מכונת התופים עובדת על ערוץ תופים.<br>${ex ? `<button class="btn gold" id="sq-go">מעבר לערוץ "${esc(ex.name)}"</button>` : '<button class="btn gold" id="sq-add">ערוץ תופים חדש</button>'}</div>`;
      if (ex) $("#sq-go").onclick = () => { select(ex.id); btab("steps"); };
      else $("#sq-add").onclick = async () => { const r = await call("create_track", { role: "drums", name: "תופים" }); if (r.status === "done") { ui.sel = r.track_id; refresh(); } };
      return;
    }
    const c = clipOf(t, "midi");
    if (!c) { p.innerHTML = `<div class="empty-hint">אין עדיין קטע תופים בערוץ.<br><button class="btn gold" id="sq-new">קטע תופים חדש של 4 תיבות בסמן</button></div>`; $("#sq-new").onclick = () => newClipAtPlayhead(t); return; }
    const STEP = TPB / 4, BAR = Math.round((secPerBar() * bpm() / 60) * TPB), NS = Math.round(BAR / STEP), bars = Math.max(4, Math.round(Math.max(c.length, secPerBar()) / secPerBar()));
    const notes = c.notes.map((n) => ({ pitch: n[0], st: n[5], dt: n[6], vel: n[3] }));
    const on = (pitch, s) => notes.some((n) => n.pitch === pitch && n.st % BAR === s * STEP && n.st < BAR);
    const ROWS = t.instrument === "world_percussion" ? PERC : DRUMS;
    p.innerHTML = `<div class="seq"><div class="seqrow" style="direction:rtl;display:flex;gap:8px;align-items:center;margin-bottom:4px"><b>${esc(t.name)}</b><span class="lbl">מתכנתים תיבה אחת, והיא חוזרת על כל ${bars} התיבות של הקטע</span></div>
      ${ROWS.map(([n, pitch]) => `<div class="seqrow" style="grid-template-columns:110px repeat(${NS},1fr)"><span class="nm">${n}</span>${Array.from({ length: NS }, (_, s) => `<button class="step ${on(pitch, s) ? "on" : ""}" data-p="${pitch}" data-s="${s}"></button>`).join("")}</div>`).join("")}</div>`;
    $$("#p-steps .step").forEach((b) => (b.onclick = () => {
      const pitch = +b.dataset.p, s = +b.dataset.s, was = on(pitch, s);
      let list = notes.filter((n) => !(n.pitch === pitch && n.st % BAR === s * STEP));
      if (!was) for (let k = 0; k < bars; k++) list.push({ pitch, st: k * BAR + s * STEP, dt: STEP, vel: [36, 38, 69, 48].includes(pitch) ? 110 : 88 });
      act("set_midi_notes", { clip_id: c.id, notes: list.map((n) => `${n.pitch},${n.st},${n.dt},${n.vel}`).join(";") });
    }));
  }
  function renderFx() {
    const t = selTrack(), p = $("#p-fx");
    if (!t) { p.innerHTML = '<div class="empty-hint">בחר ערוץ</div>'; return; }
    const ins = t.inserts || [];
    p.innerHTML = `<div class="fx"><div class="chain"><div class="sect">שרשרת האפקטים של "${esc(t.name)}" — הצליל עובר מלמעלה למטה</div>
      ${ins.map((x, i) => `<div class="slot ${x.on ? "" : "off"}" style="cursor:default"><i data-tog="${i}" style="cursor:pointer" title="הפעלה / כיבוי"></i><span style="flex:1;cursor:pointer" data-ins="${i}">${i + 1}. ${esc(E.DEFS[x.kind].he)}</span>
        <button class="mini" data-up="${i}" title="למעלה" ${i ? "" : "disabled"}>▲</button><button class="mini" data-dn="${i}" title="למטה" ${i < ins.length - 1 ? "" : "disabled"}>▼</button><button class="mini" data-rm="${i}" title="הסרה">✕</button></div>`).join("") || '<div class="assist-off">אין עדיין אפקטים בערוץ. בוחרים מהרשימה מימין.</div>'}
      <div class="sect" style="margin-top:6px">אקולייזר וקומפרסור מהירים של הערוץ, ושליחות: במפקח</div></div>
      <div class="fxgrid">${Object.entries(E.DEFS).map(([k, d]) => `<button class="fxcard" data-add="${k}" ${ins.length >= E.MAX_INSERTS ? "disabled" : ""}><span class="tag">${esc(d.group)}</span><b>${esc(d.he)}</b><small>${esc(d.desc)}</small></button>`).join("")}</div></div>`;
    $$("#p-fx [data-ins]").forEach((b) => (b.onclick = () => openIns(t, +b.dataset.ins)));
    $$("#p-fx [data-tog]").forEach((b) => (b.onclick = () => toggleIns(t, +b.dataset.tog)));
    $$("#p-fx [data-add]").forEach((b) => (b.onclick = () => addIns(t, b.dataset.add)));
    const mv = (i, j) => { const l = ins.slice(); [l[i], l[j]] = [l[j], l[i]]; return saveIns(t, l); };
    $$("#p-fx [data-up]").forEach((b) => (b.onclick = () => mv(+b.dataset.up, +b.dataset.up - 1)));
    $$("#p-fx [data-dn]").forEach((b) => (b.onclick = () => mv(+b.dataset.dn, +b.dataset.dn + 1)));
    $$("#p-fx [data-rm]").forEach((b) => (b.onclick = () => { const l = ins.slice(); const [x] = l.splice(+b.dataset.rm, 1); saveIns(t, l, `"${E.DEFS[x.kind].he}" הוסר`); }));
  }
  // ---- insert effects (effects.js) ----
  const saveIns = (t, list, msg) => act("set_track_inserts", { track_id: t.id, inserts: JSON.stringify(list) }, msg);
  const toggleIns = (t, i) => { const l = JSON.parse(JSON.stringify(t.inserts)); l[i].on = !l[i].on; return saveIns(t, l, l[i].on ? "האפקט פעיל" : "האפקט עקוף"); };
  async function addIns(t, kind) {
    if ((t.inserts || []).length >= E.MAX_INSERTS) return toast("עד 8 אפקטים בערוץ", true);
    const r = await saveIns(t, [...(t.inserts || []), { kind, on: true, p: E.defaults(kind) }], `"${E.DEFS[kind].he}" נוסף לערוץ "${t.name}"`);
    if (r.status === "done") openIns(selTrack(), selTrack().inserts.length - 1);
  }
  function addInsMenu(t, x, y) {
    const groups = {}; for (const [k, d] of Object.entries(E.DEFS)) (groups[d.group] = groups[d.group] || []).push([k, d]);
    pop(x, y, Object.entries(groups).map(([g, l]) => `<div class="lbl" style="padding:4px 10px">${esc(g)}</div>${l.map(([k, d]) => `<button data-k="${k}"><span>${esc(d.he)}</span><span class="lbl">${esc(d.desc)}</span></button>`).join("")}`).join(""), (pp) => {
      $$("[data-k]", pp).forEach((b) => (b.onclick = () => { closePop(); addIns(t, b.dataset.k); }));
    });
  }
  const isFreq = (d) => d[4] === "Hz";
  const toKnob = (d, v) => (isFreq(d) ? (Math.log(v / d[1]) / Math.log(d[2] / d[1])) * 2 - 1 : ((v - d[1]) / (d[2] - d[1])) * 2 - 1);
  const fromKnob = (d, k) => { const u = (k + 1) / 2; let v = isFreq(d) ? d[1] * Math.pow(d[2] / d[1], u) : d[1] + u * (d[2] - d[1]); if (d[4] === "steps") v = Math.round(v); return Math.round(v * 100) / 100; };
  function fmtP(kind, d, v) {
    if (kind === "delay" && d[0] === "div") return E.DIVS[v][0];
    if (kind === "filter" && d[0] === "type") return E.FILTERS_HE[v];
    if (d[4] === "Hz") return v >= 1000 ? (v / 1000).toFixed(1) + "k Hz" : Math.round(v) + " Hz";
    if (d[4] === "" && d[2] <= 2) return Math.round(v * 100) + "%";
    return (Math.round(v * 10) / 10) + " " + d[4];
  }
  function eqCurve(p) { // the real filter response, from the same Web Audio nodes the sound uses
    try {
      const oc = new OfflineAudioContext(1, 1, 44100), N = 200, fr = new Float32Array(N), mag = new Float32Array(N), ph = new Float32Array(N), tot = new Float32Array(N).fill(0);
      for (let i = 0; i < N; i++) fr[i] = 20 * Math.pow(1000, i / (N - 1));
      const bq = []; // the same five filters effects.js builds for "eq"
      const B2 = (type, f, g, q) => { const b = oc.createBiquadFilter(); b.type = type; b.frequency.value = f; if (g !== undefined) b.gain.value = g; if (q) b.Q.value = q; return b; };
      bq.push(B2("highpass", p.hp, undefined, 0.707), B2("lowshelf", p.low_f, p.low_g), B2("peaking", p.mid_f, p.mid_g, p.mid_q), B2("peaking", p.pres_f, p.pres_g, 1.1), B2("highshelf", p.high_f, p.high_g));
      for (const b of bq) { b.getFrequencyResponse(fr, mag, ph); for (let i = 0; i < N; i++) tot[i] += 20 * Math.log10(Math.max(1e-6, mag[i])); }
      let d = ""; for (let i = 0; i < N; i++) d += (i ? "L" : "M") + ((i / (N - 1)) * 600).toFixed(1) + "," + (95 - Math.max(-15, Math.min(15, tot[i])) * 6).toFixed(1);
      return d;
    } catch { return ""; }
  }
  function openIns(t, i) {
    const x = t.inserts[i], d0 = E.DEFS[x.kind], p = { ...x.p };
    $("#pw-title").textContent = d0.he + " · " + t.name;
    const pres = E.PRESETS[x.kind] || [];
    $("#pw-preset").innerHTML = '<option value="">הגדרות מוכנות…</option><option value="-1">ברירת מחדל</option>' + pres.map(([n], k) => `<option value="${k}">${esc(n)}</option>`).join("");
    $("#pw-bypass").style.display = ""; $("#pw-bypass").textContent = x.on ? "עקיפה" : "הפעלה"; $("#pw-bypass").classList.toggle("gold", !x.on);
    const commit = (np, msg) => { const l = JSON.parse(JSON.stringify(selTrack().inserts)); l[i] = { ...l[i], p: np }; return saveIns(t, l, msg); };
    const b = $("#pw-body");
    b.innerHTML = `${x.kind === "eq" ? '<svg class="eqsvg" viewBox="0 0 600 190" preserveAspectRatio="none"><line x1="0" x2="600" y1="95" y2="95" stroke="#263241"/>' + [100, 1000, 10000].map((f) => `<line x1="${(Math.log10(f / 20) / 3) * 600}" x2="${(Math.log10(f / 20) / 3) * 600}" y1="0" y2="190" stroke="#1d2733"/><text x="${(Math.log10(f / 20) / 3) * 600 + 3}" y="184" fill="#5c6878" font-size="10">${f >= 1000 ? f / 1000 + "k" : f}</text>`).join("") + '<path id="eqc" stroke="#e2b47e" stroke-width="2.2" fill="none"/></svg>' : ""}
      <div class="knobs">${d0.params.map((d) => `<div class="kwrap"><div class="knob" data-pk="${d[0]}" style="--a:${toKnob(d, p[d[0]]) * 135}deg"></div><b data-pv="${d[0]}">${fmtP(x.kind, d, p[d[0]])}</b>${esc(d[5])}</div>`).join("")}</div>
      <div class="lbl">${esc(d0.desc)}. גוררים כפתור למעלה או למטה; לחיצה כפולה מחזירה לברירת המחדל.</div>
      <div class="row"><span style="flex:1"></span><button class="btn" id="pw-rm">הסרת האפקט מהערוץ</button></div>`;
    const draw = () => { if (x.kind === "eq") $("#eqc").setAttribute("d", eqCurve(p)); };
    draw();
    $$("[data-pk]", b).forEach((kn) => {
      const d = d0.params.find((q) => q[0] === kn.dataset.pk);
      kn.ondblclick = () => { p[d[0]] = d[3]; kn.style.setProperty("--a", toKnob(d, d[3]) * 135 + "deg"); $(`[data-pv="${d[0]}"]`, b).textContent = fmtP(x.kind, d, d[3]); draw(); commit({ ...p }); };
      kn.onpointerdown = (e) => {
        const y0 = e.clientY, k0 = toKnob(d, p[d[0]]); let k = k0; kn.setPointerCapture(e.pointerId);
        kn.onpointermove = (ev) => { k = Math.max(-1, Math.min(1, k0 + (y0 - ev.clientY) / 120)); p[d[0]] = fromKnob(d, k); kn.style.setProperty("--a", k * 135 + "deg"); $(`[data-pv="${d[0]}"]`, b).textContent = fmtP(x.kind, d, p[d[0]]); draw(); };
        kn.onpointerup = () => { kn.onpointermove = kn.onpointerup = null; if (Math.abs(k - k0) > 0.002) commit({ ...p }); };
      };
    });
    $("#pw-preset").onchange = (e) => { const v = e.target.value; if (v === "") return; const np = v === "-1" ? E.defaults(x.kind) : { ...E.defaults(x.kind), ...pres[+v][1] }; commit(np, "הוגדר").then(() => openIns(selTrack(), i)); };
    $("#pw-bypass").onclick = () => toggleIns(selTrack(), i).then(() => openIns(selTrack(), i));
    $("#pw-rm").onclick = () => { const l = JSON.parse(JSON.stringify(selTrack().inserts)); l.splice(i, 1); $("#pw").classList.remove("on"); saveIns(t, l, `"${d0.he}" הוסר`); };
    $("#pw").classList.add("on");
  }
  function renderMaster() {
    $("#p-master").innerHTML = `<div class="mst"><div class="box"><h4>מאסטר — העוצמה הסופית של השיר</h4><div class="row"><span class="lbl">יעד</span><select class="sel" id="ms-p"><option value="streaming">סטרימינג (ספוטיפיי / יוטיוב) −14</option><option value="commercial">מסחרי חזק −10</option><option value="natural">טבעי ודינמי −16</option><option value="broadcast">שידור −23</option></select></div>
      <button class="btn" id="ms-prev">הכנת מאסטר להשוואה</button><div class="row"><button class="btn" id="ms-a">לפני</button><button class="btn" id="ms-b">אחרי</button><button class="btn" id="ms-s">עצירה</button></div><div class="lbl" id="ms-msg"></div></div>
      <div class="box"><h4>עוצמה כוללת</h4><div class="big" id="ms-lufs">—</div><div class="lbl">מוצג אחרי "הכנת מאסטר"</div></div>
      <div class="box"><h4>ייצוא</h4><div class="row"><span class="lbl">פורמט</span><select class="sel" id="ex-f"><option value="wav16">WAV 16 ביט</option><option value="wav24">WAV 24 ביט</option></select></div>
      <div class="checks"><label><input type="checkbox" id="ex-m" checked> עם מאסטר</label></div><button class="btn gold" id="ex-go">ייצוא השיר</button><button class="btn" id="ex-st">ייצוא כל ערוץ לקובץ נפרד</button></div></div>`;
    $("#ms-prev").onclick = async () => { $("#ms-msg").textContent = "מכין…"; const r = await call("master_preview", { preset: $("#ms-p").value }); if (r.status === "done") { $("#ms-msg").textContent = r.message; const a = r.report && r.report.after; $("#ms-lufs").textContent = a && isFinite(a.integrated) ? a.integrated.toFixed(1) + " LUFS" : "—"; } else $("#ms-msg").textContent = r.message || ""; };
    $("#ms-a").onclick = () => call("master_play", { which: "before" }); $("#ms-b").onclick = () => call("master_play", { which: "after" }); $("#ms-s").onclick = () => call("master_play", { which: "stop" });
    $("#ex-go").onclick = async () => { toast("מכין את הקובץ…"); const r = await call("export_audio", { master: $("#ex-m").checked ? $("#ms-p").value : "", format: $("#ex-f").value }); if (r.status === "done") toast(r.message); };
    $("#ex-st").onclick = () => exportSong(true);
  }

  // ---------------- playing an instrument: computer keyboard, on-screen keys, MIDI keyboard ----------------
  const KEYMAP = { KeyA: 0, KeyW: 1, KeyS: 2, KeyE: 3, KeyD: 4, KeyF: 5, KeyT: 6, KeyG: 7, KeyY: 8, KeyH: 9, KeyU: 10, KeyJ: 11, KeyK: 12, KeyO: 13, KeyL: 14, KeyP: 15, Semicolon: 16 };
  ui.oct = 4; ui.keys = false; let midiIn = null;
  const down = new Set();
  function playTrack() { const s0 = selTrack(); if (s0 && s0.kind === "midi") return s0; return tracks().find((t) => t.armed && t.kind === "midi") || null; }
  function noteOn(pitch, vel = 100) { const t = playTrack(); if (!t) { toast("בוחרים ערוץ של כלי וירטואלי כדי לנגן", true); return false; } LAB.call("live_note", { track_id: t.id, pitch, velocity: vel, on: "true" }); keyLit(pitch, true); return true; }
  function noteOff(pitch) { const t = playTrack(); if (t) LAB.call("live_note", { track_id: t.id, pitch, on: "false" }); keyLit(pitch, false); }
  function keyLit(pitch, on) { const k = $(`#p-keys [data-n="${pitch}"]`); if (k) k.classList.toggle("down", on); }
  function setKeysUi() { $("#b-keys").classList.toggle("on", ui.keys); }
  $("#b-keys").onclick = () => { ui.keys = !ui.keys; setKeysUi(); toast(ui.keys ? "נגינה במקלדת המחשב: A עד K הם הקלידים, W E T Y U השחורים, Z/X אוקטבה. רווח עדיין עוצר" : "נגינה במקלדת כבויה — המקשים חוזרים לקיצורים"); };
  document.addEventListener("keyup", (e) => { if (!ui.keys) return; const k = KEYMAP[e.code]; if (k === undefined) return; const pitch = 12 * (ui.oct + 1) + k; if (down.has(e.code)) { down.delete(e.code); noteOff(pitch); } });
  function keysDown(e) {
    if (!ui.keys || e.ctrlKey || e.metaKey || e.altKey) return false;
    if (e.code === "KeyZ") { ui.oct = Math.max(1, ui.oct - 1); toast(`אוקטבה ${ui.oct}`); return true; }
    if (e.code === "KeyX") { ui.oct = Math.min(7, ui.oct + 1); toast(`אוקטבה ${ui.oct}`); return true; }
    const k = KEYMAP[e.code]; if (k === undefined) return false;
    if (!e.repeat && !down.has(e.code)) { down.add(e.code); noteOn(12 * (ui.oct + 1) + k); }
    return true;
  }
  function renderKeys() {
    const t = playTrack(), lo = 12 * (ui.oct + 1) - 12, hi = lo + 36; let w = "", b = "", x = 0;
    const isB = (n) => [1, 3, 6, 8, 10].includes(n % 12);
    for (let n = lo; n < hi; n++) if (!isB(n)) { w += `<div class="wk" data-n="${n}">${n % 12 === 0 ? "C" + (Math.floor(n / 12) - 1) : ""}</div>`; x++; }
    let i = 0; for (let n = lo; n < hi; n++) { if (!isB(n)) { i++; continue; } b += `<div class="bk" data-n="${n}" style="left:calc(${(i / x) * 100}% - 1.2%)"></div>`; }
    $("#p-keys").innerHTML = `<div class="keysw"><div class="prtools"><b>${t ? esc(t.name) : "בחר ערוץ של כלי וירטואלי"}</b><span class="lbl">לוחצים על הקלידים או מנגנים במקלדת המחשב (כפתור המקלדת בשורת השליטה). כדי להקליט: R על הערוץ, ואז הקלטה.</span><span style="flex:1"></span><button class="btn" id="k-dn">אוקטבה −</button><button class="btn" id="k-up">אוקטבה +</button></div><div class="kb">${w}${b}</div></div>`;
    $$("#p-keys [data-n]").forEach((k) => { const n = +k.dataset.n; k.onpointerdown = (e) => { k.setPointerCapture(e.pointerId); noteOn(n, 96); }; k.onpointerup = k.onpointercancel = () => noteOff(n); });
    $("#k-dn").onclick = () => { ui.oct = Math.max(1, ui.oct - 1); renderKeys(); }; $("#k-up").onclick = () => { ui.oct = Math.min(7, ui.oct + 1); renderKeys(); };
  }
  $("#st-midi").onclick = async () => {
    if (!navigator.requestMIDIAccess) { $("#st-midi-s").textContent = "הדפדפן הזה לא תומך במקלדות MIDI. כרום או אדג' כן"; return; }
    try {
      midiIn = await navigator.requestMIDIAccess();
      const hook = () => { let n = 0; for (const inp of midiIn.inputs.values()) { n++; inp.onmidimessage = (m) => { const [st, d1, d2] = m.data, c = st & 0xf0; if (c === 0x90 && d2 > 0) noteOn(d1, d2); else if (c === 0x80 || (c === 0x90 && d2 === 0)) noteOff(d1); }; } $("#st-midi-s").textContent = n ? `מחוברות ${n} מקלדות` : "לא נמצאה מקלדת. מחברים בכבל ולוחצים שוב"; };
      hook(); midiIn.onstatechange = hook;
    } catch { $("#st-midi-s").textContent = "לא התקבלה הרשאה למקלדת"; }
  };

  // ---------------- views ----------------
  function toggleView(k, state) { const a = $("#app"), cls = "no-" + k, on = state === undefined ? a.classList.contains(cls) : state; a.classList.toggle(cls, !on); $$(`.vbtn[data-view="${k}"]`).forEach((b) => b.classList.toggle("on", on)); setTimeout(() => { renderRuler(); renderTracks(); }, 20); }
  $$(".vbtn").forEach((b) => (b.onclick = () => toggleView(b.dataset.view)));

  // ---------------- undo / keys ----------------
  async function undo() { const h = (LAB.call("history_list", {}) || [])[0]; const r = await call("undo", {}); refresh(); if (r.status === "done") toast("בוטל: " + (h ? h.label : "הפעולה האחרונה")); }
  async function redo() { const r = await call("redo", {}); refresh(); if (r.status === "done") toast("הפעולה בוצעה שוב"); }
  $("#undo").onclick = undo; $("#redo").onclick = redo;
  $("#projname").onclick = openProjects;
  $("#setbtn").onclick = openSettings;
  $("#helpbtn").onclick = () => $("#hp").classList.add("on");
  const HELP = [["רווח", "ניגון / עצירה (גם עוצר הקלטה)"], ["R", "הקלטה"], ["Enter", "חזרה להתחלה"], ["C", "לולאה"], ["K", "מטרונום"], [", / .", "תיבה אחורה / קדימה"], ["M / S", "השתקה / סולו לערוץ הנבחר"], ["↑ / ↓", "מעבר בין ערוצים"], ["X", "מיקסר"], ["E", "עורך תווים"], ["I", "מפקח"], ["Y", "ספרייה"], ["Ctrl+Alt+N", "ערוץ חדש"], ["Ctrl+Z / Ctrl+Shift+Z", "ביטול / ביצוע שוב"], ["Ctrl+D", "שכפול קטע"], ["Delete", "מחיקת הקטע הנבחר"], ["Ctrl + גלגלת", "הגדלה / הקטנה"], ["Esc", "סגירת חלון"], ["?", "החלון הזה"]];
  $("#helpgrid").innerHTML = HELP.map(([k, d]) => `<div><span>${d}</span><span class="kbd" style="font-size:12px;color:var(--gold)">${k}</span></div>`).join("");
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input,select,textarea")) { if (e.key === "Escape") e.target.blur(); return; }
    const k = e.key, low = k.toLowerCase(), code = e.code;
    if (k === "Escape") { closePop(); $$(".menu").forEach((x) => x.classList.remove("open")); if ($("#ask").classList.contains("on")) return $("#ask-no").click(); $$(".scrim").forEach((s) => s.classList.remove("on")); return; }
    if (k === "Enter" && $("#nt").classList.contains("on")) { e.preventDefault(); return $("#nt-ok").click(); }
    if (k === "Enter" && $("#ask").classList.contains("on")) { e.preventDefault(); return $("#ask-ok").click(); }
    if ($$(".scrim.on").length) return;
    if (keysDown(e)) { e.preventDefault(); return; }
    if ((e.ctrlKey || e.metaKey) && code === "KeyK") { e.preventDefault(); $("#b-keys").click(); return; }
    if (e.altKey && code === "KeyZ") { e.preventDefault(); return openHistory(); }
    if (e.altKey && code === "KeyN") { e.preventDefault(); return newProject(); }
    if (code === "Space") { e.preventDefault(); const p = LAB.call("pulse", {}); (p.playing || p.recording.active ? stopAll : play)(); return; }
    if (e.ctrlKey || e.metaKey) {
      if (code === "KeyZ") { e.preventDefault(); return e.shiftKey ? redo() : undo(); }
      if (code === "KeyY") { e.preventDefault(); return redo(); }
      if (code === "KeyN" && e.altKey) { e.preventDefault(); return openNT(); }

      if (code === "KeyO") { e.preventDefault(); return openProjects(); }
      if (code === "KeyS") { e.preventDefault(); return saveVersion(); }
      if (code === "KeyE") { e.preventDefault(); return exportSong(false); }
      if (code === "KeyI") { e.preventDefault(); return $("#f-audio").click(); }
      if (code === "KeyD") { e.preventDefault(); return dupRegion(); }
      if (code === "KeyC") { e.preventDefault(); return copyRegion(); }
      if (code === "KeyV") { e.preventDefault(); return pasteRegion(); }

      return;
    }
    const t = selTrack();
    const map = { KeyR: rec, Enter: () => $("#b-start").click(), NumpadEnter: () => $("#b-start").click(), KeyC: () => $("#b-cyc").click(), KeyK: () => $("#b-metro").click(), Comma: () => $("#b-rew").click(), Period: () => $("#b-ff").click(),
      KeyX: () => btab("mixer"), KeyE: () => btab("piano"), KeyI: () => toggleView("insp"), KeyY: () => toggleView("lib"), KeyQ: () => quantizeSel(), KeyT: () => splitAtPlayhead(),
      KeyM: () => t && act("set_mute", { track_id: t.id, on: String(!t.mute) }), KeyS: () => t && act("set_solo", { track_id: t.id, on: String(!t.solo) }),
      ArrowUp: () => { const i = tracks().findIndex((x) => x.id === ui.sel); if (i > 0) select(tracks()[i - 1].id); }, ArrowDown: () => { const i = tracks().findIndex((x) => x.id === ui.sel); if (i < tracks().length - 1) select(tracks()[i + 1].id); },
      Delete: () => (ui.bottom === "piano" && ui.pr.remove && ui.pr.remove()) || delRegion(), Backspace: () => (ui.bottom === "piano" && ui.pr.remove && ui.pr.remove()) || delRegion(), Equal: () => zoomBy(1.25), Minus: () => zoomBy(0.8), NumpadAdd: () => zoomBy(1.25), NumpadSubtract: () => zoomBy(0.8) };
    const f = map[code] || (k === "?" ? () => $("#hp").classList.add("on") : null);
    if (f) { e.preventDefault(); f(); }
    void low;
  });

  document.addEventListener("change", (e) => { if (e.target.matches("select, input[type=range]") && !e.target.closest(".dlg")) e.target.blur(); }, true);
  document.addEventListener("pointerup", (e) => { if (e.target.matches && e.target.matches("input[type=range]")) setTimeout(() => e.target.blur(), 0); }, true);
  // double-click on any fader puts it back to 0 dB
  document.addEventListener("dblclick", (e) => {
    const f = e.target.closest && e.target.closest("input.vfader"); if (!f) return;
    const sid = f.closest("[data-id]") && f.closest("[data-id]").dataset.id, tid = f.id === "in-vol" ? ui.sel : sid;
    if (f.id === "mx-master") return act("set_master_gain", { db: 0 }, "עוצמה ראשית 0 dB");
    if (tid) act("set_gain", { track_id: tid, db: 0 }, "עוצמה 0 dB");
  });

  // ---------------- render ----------------
  function renderAll() {
    if (S && S.project) { $("#projname").textContent = S.project.name; if (ui.sel && !tracks().some((t) => t.id === ui.sel)) ui.sel = null; if (!ui.sel && tracks().length) ui.sel = tracks()[0].id; }
    renderRuler(); renderTracks(); renderInsp(); renderBottom(); setTransportUi();
  }
  // ---------------- start ----------------
  buildMenus();
  S = LAB.call("state", {});
  if (!S.project) {
    const list = LAB.call("list_projects", {}) || [];
    const r = list.length ? LAB.call("open_project", { id: list[0].id }) : LAB.call("new_project", { name: "שיר חדש" });
    Promise.resolve(r).then(() => { refresh(); window.__HUM_READY__ = true; });
  } else { renderAll(); window.__HUM_READY__ = true; }
  renderLib(); btab(ui.bottom);
  window.addEventListener("resize", () => { renderRuler(); renderTracks(); });
  setInterval(() => { if (!pulseTimer) { const p = LAB.call("pulse", {}); if (p.revision !== lastRev || p.playing || p.recording.active) startPulse(); } }, 400);
  window.__HUM_STUDIO__ = { ui, refresh, call };
})();
