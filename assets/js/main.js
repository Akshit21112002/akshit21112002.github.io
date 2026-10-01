/* =========================================================
   Akshit Singh — site script
   Background art: one scientific / AI idea per page, drawn as points.
     latent        About          A slowly turning vision-language embedding space
     diffusionLoop (spare)        Diffusion model denoising noise into science objects
     terrain       (spare)        LiDAR scan of drifting ridges
     landscape     Publications   Gradient descent on a loss landscape
     network       Journey        Forward and backward passes in a neural net
     interference  CV             Two-source wave interference
   Pick the scene with data-scene="..." on <body>.
   ========================================================= */
(() => {
  "use strict";

  const body = document.body;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isLocal = location.protocol === "file:" || /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

  /* ---------------------------------------------------------
     RENDERER (shared by all scenes)
     --------------------------------------------------------- */
  const canvas = document.getElementById("bg-art");
  const ctx = canvas.getContext("2d");
  const AMBER = [255, 181, 71];
  const BLUE = [138, 178, 255];
  const pack = (r, g, b) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  let W, H, scale, gap, cols, rows, maxR, img, buf, disks, rand;
  const mouse = { x: -1e4, y: -1e4, tx: -1e4, ty: -1e4, on: false };
  let start = performance.now();
  let exitStart = null;
  let lastFrame = 0;
  let simTime = 0;
  let stepAcc = 0;
  let paused = reduceMotion;
  let needsDraw = true;
  const STEP = 1000 / 30;

  // per-frame transform state used by dot()
  const F = { sweeping: false, sweepR: 0, ringW: 0, ox: 0, oy: 0, ep: 0, epY: 0, push: false, MR: 0, MR2: 0 };

  function buildDisks(max) {
    disks = [];
    for (let r = 0; r <= max; r++) {
      const pts = [];
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++)
          if (dx * dx + dy * dy <= r * r + r * 0.6) pts.push(dx, dy);
      disks.push(pts);
    }
  }

  // Draw one point. Handles the load sweep, cursor push and exit scatter for every scene.
  function dot(cx, cy, rad, r, g, b, i) {
    if (F.sweeping) {
      const dx = cx - F.ox, dy = cy - F.oy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > F.sweepR) return;
      const edge = F.sweepR - d;
      if (edge < F.ringW) {
        const f = 1 - edge / F.ringW;
        r += (AMBER[0] - r) * f; g += (AMBER[1] - g) * f; b += (AMBER[2] - b) * f;
        rad = Math.max(rad, 1 + f * maxR * 0.6);
      }
    }
    if (F.push) {
      const mdx = cx - mouse.x, mdy = cy - mouse.y;
      const md2 = mdx * mdx + mdy * mdy;
      if (md2 < F.MR2 && md2 > 0.01) {
        const md = Math.sqrt(md2);
        const f = 1 - md / F.MR;
        const p = f * f * 26 * scale;
        cx += (mdx / md) * p; cy += (mdy / md) * p;
      }
    }
    if (F.ep > 0) {
      const q = rand[i % rand.length];
      cy += F.epY * (40 + q * 180) * scale;
      cx += (q - 0.5) * F.epY * 40 * scale;
      rad *= 1 - F.ep;
    }
    const ri = rad < 0 ? 0 : rad > disks.length - 1 ? disks.length - 1 : Math.round(rad);
    const disk = disks[ri];
    const col = pack(clamp(r, 0, 255) | 0, clamp(g, 0, 255) | 0, clamp(b, 0, 255) | 0);
    const px = cx | 0, py = cy | 0;
    if (px - ri >= 0 && py - ri >= 0 && px + ri < W && py + ri < H) {
      for (let d = 0; d < disk.length; d += 2) buf[(py + disk[d + 1]) * W + px + disk[d]] = col;
    } else {
      for (let d = 0; d < disk.length; d += 2) {
        const x = px + disk[d], y = py + disk[d + 1];
        if (x >= 0 && y >= 0 && x < W && y < H) buf[y * W + x] = col;
      }
    }
  }

  // Point sized by brightness, used by the field-style scenes
  function fieldDot(c, rr, R, G, B) {
    const lum = (R * 0.299 + G * 0.587 + B * 0.114) / 255;
    if (lum < 0.035) return;
    dot(c * gap + gap * 0.5, rr * gap + gap * 0.5, 0.25 + lum * maxR * 1.05, R, G, B, rr * cols + c);
  }

  function faintGrid() {
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) dot(c * gap + gap * 0.5, r * gap + gap * 0.5, 0, 46, 48, 92, r * cols + c);
  }

  const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

  /* ---------------------------------------------------------
     SCENES
     --------------------------------------------------------- */
  const scenes = {};

  // 1. LiDAR scan of drifting ridges --------------------------
  scenes.terrain = {
    title: "LiDAR scan",
    text: "Terrain sampled as points, the way a neural distance field learns a surface.",
    push: true,
    draw(t) {
      const ridges = [
        { base: 0.5, amp: 0.06, col: [128, 96, 160], k: 0.0 },
        { base: 0.63, amp: 0.07, col: [74, 58, 118], k: 1.9 },
        { base: 0.77, amp: 0.08, col: [36, 32, 74], k: 3.7 },
      ];
      const ts = t * 0.05, sunU = 0.7, sunV = 0.4, asp = cols / rows;
      for (let c = 0; c < cols; c++) {
        const u = c / cols;
        const ry = ridges.map((R) => R.base + R.amp * (Math.sin(u * 5 + ts + R.k) * 0.6 + Math.sin(u * 11.5 + R.k * 1.7 - ts * 0.7) * 0.3 + Math.sin(u * 27 + R.k) * 0.1));
        for (let r = 0; r < rows; r++) {
          const v = r / rows;
          let R, G, B;
          if (v > ry[2]) { const f = 1 - (v - ry[2]) * 0.6; R = 36 * f; G = 32 * f; B = 74 * f; }
          else if (v > ry[1]) [R, G, B] = ridges[1].col;
          else if (v > ry[0]) [R, G, B] = ridges[0].col;
          else {
            const h = Math.pow(v / ry[0], 1.8);
            R = 30 + h * 210; G = 26 + h * 100; B = 72 + h * 30;
            const du = (u - sunU) * asp, dv = v - sunV, sd = Math.sqrt(du * du + dv * dv);
            if (sd < 0.07) { R = 255; G = 214; B = 160; }
            else if (sd < 0.25) { const g = (1 - (sd - 0.07) / 0.18) * 0.5; R += (255 - R) * g; G += (190 - G) * g; B += (140 - B) * g; }
            const s = rand[r * cols + c];
            if (v < 0.35 && s > 0.985) { const tw = 150 + 100 * Math.sin(t * 2 + s * 900); R = G = B = tw; }
          }
          fieldDot(c, r, R * 0.92 + 6, G * 0.92 + 6, B * 0.92 + 8);
        }
      }
    },
  };

  // 0. Latent space: a calm, slowly turning embedding space --------------------
  scenes.latent = {
    title: "Latent space",
    text: "How a vision-language model organizes meaning: images (amber) and captions (blue) about the same thing sit close together.",
    push: true,
    concepts: ["mountains at dusk", "a dog on the beach", "city lights at night", "a bowl of ramen", "a spiral galaxy",
               "a chest x-ray", "handwritten digits", "a red vintage car", "sheet music"],
    init() {
      const gauss = () => { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); };
      const K = this.concepts.length;
      const n = clamp(Math.round((W * H) / (scale * scale) / 800), 900, 1700);
      // cluster centres spread on a fibonacci sphere
      this.centers = [];
      for (let k = 0; k < K; k++) {
        const y = 1 - (2 * (k + 0.5)) / K, r = Math.sqrt(1 - y * y), th = k * 2.39996;
        const d = 0.72 + Math.random() * 0.12;
        this.centers.push([Math.cos(th) * r * d, y * d * 0.85, Math.sin(th) * r * d]);
      }
      this.n = n;
      this.p = new Float32Array(n * 3);
      this.kind = new Uint8Array(n);        // 0 image, 1 text, 2 background
      this.ph = new Float32Array(n);
      const per = [];
      for (let i = 0; i < n; i++) {
        this.ph[i] = Math.random() * Math.PI * 2;
        if (i % 12 === 11) {
          const v = [gauss(), gauss(), gauss()], m = Math.hypot(...v) || 1, rr = 1.15 * Math.cbrt(Math.random());
          this.p.set([(v[0] / m) * rr, (v[1] / m) * rr, (v[2] / m) * rr], i * 3);
          this.kind[i] = 2;
          continue;
        }
        const k = i % K, c = this.centers[k];
        const sx = 0.09 + (k % 3) * 0.02;
        this.p.set([c[0] + gauss() * sx, c[1] + gauss() * sx * 0.8, c[2] + gauss() * sx], i * 3);
        this.kind[i] = (i / K) & 1;
        (per[k] = per[k] || [[], []])[this.kind[i]].push(i);
      }
      // a few matched image-caption pairs per concept, drawn as faint links
      this.pairs = [];
      per.forEach((g) => { for (let j = 0; j < 3 && j < g[0].length && j < g[1].length; j++) this.pairs.push([g[0][j], g[1][j]]); });
      this.proj = new Float32Array(n * 3);
      this.cproj = this.centers.map(() => [0, 0, 1]);
    },
    project(x, y, z, o) {
      let X = x * o.cy + z * o.sy, Z = -x * o.sy + z * o.cy;
      const Y = y * o.cp - Z * o.sp; Z = y * o.sp + Z * o.cp;
      const s = 3 / (3 + Z);
      return [o.cx0 + X * o.R * s, o.cy0 + Y * o.R * s, s, Z];
    },
    draw(t) {
      faintGrid();
      const wide = W > 900 * scale;
      const yaw = t * 0.035, pitch = 0.32 + Math.sin(t * 0.021) * 0.06;
      const o = {
        cy: Math.cos(yaw), sy: Math.sin(yaw), cp: Math.cos(pitch), sp: Math.sin(pitch),
        cx0: W * (wide ? 0.75 : 0.5), cy0: H * (wide ? 0.45 : 0.33), R: Math.min(W * (wide ? 0.21 : 0.4), H * 0.36),
      };
      this.wide = wide;
      const P = this.p, pr = this.proj;
      for (let i = 0; i < this.n; i++) {
        const k = i * 3, w = 0.012 * Math.sin(t * 0.25 + this.ph[i]);
        const q = this.project(P[k] + w, P[k + 1] + w * 0.7, P[k + 2] - w, o);
        pr[k] = q[0]; pr[k + 1] = q[1]; pr[k + 2] = q[2];
      }
      // links between matched pairs
      for (const [a, b] of this.pairs) {
        const ax = pr[a * 3], ay = pr[a * 3 + 1], bx = pr[b * 3], by = pr[b * 3 + 1];
        const steps = Math.max(2, Math.floor(Math.hypot(bx - ax, by - ay) / (gap * 0.9)));
        for (let j = 1; j < steps; j++) { const f = j / steps; dot(ax + (bx - ax) * f, ay + (by - ay) * f, 0, 120, 122, 178, a + j); }
      }
      for (let i = 0; i < this.n; i++) {
        const k = i * 3, s = pr[k + 2], kd = this.kind[i];
        const depth = clamp(0.35 + (s - 0.75) * 1.4, 0.3, 1.05);
        const col = kd === 0 ? AMBER : kd === 1 ? BLUE : [150, 150, 196];
        const size = kd === 2 ? 0.25 : 0.55;
        dot(pr[k], pr[k + 1], maxR * size * s * 1.1, col[0] * depth, col[1] * depth, col[2] * depth, i);
      }
      this.centers.forEach((c, k) => { this.cproj[k] = this.project(c[0], c[1], c[2], o); });
    },
    overlay(g) {
      if (!this.wide || F.sweeping) return;
      const fade = 1 - F.ep;
      g.textAlign = "left";
      g.textBaseline = "middle";
      this.cproj.forEach(([x, y, s], k) => {
        const a = clamp((s - 0.97) * 3, 0, 1) * 0.85 * fade;
        if (a < 0.03 || x < W * 0.52) return;
        const lx = x + 34 * scale * s, ly = y - 26 * scale * s;
        g.font = "italic " + Math.round(15 * scale * s) + 'px "Times New Roman", Times, serif';
        g.lineJoin = "round";
        g.lineWidth = 5 * scale;
        g.strokeStyle = "rgba(18, 20, 43, " + (a * 0.9).toFixed(3) + ")";
        g.strokeText(this.concepts[k], lx, ly);
        g.fillStyle = "rgba(226, 224, 244, " + a.toFixed(3) + ")";
        g.fillText(this.concepts[k], lx, ly);
      });
    },
  };
  // The About page used to say data-scene="diffusion"; it now shows the calm latent space.
  const SCENE_ALIASES = { diffusion: "latent" };

  // Spare: diffusion loop (denoising noise into a galaxy, attractor and DNA) --
  scenes.diffusionLoop = {
    title: "Diffusion",
    text: "A generative model denoising pure noise into a galaxy, a strange attractor and DNA, then letting each dissolve back.",
    push: true,
    shapes: ["spiral galaxy", "Lorenz attractor", "DNA double helix"],
    CYCLE: 13,
    init() {
      const n = clamp(Math.round((W * H) / (scale * scale) / 520), 1200, 2800);
      this.n = n;
      const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      this.gauss = gauss;
      this.noise = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) this.noise[i] = gauss();
      const T = [], C = [];

      // spiral galaxy (two arms + bulge)
      let p = new Float32Array(n * 3), c = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        let x, y, z, col;
        if (i % 7 === 0) {
          x = gauss() * 0.12; y = gauss() * 0.07; z = gauss() * 0.12; col = [255, 206, 150];
        } else {
          const r = 0.08 + 0.92 * Math.pow(Math.random(), 0.75);
          const th = (i % 2) * Math.PI + r * 5.6 + gauss() * 0.3 * (1.1 - r * 0.5);
          x = r * Math.cos(th); z = r * Math.sin(th); y = gauss() * 0.035;
          col = Math.random() < 0.06 ? [240, 238, 250] : lerp3([255, 190, 120], BLUE, Math.min(1, r * 1.3));
        }
        p.set([x, y, z], i * 3); c.set(col, i * 3);
      }
      T.push(p); C.push(c);

      // Lorenz attractor (sigma 10, rho 28, beta 8/3)
      p = new Float32Array(n * 3); c = new Float32Array(n * 3);
      let lx = 0.1, ly = 0, lz = 0;
      const dt = 0.005, lstep = () => { const dx = 10 * (ly - lx), dy = lx * (28 - lz) - ly, dz = lx * ly - (8 / 3) * lz; lx += dx * dt; ly += dy * dt; lz += dz * dt; };
      for (let k = 0; k < 2000; k++) lstep();
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 4; k++) lstep();
        p.set([lx / 27, -(lz - 24) / 27, ly / 27], i * 3);
        c.set(lerp3(AMBER, BLUE, clamp(0.5 + lx / 30, 0, 1)), i * 3);
      }
      T.push(p); C.push(c);

      // DNA double helix with base-pair rungs
      p = new Float32Array(n * 3); c = new Float32Array(n * 3);
      const turns = 2.6 * Math.PI, rad = 0.36, rungs = 26;
      for (let i = 0; i < n; i++) {
        if (i % 10 < 7) {
          const s = Math.random() * 2 - 1, strand = i % 2, ph = s * turns + strand * Math.PI;
          p.set([Math.cos(ph) * rad + gauss() * 0.012, s * 0.95, Math.sin(ph) * rad + gauss() * 0.012], i * 3);
          c.set(strand ? BLUE : AMBER, i * 3);
        } else {
          const s = ((Math.floor(Math.random() * rungs) + 0.5) / rungs) * 2 - 1, ph = s * turns, u = Math.random();
          const ax = Math.cos(ph) * rad, az = Math.sin(ph) * rad;
          p.set([ax + (-2 * ax) * u, s * 0.95, az + (-2 * az) * u], i * 3);
          c.set([170, 168, 205], i * 3);
        }
      }
      T.push(p); C.push(c);
      this.targets = T; this.colors = C;
      this.status = "";
    },
    step() {
      // Brownian jitter: an Ornstein-Uhlenbeck walk keeps the noise alive
      const N = this.noise, a = 0.985, b = Math.sqrt(1 - a * a);
      for (let i = 0; i < N.length; i++) N[i] = N[i] * a + b * ((Math.random() + Math.random() + Math.random()) * 2 - 3) * 1.15;
    },
    phase(t) {
      const tt = reduceMotion ? 7 : t + 0.2;
      const cyc = Math.floor(tt / this.CYCLE), u = tt - cyc * this.CYCLE;
      const sm = (x) => x * x * (3 - 2 * x);
      let ab, label;
      if (u < 0.6) { ab = 0; label = "Pure noise, step 1000"; }
      else if (u < 5.6) { const q = (u - 0.6) / 5; ab = sm(q); label = "Denoising, step " + Math.round((1 - q) * 1000); }
      else if (u < 10.4) { ab = 1; label = "Sample: " + this.shapes[cyc % 3]; }
      else { const q = (u - 10.4) / 2.6; ab = 1 - sm(q); label = "Adding noise, step " + Math.round(q * 1000); }
      return { ab, label, shape: cyc % 3, u };
    },
    draw(t) {
      faintGrid();
      const { ab, label, shape } = this.phase(t);
      this.status = label;
      const P = this.targets[shape], Cc = this.colors[shape], N = this.noise;
      const sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab) * 1.25;
      const wide = W > 900 * scale;
      const cx0 = W * (wide ? 0.74 : 0.5), cy0 = H * (wide ? 0.44 : 0.33);
      const R = Math.min(W * (wide ? 0.2 : 0.4), H * 0.34);
      // shape-specific orientation
      const yaw = t * (shape === 0 ? 0.0 : 0.22);
      const spin = shape === 0 ? t * 0.12 : 0;
      const pitch = shape === 0 ? 1.12 : 0.22;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const cs = Math.cos(spin), ss = Math.sin(spin);
      const dimCol = [120, 120, 172];
      for (let i = 0; i < this.n; i++) {
        const k = i * 3;
        let x = P[k], y = P[k + 1], z = P[k + 2];
        if (spin) { const nx = x * cs - z * ss; z = x * ss + z * cs; x = nx; }
        x = sa * x + sn * N[k]; y = sa * y + sn * N[k + 1]; z = sa * z + sn * N[k + 2];
        let X = x * cy + z * sy, Z = -x * sy + z * cy;
        const Y = y * cp - Z * sp; Z = y * sp + Z * cp;
        if (Z < -2.4) continue;
        const s = clamp(3 / (3 + Z), 0.45, 1.6);
        const px = cx0 + X * R * s, py = cy0 + Y * R * s;
        const m = ab * ab;
        const r = dimCol[0] + (Cc[k] - dimCol[0]) * m, g = dimCol[1] + (Cc[k + 1] - dimCol[1]) * m, bb = dimCol[2] + (Cc[k + 2] - dimCol[2]) * m;
        const depth = clamp(0.8 - Z * 0.25, 0.4, 1.1);
        dot(px, py, maxR * (0.35 + 0.45 * ab) * s, r * depth, g * depth, bb * depth, i);
      }
    },
  };

  // 2. Gradient descent on a loss landscape -------------------
  scenes.landscape = {
    title: "Gradient descent",
    text: "Optimizers settling into the valleys of a loss landscape. Amber is noisy SGD; blue uses momentum.",
    push: true,
    init() {
      const asp = cols / rows;
      const wide = W > 900 * scale;
      const wells = [
        [0.72, 0.36, 0.9, 0.12], [0.86, 0.7, 0.7, 0.09], [0.56, 0.74, 0.55, 0.1],
        [wide ? 0.4 : 0.3, 0.24, 0.45, 0.13], [0.92, 0.2, 0.4, 0.07], [0.18, 0.7, 0.35, 0.12],
      ];
      const L = new Float32Array(cols * rows);
      let lo = Infinity, hi = -Infinity;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const u = c / cols, v = r / rows;
          let l = 0.35 * (((u - 0.68) * asp) ** 2 + (v - 0.5) ** 2);
          for (const [wu, wv, a, s] of wells) {
            const du = (u - wu) * asp, dv = v - wv;
            l -= a * Math.exp(-(du * du + dv * dv) / (2 * s * s));
          }
          l += 0.04 * Math.sin(u * asp * 14) * Math.cos(v * 11);
          L[r * cols + c] = l;
          if (l < lo) lo = l; if (l > hi) hi = l;
        }
      for (let i = 0; i < L.length; i++) L[i] = (L[i] - lo) / (hi - lo);
      this.L = L;
      const n = clamp(Math.round((cols * rows) / 650), 14, 44);
      this.ps = Array.from({ length: n }, (_, i) => this.spawn({ kind: i % 2 }));
    },
    spawn(p) {
      p.x = 2 + Math.random() * (cols - 4); p.y = 2 + Math.random() * (rows - 4);
      p.vx = 0; p.vy = 0; p.age = 0; p.still = 0; p.trail = [];
      return p;
    },
    step() {
      const L = this.L;
      for (const p of this.ps) {
        const ix = clamp(Math.round(p.x), 1, cols - 2), iy = clamp(Math.round(p.y), 1, rows - 2);
        const gx = (L[iy * cols + ix + 1] - L[iy * cols + ix - 1]) / 2;
        const gy = (L[(iy + 1) * cols + ix] - L[(iy - 1) * cols + ix]) / 2;
        const mom = p.kind ? 0.93 : 0.8, noise = p.kind ? 0.02 : 0.12, lr = p.kind ? 1.2 : 2.6;
        p.vx = mom * p.vx - lr * gx + (Math.random() - 0.5) * noise;
        p.vy = mom * p.vy - lr * gy + (Math.random() - 0.5) * noise;
        const sp = Math.hypot(p.vx, p.vy);
        if (sp > 1.2) { p.vx *= 1.2 / sp; p.vy *= 1.2 / sp; }
        p.x = clamp(p.x + p.vx, 1, cols - 2); p.y = clamp(p.y + p.vy, 1, rows - 2);
        p.trail.push(p.x, p.y);
        if (p.trail.length > 90) p.trail.splice(0, 2);
        p.age++;
        p.still = sp < 0.05 ? p.still + 1 : 0;
        if (p.still > 60 || p.age > 30 * 16) this.spawn(p);
      }
    },
    draw(t) {
      const L = this.L, low = [255, 170, 92], high = [52, 50, 108];
      const phase = t * 0.12;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const l = L[r * cols + c];
          let band = (l * 16 - phase) % 1; if (band < 0) band += 1;
          const contour = band < 0.13;
          const [R, G, B] = lerp3(low, high, Math.pow(l, 0.7));
          const k = contour ? 1.15 : 0.32;
          const rad = (contour ? 0.45 : 0.12) * maxR + (1 - l) * 0.45 * maxR;
          dot(c * gap + gap * 0.5, r * gap + gap * 0.5, rad, R * k, G * k, B * k, r * cols + c);
        }
      for (const p of this.ps) {
        const col = p.kind ? BLUE : AMBER;
        const tr = p.trail, n = tr.length / 2;
        for (let j = 0; j < n; j += 2) {
          const f = j / n;
          dot(tr[j * 2] * gap + gap * 0.5, tr[j * 2 + 1] * gap + gap * 0.5, 0.4 + f * maxR * 0.6, col[0] * f, col[1] * f, col[2] * f, j);
        }
        dot(p.x * gap + gap * 0.5, p.y * gap + gap * 0.5, maxR * 1.6, col[0], col[1], col[2], 7);
      }
    },
  };

  // 3. Forward and backward passes in a neural network --------
  scenes.network = {
    title: "Forward and backward passes",
    text: "Signals flow forward through a neural network; errors flow back to teach it.",
    push: true,
    init() {
      const wide = W > 900 * scale;
      const sizes = wide ? [3, 6, 8, 8, 6, 4, 2] : [3, 5, 6, 5, 2];
      const x0 = W * (wide ? 0.4 : 0.1), x1 = W * (wide ? 0.94 : 0.9);
      const y0 = H * 0.14, y1 = H * 0.86;
      this.nodes = []; this.edges = []; this.outE = []; this.inE = []; this.layerOf = [];
      const layers = sizes.map((n, li) => {
        const x = x0 + ((x1 - x0) * li) / (sizes.length - 1);
        const span = Math.min(y1 - y0, n * 95 * scale);
        const yc = (y0 + y1) / 2 + (wide ? 0 : -H * 0.05);
        return Array.from({ length: n }, (_, k) => {
          const id = this.nodes.length;
          this.nodes.push({ x, y: n === 1 ? yc : yc - span / 2 + (span * k) / (n - 1), act: 0, back: false });
          this.outE.push([]); this.inE.push([]); this.layerOf.push(li);
          return id;
        });
      });
      this.nLayers = sizes.length;
      for (let li = 0; li < layers.length - 1; li++)
        for (const a of layers[li])
          for (const b of layers[li + 1]) {
            const A = this.nodes[a], B = this.nodes[b];
            const len = Math.hypot(B.x - A.x, B.y - A.y);
            const n = Math.max(2, Math.floor(len / (gap * 1.15)));
            const e = { a, b, w: Math.random() ** 2, n };
            this.outE[a].push(this.edges.length); this.inE[b].push(this.edges.length);
            this.edges.push(e);
          }
      this.inputs = layers[0]; this.outputs = layers[layers.length - 1];
      this.pulses = []; this.clock = 0;
    },
    pick(list) {
      let s = 0; for (const e of list) s += this.edges[e].w + 0.05;
      let q = Math.random() * s;
      for (const e of list) { q -= this.edges[e].w + 0.05; if (q <= 0) return e; }
      return list[list.length - 1];
    },
    step() {
      this.clock++;
      if (this.clock % 9 === 0 && this.pulses.length < 50) {
        const a = this.inputs[(Math.random() * this.inputs.length) | 0];
        this.nodes[a].act = 1; this.nodes[a].back = false;
        this.pulses.push({ e: this.pick(this.outE[a]), p: 0, dir: 1 });
      }
      for (let i = this.pulses.length - 1; i >= 0; i--) {
        const P = this.pulses[i];
        P.p += 0.045;
        if (P.p < 1) continue;
        const e = this.edges[P.e];
        const node = P.dir > 0 ? e.b : e.a;
        const N = this.nodes[node];
        N.act = 1; N.back = P.dir < 0;
        this.pulses.splice(i, 1);
        if (P.dir > 0) {
          if (this.outE[node].length) this.pulses.push({ e: this.pick(this.outE[node]), p: 0, dir: 1 });
          else if (Math.random() < 0.45) this.pulses.push({ e: this.pick(this.inE[node]), p: 0, dir: -1 });
        } else if (this.inE[node].length) {
          this.pulses.push({ e: this.pick(this.inE[node]), p: 0, dir: -1 });
        }
      }
      for (const N of this.nodes) N.act *= 0.93;
    },
    draw() {
      faintGrid();
      const nodes = this.nodes;
      for (let ei = 0; ei < this.edges.length; ei++) {
        const e = this.edges[ei], A = nodes[e.a], B = nodes[e.b];
        const k = 50 + e.w * 120;
        for (let j = 1; j < e.n; j++) {
          const f = j / e.n;
          dot(A.x + (B.x - A.x) * f, A.y + (B.y - A.y) * f, e.w > 0.5 ? 1 : 0, k * 0.8, k * 0.82, k * 1.2, ei * 31 + j);
        }
      }
      for (const P of this.pulses) {
        const e = this.edges[P.e], A = nodes[e.a], B = nodes[e.b];
        const col = P.dir > 0 ? AMBER : BLUE;
        for (let s = 0; s < 5; s++) {
          let f = P.dir > 0 ? P.p - s * 0.035 : 1 - P.p + s * 0.035;
          if (f < 0 || f > 1) continue;
          const fade = 1 - s / 5;
          dot(A.x + (B.x - A.x) * f, A.y + (B.y - A.y) * f, maxR * (0.4 + fade * 0.8), col[0] * fade, col[1] * fade, col[2] * fade, s);
        }
      }
      for (let i = 0; i < nodes.length; i++) {
        const N = nodes[i];
        const [R, G, B] = lerp3([150, 150, 196], N.back ? BLUE : AMBER, N.act);
        dot(N.x, N.y, maxR * (1.7 + N.act * 0.9), R, G, B, i);
        dot(N.x, N.y, maxR * 0.9, 18, 20, 43, i);
        dot(N.x, N.y, maxR * (0.4 + N.act * 0.5), R, G, B, i);
      }
    },
  };

  // 4. Two-source wave interference --------------------------
  scenes.interference = {
    title: "Two-source interference",
    text: "Waves reinforce and cancel, as in Young\u2019s double-slit experiment. Move your cursor to add a third source.",
    push: false,
    draw(t) {
      const wide = W > 900 * scale;
      const cx0 = W * (wide ? 0.66 : 0.5), cy0 = H * 0.42;
      const sep = W * (wide ? 0.06 : 0.1), th = t * 0.06;
      const s1x = cx0 + Math.cos(th) * sep, s1y = cy0 + Math.sin(th) * sep;
      const s2x = cx0 - Math.cos(th) * sep, s2y = cy0 - Math.sin(th) * sep;
      const k = (2 * Math.PI) / (gap * 6.5), w = t * 1.6, fall = W * 0.7;
      const m = mouse.on && mouse.x > -1e3;
      const warm = [255, 176, 80], cool = [120, 150, 255];
      for (let r = 0; r < rows; r++) {
        const y = r * gap + gap * 0.5;
        for (let c = 0; c < cols; c++) {
          const x = c * gap + gap * 0.5;
          const r1 = Math.hypot(x - s1x, y - s1y), r2 = Math.hypot(x - s2x, y - s2y);
          let a = Math.sin(k * r1 - w) / (1 + r1 / fall) + Math.sin(k * r2 - w) / (1 + r2 / fall);
          let norm = 2;
          if (m) { const r3 = Math.hypot(x - mouse.x, y - mouse.y); a += 0.6 * Math.sin(k * r3 - w) / (1 + r3 / (fall * 0.5)); norm = 2.6; }
          a /= norm;
          const I = a * a;
          if (I < 0.012) continue;
          const col = a > 0 ? warm : cool, b = 0.35 + Math.sqrt(I) * 0.75;
          dot(x, y, 0.2 + Math.sqrt(I) * maxR * 1.05, col[0] * b, col[1] * b, col[2] * b, r * cols + c);
        }
      }
      dot(s1x, s1y, maxR * 1.4, 240, 238, 250, 1);
      dot(s2x, s2y, maxR * 1.4, 240, 238, 250, 2);
    },
  };

  const scene = scenes[SCENE_ALIASES[body.dataset.scene] || body.dataset.scene] || scenes.latent;

  function resize() {
    const cssW = window.innerWidth, cssH = window.innerHeight;
    scale = Math.min(1, 1600 / cssW);
    W = Math.max(1, Math.round(cssW * scale));
    H = Math.max(1, Math.round(cssH * scale));
    canvas.width = W; canvas.height = H;
    gap = Math.max(4, Math.round((cssW < 700 ? 7 : 9) * scale));
    cols = Math.ceil(W / gap) + 1;
    rows = Math.ceil(H / gap) + 1;
    maxR = Math.max(1, Math.floor(gap * 0.42));
    buildDisks(maxR * 3);
    img = ctx.createImageData(W, H);
    buf = new Uint32Array(img.data.buffer);
    rand = new Float32Array(cols * rows);
    for (let i = 0; i < rand.length; i++) rand[i] = Math.random();
    scene.init && scene.init();
    if (scene.step && reduceMotion) for (let i = 0; i < 90; i++) scene.step();
    needsDraw = true;
  }

  function frame(now) {
    buf.fill(0);
    F.ox = W * 0.5; F.oy = H * 1.02;
    F.sweeping = false;
    if (!reduceMotion) {
      const p = Math.min(1, (now - start) / 1800);
      if (p < 1) { F.sweeping = true; F.sweepR = (1 - Math.pow(1 - p, 3)) * Math.hypot(W * 0.5, H * 1.02) * 1.05; F.ringW = 34 * scale; }
    }
    F.ep = exitStart ? Math.min(1, (now - exitStart) / 450) : 0;
    F.epY = F.ep * F.ep;
    mouse.x += (mouse.tx - mouse.x) * 0.2;
    mouse.y += (mouse.ty - mouse.y) * 0.2;
    F.MR = 120 * scale; F.MR2 = F.MR * F.MR;
    F.push = scene.push && !reduceMotion && mouse.x > -1e3;
    scene.draw(simTime / 1000);
    ctx.putImageData(img, 0, 0);
    if (scene.overlay) scene.overlay(ctx);
    if (capStatus && scene.status !== undefined && capStatus.textContent !== scene.status) capStatus.textContent = scene.status;
  }

  function loop(now) {
    requestAnimationFrame(loop);
    const dt = Math.min(100, now - (lastFrame || now));
    const intro = !reduceMotion && now - start < 1900;
    const live = !paused || exitStart || intro;
    if (!live && !needsDraw) return;
    if (live && !intro && !exitStart && now - lastFrame < STEP - 2) return;
    lastFrame = now;
    if (!paused) {
      simTime += dt;
      if (scene.step) {
        stepAcc += dt;
        let n = 0;
        while (stepAcc >= STEP && n++ < 4) { scene.step(); stepAcc -= STEP; }
      }
    }
    needsDraw = false;
    frame(now);
  }

  window.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    mouse.tx = e.clientX * scale; mouse.ty = e.clientY * scale; mouse.on = true;
    if (mouse.x < -1e3) { mouse.x = mouse.tx; mouse.y = mouse.ty; }
    if (paused) needsDraw = true;
  }, { passive: true });
  document.documentElement.addEventListener("pointerleave", () => { mouse.tx = mouse.ty = mouse.x = mouse.y = -1e4; mouse.on = false; needsDraw = true; });

  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(resize, 120); });

  // Scene caption
  const capTitle = document.querySelector("[data-scene-title]");
  const capText = document.querySelector("[data-scene-text]");
  const capStatus = document.querySelector("[data-scene-status]");
  if (capTitle) capTitle.textContent = scene.title;
  if (capText) capText.textContent = scene.text;

  // Pause / play
  const pauseBtn = document.querySelector("[data-pause]");
  function setPaused(p) {
    paused = p;
    if (pauseBtn) {
      pauseBtn.setAttribute("aria-pressed", String(p));
      pauseBtn.querySelector("span").textContent = p ? "Play animation" : "Pause animation";
    }
    needsDraw = true;
  }
  if (pauseBtn) pauseBtn.addEventListener("click", () => setPaused(!paused));

  /* ---------------------------------------------------------
     PAGE TRANSITIONS
     --------------------------------------------------------- */
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (a.target && a.target !== "_self") return;
    if (a.hasAttribute("download")) return;
    const url = new URL(a.href, location.href);
    if (url.origin !== location.origin || !/\.html$|\/$/.test(url.pathname)) return;
    if (url.pathname === location.pathname && url.hash) return;
    e.preventDefault();
    body.classList.add("leaving");
    if (reduceMotion) { location.href = url.href; return; }
    exitStart = performance.now();
    setTimeout(() => { location.href = url.href; }, 460);
  });
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) { exitStart = null; body.classList.remove("leaving"); needsDraw = true; }
  });

  /* ---------------------------------------------------------
     PAGE INTERACTIONS
     --------------------------------------------------------- */
  const hero = document.querySelector(".hero");
  const nav = document.querySelector(".site-nav");
  const timeline = document.querySelector(".timeline");
  function updateTimeline() {
    if (!timeline) return;
    const r = timeline.getBoundingClientRect();
    const p = clamp((window.innerHeight * 0.6 - r.top) / r.height, 0, 1);
    timeline.style.setProperty("--progress", p.toFixed(3));
  }
  function onScroll() {
    const h = hero ? hero.offsetHeight : window.innerHeight * 0.6;
    const p = clamp(window.scrollY / (h * 0.75), 0, 1);
    body.style.setProperty("--dim", p.toFixed(3));
    body.classList.toggle("past-hero", p > 0.9);
    nav && nav.classList.toggle("is-scrolled", window.scrollY > 24);
    updateTimeline();
  }
  window.addEventListener("scroll", onScroll, { passive: true });

  // Publication filter
  const filterButtons = document.querySelectorAll("[data-filter]");
  const pubs = document.querySelectorAll("[data-topic]");
  filterButtons.forEach((btn) =>
    btn.addEventListener("click", () => {
      const f = btn.dataset.filter;
      filterButtons.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      let shown = 0;
      pubs.forEach((p) => {
        const on = f === "all" || p.dataset.topic.split(" ").includes(f);
        p.hidden = !on;
        if (on) shown++;
      });
      const count = document.querySelector("[data-pub-count]");
      if (count) count.textContent = shown === 1 ? "1 paper" : shown + " papers";
    })
  );

  // Expandable details
  document.querySelectorAll("[data-toggle]").forEach((btn) => {
    const panel = document.getElementById(btn.getAttribute("aria-controls"));
    if (!panel) return;
    btn.addEventListener("click", () => {
      const open = btn.getAttribute("aria-expanded") !== "true";
      btn.setAttribute("aria-expanded", String(open));
      panel.classList.toggle("is-open", open);
      btn.querySelector("[data-toggle-label]").textContent = open ? btn.dataset.close : btn.dataset.open;
    });
  });

  // Mobile nav
  const navToggle = document.querySelector(".nav-toggle");
  if (navToggle) navToggle.addEventListener("click", () => {
    const open = navToggle.getAttribute("aria-expanded") !== "true";
    navToggle.setAttribute("aria-expanded", String(open));
    nav.classList.toggle("is-open", open);
  });

  // Headshot: show initials if the photo is missing
  const shot = document.querySelector(".headshot img");
  if (shot) {
    const miss = () => {
      shot.parentElement.classList.add("is-missing");
      if (isLocal) shot.parentElement.setAttribute("data-hint", "Add " + shot.getAttribute("src"));
    };
    shot.addEventListener("error", miss);
    if (shot.complete && !shot.naturalWidth) miss();
  }

  // Places gallery: play videos only while visible; hide tiles whose video is missing
  const tiles = document.querySelectorAll(".place-tile");
  const gallery = document.querySelector(".places");
  const io = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => entries.forEach((en) => {
        const v = en.target.querySelector("video");
        if (!v || reduceMotion) return;
        if (en.isIntersecting) v.play().catch(() => {}); else v.pause();
      }), { threshold: 0.25 })
    : null;
  tiles.forEach((tile) => {
    const v = tile.querySelector("video");
    if (!v) return;
    const missing = () => {
      if (tile.classList.contains("is-missing")) return;
      tile.classList.add("is-missing");
      if (isLocal) tile.querySelector(".place-media").setAttribute("data-hint", "Add " + (v.currentSrc || v.getAttribute("src")));
      else tile.remove();
      if (gallery && !gallery.querySelector(".place-tile")) gallery.closest("section").remove();
    };
    v.addEventListener("error", missing);
    const src = v.querySelector("source");
    if (src) src.addEventListener("error", missing);
    setTimeout(() => { if (v.networkState === HTMLMediaElement.NETWORK_NO_SOURCE) missing(); }, 0);
    setTimeout(() => { if (v.networkState === HTMLMediaElement.NETWORK_NO_SOURCE) missing(); }, 2500);
    if (io) io.observe(tile);
  });

  const yr = document.querySelector("[data-year]");
  if (yr) yr.textContent = new Date().getFullYear();

  resize();
  setPaused(paused);
  onScroll();
  start = performance.now();
  requestAnimationFrame(loop);
})();
