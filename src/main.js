import "./style.css";
import { createKoi } from "./koi.js";

const $ = (s) => document.querySelector(s);
const stage = $("#stage");
const loader = $("#loader");

/* ------------------------------------------------------------------ loader */
let shown = 0;
const setProgress = (p) => {
  shown = Math.max(shown, Math.round(p * 100));
  $("#loaderFill").style.width = shown + "%";
  $("#loaderPct").textContent = String(shown).padStart(2, "0");
};

function reveal() {
  setProgress(1);
  loader.classList.add("gone");
  document.body.classList.add("ready");
  setTimeout(() => loader.remove(), 800);
}

/* --------------------------------------------------------------------- koi */
createKoi({
  mount: stage,
  url: "/koi-fish-animated.glb",
  onProgress: (p) => setProgress(p * 0.98),
})
  .then((koi) => {
    reveal();

    // Fade the koi out as the hero leaves, and stop rendering once it is gone.
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;
        const o = Math.max(0, 1 - scrollY / (innerHeight * 0.62));
        stage.style.opacity = o;
        koi.setVisible(o > 0.01);
      });
    };
    // Bend the headline with the surface. Baseline warp is small and constant
    // — it should always read as being under water — and it swells as rings
    // pass over it.
    const disp = document.getElementById("waterDisp");
    if (disp && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      let shown = 1.5;
      const bend = () => {
        requestAnimationFrame(bend);
        const want = 1.5 + koi.surfaceEnergy() * 17;
        shown += (want - shown) * 0.18;
        disp.setAttribute("scale", shown.toFixed(2));
      };
      bend();
    }

    addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("visibilitychange", () =>
      koi.setVisible(!document.hidden && scrollY < innerHeight * 0.62),
    );
    onScroll();
  })
  .catch((err) => {
    // Never leave the page stuck on the loader — the site reads fine without it.
    console.error("[koi]", err);
    reveal();
  });

/* -------------------------------------------------------------------- menu */
const btn = $("#menuBtn");
const sheet = $("#sheet");
const setMenu = (open) => {
  btn.setAttribute("aria-expanded", String(open));
  sheet.classList.toggle("open", open);
  open ? sheet.removeAttribute("inert") : sheet.setAttribute("inert", "");
  document.body.style.overflow = open ? "hidden" : "";
};
btn.addEventListener("click", () =>
  setMenu(btn.getAttribute("aria-expanded") !== "true"),
);
sheet.addEventListener("click", (e) => e.target.closest("a") && setMenu(false));
addEventListener("keydown", (e) => e.key === "Escape" && setMenu(false));

/* --------------------------------------------------------- reveal on scroll */
const rise = document.querySelectorAll(
  ".panel__head, .work__row, .about__lead, .about__body, .stats, .foot__big, .foot__grid",
);
rise.forEach((el, i) => {
  el.classList.add("rise");
  el.style.transitionDelay = (i % 6) * 55 + "ms";
});
const io = new IntersectionObserver(
  (entries, obs) =>
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      obs.unobserve(e.target);
    }),
  { rootMargin: "0px 0px -12% 0px" },
);
rise.forEach((el) => io.observe(el));
