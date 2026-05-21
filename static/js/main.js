// CCFD main.js — shared utilities

// Auto-dismiss alerts
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.alert-auto').forEach(el => {
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .5s'; }, 4000);
  });

  // Animate progress bars
  document.querySelectorAll('.prog-fill[data-w]').forEach(el => {
    const w = el.getAttribute('data-w');
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.width = w + '%';
    }));
  });
});
