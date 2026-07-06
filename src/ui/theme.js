const STORAGE_KEY = 'ddpm-playground-theme';

function systemPrefersDark() {
  return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
}

function resolve(theme) {
  return theme === 'system' ? (systemPrefersDark() ? 'dark' : 'light') : theme;
}

function apply(theme) {
  document.documentElement.setAttribute('data-theme', resolve(theme));
  window.dispatchEvent(new CustomEvent('app:theme-changed'));
}

// Wires a <select> of dark/light/system options to the page theme, persisting
// the choice in localStorage and reacting to OS theme changes when on 'system'.
export function initTheme(selectEl) {
  const saved = localStorage.getItem(STORAGE_KEY) || 'dark';
  selectEl.value = saved;
  apply(saved);

  selectEl.addEventListener('change', () => {
    localStorage.setItem(STORAGE_KEY, selectEl.value);
    apply(selectEl.value);
  });

  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (selectEl.value === 'system') apply('system');
    });
  }
}
