function $(id) { return document.getElementById(id); }

function render() {
  const payload = JSON.parse($("summary-data").textContent);
  const root = $("root");
  root.innerHTML = '';
  const header = document.createElement('div');
  header.innerHTML = `<h2>${payload.dataset.identity.folder_name}</h2><div class="muted">${payload.dataset.path}</div>`;
  root.appendChild(header);

  // Per-index panels
  for (const [t, stats] of Object.entries(payload.per_index)) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.innerHTML = `<h3>${t}</h3>
      <div class="muted">proteins: ${stats.num_proteins_forward}, files referenced: ${stats.num_files_referenced}`;

    const ds = stats.by_dataset || {};
    const keys = Object.keys(ds);
    if (keys.length) {
      const table = document.createElement('table');
      table.innerHTML = '<thead><tr><th>dataset slug</th><th>files referenced</th><th>proteins referencing</th></tr></thead>';
      const sorted = keys.sort();
      let sumFiles = 0, sumProteins = 0;
      for (const slug of sorted) {
        sumFiles += ds[slug].files_referenced || 0;
        sumProteins += ds[slug].proteins_referencing || 0;
      }
      const tbody = document.createElement('tbody');
      for (const slug of sorted) {
        const f = ds[slug].files_referenced || 0;
        const p = ds[slug].proteins_referencing || 0;
        const fPct = sumFiles ? (f / sumFiles * 100).toFixed(1) : '0.0';
        const pPct = sumProteins ? (p / sumProteins * 100).toFixed(1) : '0.0';
        const row = document.createElement('tr');
        row.innerHTML = `<td>${slug}</td><td>${f} <span class="pct">${fPct}%</span></td><td>${p} <span class="pct">${pPct}%</span></td>`;
        tbody.appendChild(row);
      }
      table.appendChild(tbody);
      const tfoot = document.createElement('tfoot');
      tfoot.innerHTML = `<tr><td><strong>Total</strong></td><td><strong>${sumFiles}</strong></td><td><strong>${sumProteins}</strong></td></tr>`;
      table.appendChild(tfoot);
      panel.appendChild(table);

      const details = document.createElement('details');
      details.innerHTML = '<summary>Show batches per dataset</summary>';
      for (const slug of keys.sort()) {
        const sub = document.createElement('div');
        const batches = ds[slug].files_per_batch || {};
        const subt = document.createElement('table');
        subt.innerHTML = '<thead><tr><th>dataset slug</th><th>batch id</th><th>files</th></tr></thead>';
        const sb = document.createElement('tbody');
        for (const [b, n] of Object.entries(batches)) {
          const r = document.createElement('tr');
          r.innerHTML = `<td>${slug}</td><td>${b}</td><td>${n}</td>`;
          sb.appendChild(r);
        }
        subt.appendChild(sb);
        sub.appendChild(subt);
        details.appendChild(sub);
      }
      panel.appendChild(details);
    }
    root.appendChild(panel);
  }

  // Global rollup
  const roll = document.createElement('div');
  roll.className = 'panel';
  roll.innerHTML = '<h3>Global rollup</h3>';
  const table = document.createElement('table');
  table.innerHTML = '<thead><tr><th>dataset slug</th><th>files referenced</th><th>proteins referencing</th></tr></thead>';
  const tb = document.createElement('tbody');
  let gSumFiles = 0, gSumProteins = 0;
  for (const row of payload.global.top) {
    gSumFiles += row.files_referenced;
    gSumProteins += row.proteins_referencing;
  }
  for (const row of payload.global.top) {
    const fPct = gSumFiles ? (row.files_referenced / gSumFiles * 100).toFixed(1) : '0.0';
    const pPct = gSumProteins ? (row.proteins_referencing / gSumProteins * 100).toFixed(1) : '0.0';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.slug}</td><td>${row.files_referenced} <span class="pct">${fPct}%</span></td><td>${row.proteins_referencing} <span class="pct">${pPct}%</span></td>`;
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  const gFoot = document.createElement('tfoot');
  gFoot.innerHTML = `<tr><td><strong>Total</strong></td><td><strong>${gSumFiles}</strong></td><td><strong>${gSumProteins}</strong></td></tr>`;
  table.appendChild(gFoot);
  roll.appendChild(table);
  root.appendChild(roll);
}

window.addEventListener('DOMContentLoaded', render);
