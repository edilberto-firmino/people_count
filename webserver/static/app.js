(function(){
  const img = document.getElementById('stream');
  const countEl = document.getElementById('count');
  const fpsEl = document.getElementById('fps');
  const statusEl = document.getElementById('status');
  const confInput = document.getElementById('conf');
  const applyBtn = document.getElementById('apply');
  const refreshBtn = document.getElementById('refresh');

  let useSnapshots = false;
  let snapshotTimer = null;
  let lastFrameTime = 0;

  function setStatus(text, ok=true){
    statusEl.textContent = text;
    statusEl.style.borderColor = ok ? '#2d3b79' : '#582d2d';
    statusEl.style.background = ok ? '#1a2449' : '#2a0f0f';
    statusEl.style.color = ok ? '#a9baf0' : '#ffbdbd';
  }

  function setImgSrc(url){
    // Add cache-busting
    const ts = Date.now();
    const sep = url.includes('?') ? '&' : '?';
    img.src = `${url}${sep}ts=${ts}`;
  }

  function startStream(){
    stopSnapshots();
    useSnapshots = false;
    setImgSrc('/video');
  }

  function startSnapshots(){
    if (snapshotTimer) return;
    useSnapshots = true;
    const tick = () => {
      setImgSrc('/snapshot');
    };
    tick();
    snapshotTimer = setInterval(tick, 200);
  }

  function stopSnapshots(){
    if (snapshotTimer){
      clearInterval(snapshotTimer);
      snapshotTimer = null;
    }
  }

  // Detect stalled stream: if no load event within some time, fallback
  let stallTimer = null;
  function armStallWatch(){
    if (stallTimer) clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      // If more than 3s passed without a load, fallback
      if (!useSnapshots) startSnapshots();
    }, 3000);
  }

  img.addEventListener('load', () => {
    lastFrameTime = Date.now();
    // If we are on stream and receiving frames, keep as is
    if (!useSnapshots) {
      armStallWatch();
    }
  });

  img.addEventListener('error', () => {
    // Switch to snapshots on error
    if (!useSnapshots) startSnapshots();
  });

  async function poll(){
    try {
      const r = await fetch('/count');
      if(!r.ok) throw new Error('count not ok');
      const data = await r.json();
      countEl.textContent = data.count;
      fpsEl.textContent = (data.fps || 0).toFixed(1);
      setStatus('Online', true);
    } catch(e){
      setStatus('Offline', false);
    }
  }

  async function loadHealth(){
    try {
      const r = await fetch('/health');
      if(!r.ok) throw new Error('health not ok');
      const data = await r.json();
      if (typeof data.conf === 'number') {
        confInput.value = String(data.conf);
      }
    } catch(e){
      // ignore
    }
  }

  applyBtn.addEventListener('click', async () => {
    let conf = parseFloat(confInput.value);
    if (Number.isNaN(conf)) conf = 0.25;
    try {
      const r = await fetch('/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ conf }) });
      if(!r.ok) throw new Error('config not ok');
      const data = await r.json();
      confInput.value = String(data.conf);
    } catch(e){
      alert('Falha ao aplicar configuração.');
    }
  });

  refreshBtn.addEventListener('click', () => {
    // Try stream again
    startStream();
  });

  // init
  startStream();
  armStallWatch();
  loadHealth();
  poll();
  setInterval(poll, 800);
})();
