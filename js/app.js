/* ═══════════════════════════════════════════════════════════
   FORGE v5.0 — Health Tracker
   ═══════════════════════════════════════════════════════════ */

// ─── CONFIG ───────────────────────────────────────────────
const AI_FOOD_URL  = window.FORGE_AI_API_URL  || 'https://forge-food-ai.vercel.app/api/food/analyze';
const AI_COACH_URL = window.FORGE_AI_COACH_URL || 'https://forge-food-ai.vercel.app/api/health/coach';
const APP_VERSION  = '5.0';
const HAS_SUPABASE = !!(window.ARCHITECT_SUPABASE_URL && window.ARCHITECT_SUPABASE_ANON_KEY);

// ─── STATE ─────────────────────────────────────────────────
let DB = { calLog: [], workoutLog: [], runLog: [], waterLog: [], bodyWeightLog: [] };
let WB = null;
let GOAL = parseInt(localStorage.getItem('forge_goal') || '2400', 10);
let WATER_GOAL_ML = parseInt(localStorage.getItem('forge_water_goal') || '2500', 10);
let supabaseClient = null;
let currentUser    = null;
let userStats      = { xp: 0, level: 1, streak: 0 };
let friendsList    = [];
let TODAY          = todayStr();
let CUR_SCREEN     = 'home';
let CUR_LOG_TAB    = 'nutrition';
let CUR_STAT_TAB   = 'week';
let activeSplit    = null;
let completedBlocks = new Set(JSON.parse(localStorage.getItem('forge_blocks_' + todayStr()) || '[]'));

// Numpad state
let numpadVal = '0';
let selectedCourse = 'Breakfast';

// AI food
let aiFoodItems    = [];
let aiFoodPreviewUrl = '';

// Voice
let voiceRecognition = null;
let voiceListening   = false;

// Timer
let timerInterval     = null;
let timerRemainMs     = 0;
let timerEndAt        = 0;

// Run tracking
let runMap           = null;
let runPolyline      = null;
let runWatchId       = null;
let runTickInterval  = null;
let runActive        = false;
let runPaused        = false;
let runPoints        = [];
let runDistKm        = 0;
let runStartedAt     = null;
let runPausedAccumMs = 0;
let runPauseBeganAt  = null;
let runGpsEnabled    = true;

// ─── DATE UTILS ───────────────────────────────────────────
function todayStr() {
  const d = new Date();
  return d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate());
}
function pad(n) { return String(n).padStart(2,'0') }
function nowTime() {
  const d = new Date();
  return pad(d.getHours())+':'+pad(d.getMinutes());
}
function dow(dateStr) {
  return ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][new Date(dateStr+'T12:00:00').getDay()];
}
function fmt(n) { return n.toLocaleString() }
function fmtDecimal(n,d=1) { return Number(n).toFixed(d) }
function datePlusDays(baseStr, days) {
  const d = new Date(baseStr+'T12:00:00');
  d.setDate(d.getDate() + days);
  return d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate());
}
function formatDate(str) {
  const d = new Date(str+'T12:00:00');
  return d.toLocaleDateString('en-AU', { weekday:'short', day:'numeric', month:'short' });
}
function formatDuration(sec) {
  const m = Math.floor(sec/60), s = Math.round(sec%60);
  return pad(m)+':'+pad(s);
}
function formatPace(secPerKm) {
  if(!secPerKm || !isFinite(secPerKm)) return '—';
  const m = Math.floor(secPerKm/60), s = Math.round(secPerKm%60);
  return pad(m)+':'+pad(s);
}

// ─── TOAST ────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, success=true, error=false) {
  const el = document.getElementById('toast');
  if(!el) return;
  el.textContent = msg;
  el.className = 'show' + (success?' success':'') + (error?' error':'');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>{ el.className='' }, 2600);
}

// ─── NAVIGATION ───────────────────────────────────────────
function navTo(screen) {
  if (CUR_SCREEN === screen) return;
  const prev = document.getElementById('screen-'+CUR_SCREEN);
  const next = document.getElementById('screen-'+screen);
  if (!next) return;
  if (prev) { prev.classList.add('exiting'); setTimeout(()=>prev.classList.remove('active','exiting'),300) }
  next.classList.add('active');
  CUR_SCREEN = screen;
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  const nb = document.getElementById('nav-'+screen);
  if (nb) nb.classList.add('active');
  // Screen-specific refresh
  if (screen==='home')    renderHome();
  if (screen==='log')     renderLogScreen();
  if (screen==='run')     renderRunIdle();
  if (screen==='stats')   renderStats();
  if (screen==='profile') renderProfile();
}

function switchLogTab(tab) {
  CUR_LOG_TAB = tab;
  ['nutrition','workout','water'].forEach(t => {
    document.getElementById('tab-btn-'+t)?.classList.toggle('active', t===tab);
    document.getElementById('tab-'+t)?.classList.toggle('active', t===tab);
  });
  if (tab==='nutrition') renderNutrition();
  if (tab==='workout')   renderWorkout();
  if (tab==='water')     renderWater();
}

function switchStatsTab(tab) {
  CUR_STAT_TAB = tab;
  ['week','body','routine'].forEach(t => {
    document.getElementById('stat-tab-btn-'+t)?.classList.toggle('active', t===tab);
    document.getElementById('stat-tab-'+t)?.classList.toggle('active', t===tab);
  });
  if (tab==='week')    renderWeekStats();
  if (tab==='body')    renderBodyStats();
  if (tab==='routine') renderRoutine();
}

function showRoutineTab() { navTo('stats'); switchStatsTab('routine') }

// ─── SHEETS / MODALS ──────────────────────────────────────
let openSheetId = null;
function openSheet(id) {
  closeAllSheets(false);
  const el = document.getElementById(id);
  if (!el) return;
  document.getElementById('sheet-backdrop').classList.add('show');
  el.classList.add('show');
  openSheetId = id;
}
function closeAllSheets(hideBackdrop=true) {
  document.querySelectorAll('.sheet').forEach(s => s.classList.remove('show'));
  if (hideBackdrop) document.getElementById('sheet-backdrop')?.classList.remove('show');
  openSheetId = null;
}
function openQuickLog()          { openSheet('sheet-quick-log') }
function openFoodSheet()         { numpadVal='0'; updNumpad(); openSheet('sheet-food') }
function openVoiceSheet()        { resetVoiceState(); openSheet('sheet-voice') }
function openAiCameraSheet()     { openSheet('sheet-ai-camera') }
function openCoachModal()        { openSheet('sheet-coach'); fetchDailyCoach() }
function openBodyWeightSheet()   {
  document.getElementById('bw-date-input').value = TODAY;
  openSheet('sheet-body-weight');
}
function openGoalSheet()         {
  document.getElementById('goal-input').value = GOAL;
  openSheet('sheet-goal');
}
function openAppleHealthSheet()  { openSheet('sheet-apple-health') }
function openImportSheet()       { openSheet('sheet-import') }
function openAboutSheet()        { openSheet('sheet-about') }
function showSessionHistory()    { renderSessionHistory(); openSheet('sheet-session-history') }
function openTimerSheet()        { openSheet('sheet-timer') }
function openCustomWater() {
  const ml = prompt('Enter amount in ml:');
  if (ml && !isNaN(ml) && +ml > 0) logWater(+ml);
}

// ─── SUPABASE / AUTH ──────────────────────────────────────
async function initSupabase() {
  if (!HAS_SUPABASE) return;
  try {
    supabaseClient = supabase.createClient(
      window.ARCHITECT_SUPABASE_URL,
      window.ARCHITECT_SUPABASE_ANON_KEY
    );
    const { data: { session } } = await supabaseClient.auth.getSession();
    if (session?.user) await onSignIn(session.user);
    supabaseClient.auth.onAuthStateChange(async (_evt, sess) => {
      if (sess?.user) await onSignIn(sess.user);
      else onSignOut();
    });
  } catch(e) { console.warn('Supabase init failed', e) }
}

async function signInWithGoogle() {
  if (!supabaseClient) { toast('Not connected to server', false, true); return }
  await supabaseClient.auth.signInWithOAuth({
    provider: 'google',
    options: { redirectTo: window.location.href }
  });
}
function skipAuth() {
  document.getElementById('auth-screen').classList.add('hidden');
  initApp();
}
async function onSignIn(user) {
  currentUser = user;
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('signout-btn').style.display = 'flex';
  document.getElementById('danger-clear-row').style.display = 'flex';
  await loadFromSupabase();
  await loadUserStats();
  initApp();
}
function onSignOut() {
  currentUser = null;
  document.getElementById('signout-btn').style.display = 'none';
  document.getElementById('danger-clear-row').style.display = 'none';
}
async function confirmSignOut() {
  if (!confirm('Sign out?')) return;
  await supabaseClient?.auth.signOut();
  toast('Signed out');
}

// ─── WORKBOOK / LOCAL STORAGE ─────────────────────────────
function loadLocalDB() {
  const raw = localStorage.getItem('forge_db_v5');
  if (raw) {
    try { const d = JSON.parse(raw); Object.assign(DB, d) } catch(e){}
  }
  // Migrate from older keys
  if (!DB.waterLog)       DB.waterLog = [];
  if (!DB.bodyWeightLog)  DB.bodyWeightLog = [];
  GOAL = parseInt(localStorage.getItem('forge_goal') || '2400', 10);
}
function saveLocalDB() {
  localStorage.setItem('forge_db_v5', JSON.stringify(DB));
}

// ─── SUPABASE DATA LAYER ───────────────────────────────────
async function loadFromSupabase() {
  if (!supabaseClient || !currentUser) return;
  try {
    const uid = currentUser.id;
    const [cal, work, run, water, bw] = await Promise.all([
      supabaseClient.from('calorie_log').select('*').eq('user_id', uid).order('Timestamp', {ascending:false}),
      supabaseClient.from('workout_log').select('*').eq('user_id', uid),
      supabaseClient.from('run_log').select('*').eq('user_id', uid).order('Start_Timestamp', {ascending:false}),
      supabaseClient.from('water_log').select('*').eq('user_id', uid).order('Timestamp', {ascending:false}).catch(()=>({data:[]})),
      supabaseClient.from('body_weight_log').select('*').eq('user_id', uid).order('Date', {ascending:false}).catch(()=>({data:[]})),
    ]);
    if (cal.data)  DB.calLog      = cal.data;
    if (work.data) DB.workoutLog  = work.data;
    if (run.data)  DB.runLog      = run.data;
    if (water.data) DB.waterLog   = water.data;
    if (bw.data)   DB.bodyWeightLog = bw.data;
    saveLocalDB();
  } catch(e) { console.warn('Supabase load error', e) }
}

async function insertCalorieRow(row) {
  if (supabaseClient && currentUser) {
    await supabaseClient.from('calorie_log').insert({ ...row, user_id: currentUser.id });
  }
}
async function deleteCalorieRow(id) {
  if (supabaseClient && currentUser) {
    await supabaseClient.from('calorie_log').delete().eq('ID', id).eq('user_id', currentUser.id);
  }
  DB.calLog = DB.calLog.filter(r => r.ID !== id);
  saveLocalDB();
}
async function insertWorkoutRow(row) {
  if (supabaseClient && currentUser) {
    await supabaseClient.from('workout_log').insert({ ...row, user_id: currentUser.id });
  }
}
async function deleteWorkoutRow(id) {
  if (supabaseClient && currentUser) {
    await supabaseClient.from('workout_log').delete().eq('ID', id).eq('user_id', currentUser.id);
  }
  DB.workoutLog = DB.workoutLog.filter(r => r.ID !== id);
  saveLocalDB();
}
async function insertWaterRow(row) {
  if (supabaseClient && currentUser) {
    await supabaseClient.from('water_log').insert({ ...row, user_id: currentUser.id }).catch(()=>{});
  }
}
async function insertBodyWeightRow(row) {
  if (supabaseClient && currentUser) {
    await supabaseClient.from('body_weight_log').insert({ ...row, user_id: currentUser.id }).catch(()=>{});
  }
}
async function loadUserStats() {
  if (!supabaseClient || !currentUser) return;
  try {
    const { data } = await supabaseClient.from('user_stats').select('*').eq('user_id', currentUser.id).single();
    if (data) userStats = { xp: data.xp||0, level: data.level||1, streak: data.streak||0 };
  } catch(e){}
}
async function updateUserStats() {
  if (!supabaseClient || !currentUser) return;
  try {
    await supabaseClient.from('user_stats').upsert({
      user_id: currentUser.id,
      email: currentUser.email,
      xp: userStats.xp,
      level: userStats.level,
      streak: userStats.streak
    }, { onConflict: 'user_id' });
  } catch(e){}
}

// ─── XP / GAMIFICATION ────────────────────────────────────
const XP_PER_LEVEL = 100;
function addXP(amount) {
  userStats.xp += amount;
  const newLevel = Math.floor(userStats.xp / XP_PER_LEVEL) + 1;
  if (newLevel > userStats.level) {
    userStats.level = newLevel;
    spawnConfetti();
    toast(`LEVEL ${newLevel} UNLOCKED! 🎉`, true);
  }
  updateUserStats();
}
function updateStreak() {
  const key = 'forge_streak_last';
  const last = localStorage.getItem(key);
  if (last === TODAY) return;
  const yesterday = datePlusDays(TODAY, -1);
  if (last === yesterday) userStats.streak++;
  else if (last !== TODAY) userStats.streak = 1;
  localStorage.setItem(key, TODAY);
  updateUserStats();
}

// ─── HOME SCREEN ──────────────────────────────────────────
function renderHome() {
  // Date/greeting
  const now = new Date();
  const hour = now.getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
  const name = currentUser?.email?.split('@')[0] || '';
  document.getElementById('home-greeting').textContent = greeting + (name ? `, ${name}` : '');
  document.getElementById('home-date').textContent = now.toLocaleDateString('en-AU', { weekday:'long', day:'numeric', month:'long' });

  // Calorie ring
  const todayCals = DB.calLog.filter(r => r.Date===TODAY).reduce((s,r)=>s+(r.Calories_kcal||0),0);
  const calPct    = Math.min(todayCals / GOAL, 1);
  const calCirc   = 2 * Math.PI * 70; // r=70
  document.getElementById('ring-cal-num').textContent   = fmt(todayCals);
  document.getElementById('ring-cal-label').textContent = `${fmt(todayCals)} / ${fmt(GOAL)}`;
  document.getElementById('cal-bar').style.width        = (calPct*100)+'%';
  const calEl = document.getElementById('ring-cal');
  if (calEl) calEl.style.strokeDashoffset = calCirc * (1-calPct);

  // Workout ring
  const todaySets = DB.workoutLog.filter(r=>r.Date===TODAY).length;
  const splits    = [...new Set(DB.workoutLog.filter(r=>r.Date===TODAY).map(r=>r.Split_Name).filter(Boolean))];
  const workPct   = Math.min(todaySets / 24, 1);
  const workCirc  = 2 * Math.PI * 54;
  document.getElementById('ring-work-label').textContent = `${todaySets} sets`;
  document.getElementById('ring-work-split').textContent = splits.length ? splits.join(' + ') : 'No session yet';
  const workEl = document.getElementById('ring-work');
  if (workEl) workEl.style.strokeDashoffset = workCirc * (1-workPct);

  // Water chip
  const todayWater = DB.waterLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Amount_ml||0),0);
  const waterPct   = Math.min(todayWater / WATER_GOAL_ML, 1);
  document.getElementById('home-water').textContent = todayWater >= 1000 ? fmtDecimal(todayWater/1000)+'L' : todayWater+'ml';
  document.getElementById('water-home-bar').style.width = (waterPct*100)+'%';

  // Streak
  document.getElementById('home-streak').textContent = userStats.streak;
  document.getElementById('home-xp').textContent     = fmt(userStats.xp)+' XP';

  // Protein
  const todayProtein = DB.calLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Protein_g||0),0);
  const mealCount    = DB.calLog.filter(r=>r.Date===TODAY).length;
  document.getElementById('home-protein').textContent    = todayProtein>0 ? todayProtein+'g' : '—';
  document.getElementById('home-meals-count').textContent= `${mealCount} meal${mealCount!==1?'s':''} logged`;

  // Today's split
  document.getElementById('home-split').textContent = splits.length ? splits[0] : '—';
  document.getElementById('home-sets').textContent  = `${todaySets} set${todaySets!==1?'s':''} done`;

  // Routine strip
  renderRoutineStrip();

  // Recent activity
  renderRecentActivity();

  // AI coach (from cache)
  const cached = localStorage.getItem('coach_feedback_'+TODAY);
  if (cached) document.getElementById('coach-feedback-home').textContent = cached;
  else fetchDailyCoachQuiet();
}

function renderRecentActivity() {
  const el = document.getElementById('recent-activity');
  if (!el) return;
  const items = [];

  // Last 5 cal entries today
  DB.calLog.filter(r=>r.Date===TODAY).slice(-5).reverse().forEach(r => {
    items.push({ type:'food', name: extractFoodName(r), meta: r.Course, val: (r.Calories_kcal||0)+'kcal', icon:'🍽️' });
  });
  // Last 3 workout sets today
  const workSets = DB.workoutLog.filter(r=>r.Date===TODAY);
  if (workSets.length) {
    const lastEx = workSets[workSets.length-1];
    items.unshift({ type:'workout', name: lastEx.Exercise||'Workout', meta: lastEx.Split_Name||'', val: workSets.length+' sets', icon:'💪' });
  }
  // Last run
  const lastRun = DB.runLog[0];
  if (lastRun) {
    items.unshift({ type:'run', name: fmtDecimal(lastRun.Distance_Km||0)+'km run', meta: lastRun.Date, val: formatDuration(lastRun.Duration_Sec||0), icon:'🏃' });
  }

  if (!items.length) {
    el.innerHTML = `<div class="empty-state" style="padding:20px 0">
      <div class="empty-icon">📋</div>
      <div class="empty-title">Nothing logged yet</div>
      <div class="empty-desc">Tap + to log your first meal or workout</div>
    </div>`;
    return;
  }
  el.innerHTML = items.slice(0,6).map(it => `
    <div class="log-item">
      <div class="log-icon ${it.type}"><span>${it.icon}</span></div>
      <div class="log-info">
        <div class="log-name">${esc(it.name)}</div>
        <div class="log-meta">${esc(it.meta)}</div>
      </div>
      <div class="log-val">${it.val}</div>
    </div>`).join('');
}

function extractFoodName(row) {
  if (row.Notes?.startsWith('VOICE_ITEM:')) return row.Notes.slice(11);
  if (row.Notes?.startsWith('AI:')) return row.Notes.slice(3);
  return row.Notes || 'Food entry';
}

// ─── NUTRITION ────────────────────────────────────────────
function renderLogScreen() {
  document.getElementById('log-date-lbl').textContent = new Date().toLocaleDateString('en-AU', { weekday:'short', day:'numeric', month:'short' });
  switchLogTab(CUR_LOG_TAB);
}

function renderNutrition() {
  const todayCals    = DB.calLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Calories_kcal||0),0);
  const remaining    = Math.max(GOAL - todayCals, 0);
  const pct          = Math.min(todayCals/GOAL*100, 100);
  const todayProtein = DB.calLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Protein_g||0),0);
  const todayCarbs   = DB.calLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Carbs_g||0),0);
  const todayFat     = DB.calLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Fat_g||0),0);

  document.getElementById('nut-cal-num').textContent        = fmt(todayCals);
  document.getElementById('nut-goal').textContent           = fmt(GOAL);
  document.getElementById('nut-remaining').textContent      = fmt(remaining);
  document.getElementById('nut-progress-bar').style.width   = pct+'%';
  if (todayCals > GOAL*0.9) document.getElementById('nut-progress-bar').style.background = 'var(--red)';
  else document.getElementById('nut-progress-bar').style.background = '';

  document.getElementById('mac-protein').textContent = todayProtein+'g';
  document.getElementById('mac-carbs').textContent   = todayCarbs+'g';
  document.getElementById('mac-fat').textContent     = todayFat+'g';

  // Meal sections
  const courses = ['Breakfast','Lunch','Dinner','Snack'];
  const icons   = { Breakfast:'🌅', Lunch:'☀️', Dinner:'🌙', Snack:'🫐' };
  const wrap    = document.getElementById('meal-sections-wrap');
  wrap.innerHTML = courses.map(course => {
    const entries = DB.calLog.filter(r=>r.Date===TODAY && r.Course===course);
    const total   = entries.reduce((s,r)=>s+(r.Calories_kcal||0),0);
    return `
      <div class="meal-section">
        <div class="meal-hdr">
          <div class="meal-title">
            <span class="meal-icon">${icons[course]}</span>${course}
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <span class="meal-kcal">${total ? fmt(total)+' kcal' : ''}</span>
            <button class="meal-add" onclick="openFoodSheetFor('${course}')">+</button>
          </div>
        </div>
        ${entries.length ? entries.map(r => `
          <div class="meal-entry" id="me-${r.ID}" ontouchstart="swipeStart(event,'me-${r.ID}')" ontouchmove="swipeMove(event,'me-${r.ID}')" ontouchend="swipeEnd(event,'me-${r.ID}')">
            <div class="meal-entry-inner">
              <div class="meal-entry-name">${esc(extractFoodName(r))}</div>
              <div class="meal-entry-cal">${r.Calories_kcal} kcal</div>
            </div>
            <div class="del-hint" onclick="deleteCalEntry(${r.ID})">Delete</div>
          </div>`).join('') : `<div style="padding:8px 0;font-size:13px;color:var(--t2)">Nothing logged yet</div>`}
      </div>`;
  }).join('');
}

function openFoodSheetFor(course) {
  numpadVal = '0';
  updNumpad();
  // Pre-select course
  selectedCourse = course;
  document.querySelectorAll('.course-pill').forEach(b => {
    b.classList.toggle('active', b.dataset.course === course);
  });
  openSheet('sheet-food');
}

// ─── NUMPAD ───────────────────────────────────────────────
function numpadInput(digit) {
  if (numpadVal==='0' && digit!=='00') numpadVal = digit;
  else numpadVal += digit;
  if (numpadVal.length > 5) numpadVal = numpadVal.slice(0,5);
  updNumpad();
}
function numpadDelete() {
  numpadVal = numpadVal.slice(0,-1) || '0';
  updNumpad();
}
function updNumpad() {
  const el = document.getElementById('numpad-val');
  if (el) el.textContent = fmt(parseInt(numpadVal||'0',10));
}
function selectCourse(btn) {
  selectedCourse = btn.dataset.course;
  document.querySelectorAll('.course-pill').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}
async function submitFoodLog() {
  const cals = parseInt(numpadVal||'0',10);
  const name = document.getElementById('food-name-input')?.value?.trim() || 'Food entry';
  if (!cals) { toast('Enter calories first', false, true); return }
  const id = nextId(DB.calLog);
  const row = {
    ID: id, Date: TODAY, Day: dow(TODAY), Course: selectedCourse,
    Calories_kcal: cals, Timestamp: nowTime(),
    Notes: name, Protein_g: 0, Carbs_g: 0, Fat_g: 0
  };
  DB.calLog.push(row);
  saveLocalDB();
  await insertCalorieRow(row);
  addXP(2);
  updateStreak();
  closeAllSheets();
  renderNutrition();
  renderHome();
  if (document.getElementById('food-name-input')) document.getElementById('food-name-input').value='';
  numpadVal='0'; updNumpad();
  toast(`${name}: ${fmt(cals)} kcal logged`, true);
}

async function deleteCalEntry(id) {
  if (!confirm('Delete this entry?')) return;
  await deleteCalorieRow(id);
  renderNutrition();
  renderHome();
  toast('Deleted', true);
}

function nextId(arr) {
  return arr.length ? Math.max(...arr.map(r=>r.ID||0))+1 : 1;
}

// ─── VOICE INPUT ──────────────────────────────────────────
function resetVoiceState() {
  if (voiceRecognition && voiceListening) { voiceRecognition.stop(); voiceListening=false }
  const t = document.getElementById('voice-transcript');
  const b = document.getElementById('voice-parsed-badge');
  const btn = document.getElementById('voice-listen-btn');
  if (t) t.value = '';
  if (b) { b.style.display='none'; b.textContent='' }
  if (btn) btn.textContent = 'Start Listening';
  document.getElementById('voice-item')?.setAttribute('value','');
  document.getElementById('voice-calories')?.setAttribute('value','');
  document.getElementById('voice-waveform')?.classList.remove('voice-active');
}
function ensureVoiceRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  if (voiceRecognition) return voiceRecognition;
  voiceRecognition = new SR();
  voiceRecognition.lang = 'en-AU';
  voiceRecognition.interimResults = true;
  voiceRecognition.maxAlternatives = 1;
  voiceRecognition.onresult = e => {
    const txt = [...e.results].map(r=>r[0].transcript).join(' ');
    const el = document.getElementById('voice-transcript');
    if (el) { el.value = txt; parseVoiceDraft() }
  };
  voiceRecognition.onerror = () => { voiceListening=false; updateVoiceBtnState(false) };
  voiceRecognition.onend   = () => { voiceListening=false; updateVoiceBtnState(false) };
  return voiceRecognition;
}
function toggleVoiceCapture() {
  const r = ensureVoiceRecognition();
  if (!r) {
    const el = document.getElementById('voice-transcript');
    if (el) { el.focus(); el.click() }
    toast('Use keyboard mic on Safari', false, true);
    return;
  }
  if (voiceListening) {
    r.stop(); voiceListening=false; updateVoiceBtnState(false);
  } else {
    voiceListening=true; updateVoiceBtnState(true); r.start();
  }
}
function updateVoiceBtnState(on) {
  const btn = document.getElementById('voice-listen-btn');
  const wave = document.getElementById('voice-waveform');
  if (btn) btn.textContent = on ? 'Stop Listening' : 'Start Listening';
  wave?.classList.toggle('voice-active', on);
}
function parseVoiceMeal(text) {
  const low = String(text||'').toLowerCase();
  let course = 'Breakfast';
  if (/\blunch\b/.test(low)) course = 'Lunch';
  else if (/\bdinner\b/.test(low)) course = 'Dinner';
  else if (/\bsnack\b/.test(low)) course = 'Snack';
  const m = text.match(/(\d+)\s*(?:k?cal(?:ories?)?)/i) || text.match(/\b(\d{2,4})\b/);
  if (!m) return null;
  const calories = parseInt(m[1],10);
  let item = text
    .replace(/\bfor\s+(?:breakfast|lunch|dinner|snack)\b/ig,'')
    .replace(/\b(?:breakfast|lunch|dinner|snack)\b/ig,'')
    .replace(/(\d+)\s*(?:k?cal(?:ories?)?)/ig,'')
    .replace(/\b(?:i had|i ate|log|add|ate|had|about|roughly)\b/ig,'')
    .replace(/[.,]+/g,' ').replace(/\s+/g,' ').trim();
  return { item: item||'Meal', calories, course };
}
function parseVoiceDraft() {
  const t = document.getElementById('voice-transcript')?.value?.trim();
  const p = parseVoiceMeal(t||'');
  const badge = document.getElementById('voice-parsed-badge');
  if (p && badge) {
    badge.style.display='block';
    badge.textContent = `${p.item} • ${p.calories} kcal • ${p.course}`;
    const itemEl = document.getElementById('voice-item');
    const calEl  = document.getElementById('voice-calories');
    const crsEl  = document.getElementById('voice-course');
    if (itemEl) itemEl.value = p.item;
    if (calEl)  calEl.value  = p.calories;
    if (crsEl)  crsEl.value  = p.course;
  } else if (badge) { badge.style.display='none' }
}
function syncVoiceFields() {}
async function logVoiceMeal() {
  parseVoiceDraft();
  const item = document.getElementById('voice-item')?.value?.trim();
  const cals = parseInt(document.getElementById('voice-calories')?.value||'0',10);
  const course = document.getElementById('voice-course')?.value || 'Breakfast';
  if (!item||!cals) { toast('Say item and calories first', false, true); return }
  const id  = nextId(DB.calLog);
  const row = { ID:id, Date:TODAY, Day:dow(TODAY), Course:course, Calories_kcal:cals,
    Timestamp:nowTime(), Notes:'VOICE_ITEM:'+item, Protein_g:0, Carbs_g:0, Fat_g:0 };
  DB.calLog.push(row);
  saveLocalDB();
  await insertCalorieRow(row);
  addXP(2); updateStreak();
  closeAllSheets();
  renderNutrition(); renderHome();
  toast(`${item}: ${fmt(cals)} kcal logged`, true);
}

// ─── AI FOOD ANALYSIS ─────────────────────────────────────
function handleAiFoodImage(e) {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  aiFoodPreviewUrl = url;
  const img = document.getElementById('ai-food-img');
  const wrap = document.getElementById('ai-preview-wrap');
  if (img) img.src = url;
  if (wrap) wrap.style.display='block';
  document.getElementById('ai-camera-label').textContent = 'Change Photo';
  analyzeAiFoodImage(file);
}
async function analyzeAiFoodImage(file) {
  const res = document.getElementById('ai-food-results');
  const analyzing = document.getElementById('ai-analyzing');
  const logBtn = document.getElementById('ai-log-btn');
  if (res) res.innerHTML='';
  if (analyzing) analyzing.style.display='block';
  if (logBtn) logBtn.style.display='none';
  aiFoodItems = [];
  try {
    const form = new FormData();
    form.append('image', file);
    const r = await fetch(AI_FOOD_URL, { method:'POST', body:form });
    if (!r.ok) throw new Error('AI API error');
    const data = await r.json();
    if (data.items && data.items.length) {
      aiFoodItems = data.items;
    } else if (data.name) {
      aiFoodItems = [data];
    } else throw new Error('No items returned');
    if (analyzing) analyzing.style.display='none';
    renderAiFoodResults();
    if (logBtn) logBtn.style.display='block';
  } catch(err) {
    if (analyzing) analyzing.style.display='none';
    if (res) res.innerHTML = `<div style="color:var(--red);font-size:13px;text-align:center;padding:12px">Analysis failed. Try a clearer photo.</div>`;
    console.warn('AI food error', err);
  }
}
function renderAiFoodResults() {
  const res = document.getElementById('ai-food-results');
  if (!res) return;
  res.innerHTML = aiFoodItems.map((item,i) => `
    <div class="ai-food-item">
      <input type="checkbox" id="ai-item-${i}" checked>
      <div style="flex:1">
        <div class="ai-food-name">${esc(item.name||'Food')}</div>
        <div class="ai-food-macros">${item.protein?`P:${item.protein}g `:''} ${item.carbs?`C:${item.carbs}g `:''}${item.fat?`F:${item.fat}g`:''}</div>
      </div>
      <div class="ai-food-cal">${item.calories||0} kcal</div>
    </div>`).join('');
}
async function logAiFoodItems() {
  const course = document.getElementById('ai-course-sel')?.value || 'Breakfast';
  const selected = aiFoodItems.filter((_,i) => document.getElementById('ai-item-'+i)?.checked);
  if (!selected.length) { toast('Select at least one item', false, true); return }
  for (const item of selected) {
    const id  = nextId(DB.calLog);
    const row = { ID:id, Date:TODAY, Day:dow(TODAY), Course:course,
      Calories_kcal: item.calories||0, Timestamp:nowTime(),
      Notes:'AI:'+item.name, Protein_g:item.protein||0, Carbs_g:item.carbs||0, Fat_g:item.fat||0 };
    DB.calLog.push(row);
    await insertCalorieRow(row);
  }
  saveLocalDB();
  addXP(3); updateStreak();
  closeAllSheets();
  renderNutrition(); renderHome();
  toast(`${selected.length} item(s) logged`, true);
}

// ─── AI COACH ─────────────────────────────────────────────
async function fetchDailyCoach() {
  const key = 'coach_feedback_'+TODAY;
  const cached = localStorage.getItem(key);
  const statusEl = document.getElementById('coach-status-txt');
  const feedEl   = document.getElementById('coach-feedback-full');
  if (cached) {
    if (statusEl) statusEl.textContent = 'Daily insight loaded';
    if (feedEl)   feedEl.textContent   = cached;
    return;
  }
  if (statusEl) statusEl.textContent = 'Analyzing your week…';
  const payload = buildCoachPayload();
  try {
    const r = await fetch(AI_COACH_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    if (!r.ok) throw new Error('API error');
    const data = await r.json();
    const fb = data.feedback || data.tip || 'Keep pushing forward!';
    localStorage.setItem(key, fb);
    if (statusEl) statusEl.textContent = 'Daily insight ready';
    if (feedEl)   feedEl.textContent   = fb;
    document.getElementById('coach-feedback-home').textContent = fb;
  } catch(e) {
    if (statusEl) statusEl.textContent = 'Coach offline';
    if (feedEl)   feedEl.textContent   = 'Unable to fetch insights. Check back later.';
  }
}
async function fetchDailyCoachQuiet() {
  const key = 'coach_feedback_'+TODAY;
  if (localStorage.getItem(key)) return;
  const payload = buildCoachPayload();
  try {
    const r = await fetch(AI_COACH_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    if (!r.ok) return;
    const data = await r.json();
    const fb = data.feedback||data.tip||'';
    if (fb) {
      localStorage.setItem(key, fb);
      const el = document.getElementById('coach-feedback-home');
      if (el) el.textContent = fb;
    }
  } catch(e){}
}
function buildCoachPayload() {
  const last7 = Array.from({length:7},(_,i)=>datePlusDays(TODAY,-i-1)).reverse();
  return { goal:GOAL, days: last7.map(d=>({
    date:d,
    calories: DB.calLog.filter(r=>r.Date===d).reduce((s,r)=>s+(r.Calories_kcal||0),0),
    workouts: [...new Set(DB.workoutLog.filter(r=>r.Date===d).map(r=>r.Split_Name).filter(Boolean))]
  }))};
}

// ─── WORKOUT ──────────────────────────────────────────────
const WORKOUT_PLAN = {
  push: [
    { name:'Flat barbell bench press', target:'4×6–8', note:'Primary chest builder' },
    { name:'Incline dumbbell press',   target:'3×8–10', note:'Upper chest + shoulder' },
    { name:'Overhead press (barbell)', target:'4×6–8', note:'Shoulder mass' },
    { name:'Cable lateral raises',     target:'4×12–15', note:'Wide shoulder look' },
    { name:'Rear delt cable fly',      target:'3×15', note:'3D shoulder shape' },
    { name:'Cable chest fly',          target:'3×12', note:'Inner chest definition' },
    { name:'Tricep pushdown (rope)',   target:'3×12', note:'Arm thickness' },
    { name:'Overhead tricep extension',target:'3×12', note:'Long head definition' }
  ],
  pull: [
    { name:'Weighted pull-ups',         target:'4×6–8', note:'V-taper builder' },
    { name:'Barbell bent-over row',     target:'4×8', note:'Back thickness' },
    { name:'Cable seated row',          target:'3×10–12', note:'Mid-back' },
    { name:'Face pulls (cable)',        target:'3×15', note:'Rear delts + rotator' },
    { name:'Barbell curl',              target:'4×8–10', note:'Bicep peak + size' },
    { name:'Incline dumbbell curl',     target:'3×10–12', note:'Long head stretch' },
    { name:'Hammer curl',              target:'3×12', note:'Brachialis + forearm' }
  ],
  legs: [
    { name:'Barbell back squat',   target:'4×6–8', note:'Foundation movement' },
    { name:'Romanian deadlift',    target:'4×8–10', note:'Hamstring + glutes' },
    { name:'Leg press',            target:'3×10–12', note:'Volume on quads' },
    { name:'Walking lunges',       target:'3×12 each', note:'Balance + shape' },
    { name:'Leg curl (machine)',   target:'3×12', note:'Hamstring isolation' },
    { name:'Standing calf raises', target:'4×20', note:'Slow + full range' }
  ],
  cardio: [
    { name:'Treadmill incline walk/jog', target:'25–30 min', note:'Incline 6–8' },
    { name:'Hanging leg raises',         target:'3×15', note:'Lower abs' },
    { name:'Cable crunches',             target:'3×15', note:'Weighted ab work' },
    { name:'Plank hold',                 target:'3×60 sec', note:'Full tension' },
    { name:'Russian twists',             target:'3×20', note:'Oblique definition' }
  ]
};

function renderWorkout() {
  // Highlight active split
  document.querySelectorAll('.split-pill').forEach(b => {
    b.classList.toggle('active', b.textContent.toLowerCase() === activeSplit);
  });
  if (!activeSplit) {
    document.getElementById('exercise-list').innerHTML = `<div class="empty-state">
      <div class="empty-icon">🏋️</div>
      <div class="empty-title">Choose your split</div>
      <div class="empty-desc">Select Push, Pull, Legs or Cardio above</div>
    </div>`;
    return;
  }
  renderExerciseList(activeSplit);
}

function selectSplit(split) {
  activeSplit = split;
  document.querySelectorAll('.split-pill').forEach(b => {
    b.classList.toggle('active', b.textContent.toLowerCase()===split);
  });
  renderExerciseList(split);
}

function renderExerciseList(split) {
  const exercises = WORKOUT_PLAN[split] || [];
  const wrap = document.getElementById('exercise-list');
  wrap.innerHTML = exercises.map((ex,ei) => {
    const sets = DB.workoutLog.filter(r=>r.Date===TODAY && r.Exercise===ex.name && r.Split_Name===split);
    return `
      <div class="ex-card" id="ex-card-${ei}">
        <div class="ex-hdr">
          <div>
            <div class="ex-name">${esc(ex.name)}</div>
            <div style="display:flex;gap:6px;margin-top:4px">
              <span class="ex-tag">${ex.target}</span>
            </div>
            <div class="ex-note" style="margin-top:4px">${esc(ex.note)}</div>
          </div>
          <button class="ex-del-btn" onclick="openTimerSheet()" title="Rest timer">
            <i class="fa-solid fa-stopwatch"></i>
          </button>
        </div>
        <div class="set-hdr"><span>#</span><span>Previous</span><span>Weight</span><span>Reps</span><span></span></div>
        ${sets.map((s,si) => `
          <div class="set-row">
            <div class="set-num done">${si+1}</div>
            <div style="font-size:12px;color:var(--t2)">${s.Weight_lbs||0}lb × ${s.Reps||0}</div>
            <input class="set-input" type="number" placeholder="lbs" value="${s.Weight_lbs||''}" onchange="updateSet(${s.ID},'Weight_lbs',this.value)">
            <input class="set-input" type="number" placeholder="reps" value="${s.Reps||''}" onchange="updateSet(${s.ID},'Reps',this.value)">
            <button class="set-del" onclick="deleteWorkoutSet(${s.ID})"><i class="fa-solid fa-xmark"></i></button>
          </div>`).join('')}
        <button class="add-set-btn" onclick="addSet('${esc(ex.name)}','${split}',${ei})">
          <i class="fa-solid fa-plus" style="font-size:11px"></i> Add Set
        </button>
      </div>`;
  }).join('');
  // Timer reminder
  wrap.innerHTML += `<div style="height:20px"></div>`;
}

async function addSet(exName, split, exIdx) {
  // Get last weight/reps as default
  const prev = DB.workoutLog.filter(r=>r.Exercise===exName).slice(-1)[0];
  const weight = prev?.Weight_lbs || '';
  const reps   = prev?.Reps || '';
  const id = nextId(DB.workoutLog);
  const setNo = DB.workoutLog.filter(r=>r.Date===TODAY && r.Exercise===exName).length + 1;
  const row = {
    ID: id, Date: TODAY, Day: dow(TODAY),
    Split_Name: split, Exercise: exName,
    Set_No: setNo, Weight_lbs: weight ? parseFloat(weight) : 0,
    Reps: reps ? parseInt(reps,10) : 0,
    Notes: ''
  };
  DB.workoutLog.push(row);
  saveLocalDB();
  await insertWorkoutRow(row);
  addXP(1);
  renderExerciseList(split);
  renderHome();
  // Haptic
  if (navigator.vibrate) navigator.vibrate(20);
}
function updateSet(id, field, val) {
  const row = DB.workoutLog.find(r=>r.ID===id);
  if (row) { row[field] = field==='Reps' ? parseInt(val,10)||0 : parseFloat(val)||0; saveLocalDB() }
}
async function deleteWorkoutSet(id) {
  await deleteWorkoutRow(id);
  renderExerciseList(activeSplit);
  renderHome();
}
function renderSessionHistory() {
  const el = document.getElementById('session-history-list');
  const sessions = {};
  DB.workoutLog.forEach(r => {
    const key = r.Date+'-'+(r.Split_Name||'');
    if (!sessions[key]) sessions[key] = { date:r.Date, split:r.Split_Name||'', count:0 };
    sessions[key].count++;
  });
  const sorted = Object.values(sessions).sort((a,b)=>b.date.localeCompare(a.date)).slice(0,20);
  if (!sorted.length) { el.innerHTML = `<div style="color:var(--t2);font-size:13px;text-align:center;padding:16px">No sessions yet</div>`; return }
  el.innerHTML = sorted.map(s=>`
    <div class="log-item">
      <div class="log-icon workout"><span>💪</span></div>
      <div class="log-info">
        <div class="log-name">${esc(s.split)||'Workout'}</div>
        <div class="log-meta">${formatDate(s.date)}</div>
      </div>
      <div class="log-val">${s.count} sets</div>
    </div>`).join('');
}

// ─── WATER TRACKING ───────────────────────────────────────
function renderWater() {
  const todayWater = DB.waterLog.filter(r=>r.Date===TODAY).reduce((s,r)=>s+(r.Amount_ml||0),0);
  const pct = Math.min(todayWater/WATER_GOAL_ML*100,100);
  const goalL = WATER_GOAL_ML/1000;

  const dispEl = document.getElementById('water-total-disp');
  const goalEl = document.getElementById('water-goal-disp');
  const barEl  = document.getElementById('water-progress-bar');
  const encEl  = document.getElementById('water-encouragement');

  if (dispEl) dispEl.textContent = todayWater >= 1000 ? fmtDecimal(todayWater/1000) : todayWater;
  if (goalEl) goalEl.textContent = fmtDecimal(goalL);
  if (barEl)  barEl.style.width  = pct+'%';
  if (encEl)  encEl.textContent  = pct>=100 ? '🎉 Goal reached!' : pct>=50 ? 'Halfway there, keep going!' : 'Stay hydrated!';

  const listEl = document.getElementById('water-log-list');
  const today = DB.waterLog.filter(r=>r.Date===TODAY).slice().reverse();
  listEl.innerHTML = today.length ? today.map(r=>`
    <div class="water-log-item">
      <div>💧 ${r.Amount_ml>=1000?fmtDecimal(r.Amount_ml/1000)+'L':r.Amount_ml+'ml'}</div>
      <div class="water-log-time">${r.Timestamp||''}</div>
    </div>`).join('') :
    `<div style="font-size:13px;color:var(--t2);text-align:center;padding:12px 0">No water logged yet</div>`;
}

async function logWater(ml) {
  const id = nextId(DB.waterLog);
  const row = { ID:id, Date:TODAY, Day:dow(TODAY), Amount_ml:ml, Timestamp:nowTime() };
  DB.waterLog.push(row);
  saveLocalDB();
  await insertWaterRow(row);
  renderWater();
  renderHome();
  toast(`+${ml}ml logged`, true);
  if (navigator.vibrate) navigator.vibrate(15);
}

// ─── BODY WEIGHT ──────────────────────────────────────────
async function submitBodyWeight() {
  const kg   = parseFloat(document.getElementById('bw-input')?.value);
  const date = document.getElementById('bw-date-input')?.value || TODAY;
  if (!kg || kg < 20 || kg > 300) { toast('Enter a valid weight', false, true); return }
  const id = nextId(DB.bodyWeightLog);
  const row = { ID:id, Date:date, Weight_kg:kg };
  DB.bodyWeightLog.push(row);
  saveLocalDB();
  await insertBodyWeightRow(row);
  closeAllSheets();
  renderBodyStats();
  toast(`${kg}kg logged`, true);
}
function renderBodyStats() {
  const log = DB.bodyWeightLog.slice().sort((a,b)=>a.Date.localeCompare(b.Date));
  const latest = log[log.length-1];
  const prev   = log[log.length-2];

  const curEl = document.getElementById('body-weight-current');
  const chgEl = document.getElementById('body-weight-change');
  if (curEl) curEl.textContent = latest ? fmtDecimal(latest.Weight_kg) : '—';
  if (chgEl) {
    if (latest && prev) {
      const diff = +(latest.Weight_kg - prev.Weight_kg).toFixed(1);
      chgEl.textContent = (diff>0?'+':'')+diff+'kg since last log';
      chgEl.style.color = diff>0?'var(--red)':'var(--teal)';
    } else chgEl.textContent = 'First weigh-in';
  }

  // Weight chart (last 10 entries)
  const chartEl  = document.getElementById('weight-chart');
  const labelEl  = document.getElementById('weight-chart-labels');
  const recent   = log.slice(-10);
  if (!recent.length) {
    if (chartEl) chartEl.innerHTML='<div style="font-size:12px;color:var(--t2);padding:8px">No data</div>';
    if (labelEl) labelEl.innerHTML='';
    return;
  }
  const maxW = Math.max(...recent.map(r=>r.Weight_kg));
  const minW = Math.min(...recent.map(r=>r.Weight_kg));
  const range = maxW - minW || 1;
  if (chartEl) chartEl.innerHTML = recent.map(r=>{
    const h = Math.max(4, Math.round(((r.Weight_kg-minW)/range)*54)+6);
    return `<div class="weight-bar-col"><div class="weight-bar" style="height:${h}px"></div></div>`;
  }).join('');
  if (labelEl) labelEl.innerHTML = recent.map(r=>`<div style="font-size:9px;color:var(--t3);flex:1;text-align:center">${r.Date.slice(5)}</div>`).join('');

  // Weight log table
  const logEl = document.getElementById('body-weight-log');
  if (logEl) {
    logEl.innerHTML = log.slice().reverse().slice(0,20).map(r=>`
      <div class="flex-between" style="padding:9px 0;border-bottom:1px solid var(--b1)">
        <div style="font-size:14px">${formatDate(r.Date)}</div>
        <div style="font-size:14px;font-weight:600">${fmtDecimal(r.Weight_kg)} kg</div>
      </div>`).join('') || `<div style="font-size:13px;color:var(--t2);text-align:center;padding:12px">No weigh-ins yet</div>`;
  }
}

// ─── CALORIE GOAL ─────────────────────────────────────────
function setGoalPreset(v) {
  const el = document.getElementById('goal-input');
  if (el) el.value = v;
}
function saveGoal() {
  const v = parseInt(document.getElementById('goal-input')?.value||'2400',10);
  if (!v||v<500||v>9999) { toast('Enter 500–9999', false, true); return }
  GOAL = v;
  localStorage.setItem('forge_goal', String(v));
  document.getElementById('goal-setting-desc').textContent = `${fmt(GOAL)} kcal / day`;
  closeAllSheets();
  renderNutrition(); renderHome();
  toast(`Goal set to ${fmt(GOAL)} kcal`, true);
}

// ─── RUN TRACKING ─────────────────────────────────────────
function renderRunIdle() {
  const lastRun = DB.runLog[0];
  if (lastRun) {
    document.getElementById('run-last-stats').style.display='grid';
    document.getElementById('last-dist').textContent = fmtDecimal(lastRun.Distance_Km||0)+'km';
    document.getElementById('last-pace').textContent = formatPace(lastRun.Avg_Pace||0)+' /km';
    document.getElementById('last-dur').textContent  = formatDuration(lastRun.Duration_Sec||0);
    document.getElementById('run-last-content').innerHTML = `<div style="font-size:13px;color:var(--t2)">${formatDate(lastRun.Date)}</div>`;
  }

  const histEl = document.getElementById('run-history-list');
  histEl.innerHTML = DB.runLog.slice(0,5).map(r=>`
    <div class="card" style="margin-bottom:8px;padding:12px 14px">
      <div class="flex-between">
        <div>
          <div style="font-size:14px;font-weight:600">${fmtDecimal(r.Distance_Km||0)}km Run</div>
          <div style="font-size:12px;color:var(--t2)">${formatDate(r.Date)}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:14px;font-weight:600;color:var(--teal)">${formatDuration(r.Duration_Sec||0)}</div>
          <div style="font-size:11px;color:var(--t2)">${formatPace(r.Avg_Pace||0)} /km</div>
        </div>
      </div>
    </div>`).join('');
}

function ensureRunMap() {
  if (!window.L) return null;
  if (runMap) return runMap;
  const el = document.getElementById('run-map');
  if (!el) return null;
  runMap = L.map(el, {
    zoomControl:false, attributionControl:false,
    dragging:false, touchZoom:false, scrollWheelZoom:false, doubleClickZoom:false
  }).setView([-33.8688,151.2093],15);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', { maxZoom:19, subdomains:'abcd' }).addTo(runMap);
  return runMap;
}

function startRun() {
  if (!runGpsEnabled) { toast('GPS is off', false, true); return }
  const idle   = document.getElementById('run-idle-view');
  const active = document.getElementById('run-active-view');
  idle.style.display   = 'none';
  active.style.display = 'flex';
  document.getElementById('run-status-txt').textContent = 'GPS tracking…';

  runPoints=[]; runDistKm=0; runActive=true; runPaused=false;
  runStartedAt = Date.now(); runPausedAccumMs=0; runPauseBeganAt=null;

  const map = ensureRunMap();

  if (map) {
    setTimeout(()=>{ try{ map.invalidateSize() }catch(e){} }, 300);
  }

  if (navigator.geolocation && runGpsEnabled) {
    runWatchId = navigator.geolocation.watchPosition(
      pos => onRunPosition(pos, map),
      err => { document.getElementById('run-status-txt').textContent = 'GPS error: '+err.message },
      { enableHighAccuracy:true, maximumAge:2000, timeout:10000 }
    );
  }

  runTickInterval = setInterval(updateRunHUD, 1000);
  toast('Run started!', true);
}

function onRunPosition(pos, map) {
  if (!runActive || runPaused) return;
  const { latitude:lat, longitude:lng, accuracy } = pos.coords;
  if (accuracy > 50) return;
  const pt = [lat, lng];

  if (runPoints.length > 0) {
    const last = runPoints[runPoints.length-1];
    runDistKm += haversineKm(last[0],last[1],lat,lng);
  }
  runPoints.push(pt);

  if (map) {
    if (runPolyline) map.removeLayer(runPolyline);
    runPolyline = L.polyline(runPoints, { color:'#B8FF00', weight:4, opacity:.85 }).addTo(map);
    map.setView(pt, Math.max(map.getZoom()||15,15));
  }
}

function haversineKm(lat1,lng1,lat2,lng2) {
  const R=6371, dLat=(lat2-lat1)*Math.PI/180, dLng=(lng2-lng1)*Math.PI/180;
  const a=Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLng/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}
function getRunElapsedSec() {
  if (!runStartedAt) return 0;
  const elapsed = Date.now() - runStartedAt - runPausedAccumMs;
  return Math.max(0, Math.floor(elapsed/1000));
}
function updateRunHUD() {
  if (!runActive) return;
  const sec = getRunElapsedSec();
  const pace = runDistKm>0 ? sec/runDistKm : 0;
  document.getElementById('hud-distance').textContent = fmtDecimal(runDistKm,2);
  document.getElementById('hud-duration').textContent = formatDuration(sec);
  document.getElementById('hud-pace').textContent     = formatPace(pace);
}
function toggleRunPause() {
  runPaused = !runPaused;
  const btn = document.getElementById('run-pause-btn');
  if (runPaused) {
    runPauseBeganAt = Date.now();
    if (btn) btn.textContent = 'Resume';
  } else {
    if (runPauseBeganAt) runPausedAccumMs += Date.now()-runPauseBeganAt;
    runPauseBeganAt=null;
    if (btn) btn.textContent = 'Pause';
  }
}
function confirmStopRun() {
  if (!confirm('Finish this run?')) return;
  stopRun();
}
async function stopRun() {
  clearInterval(runTickInterval);
  if (runWatchId!=null) navigator.geolocation.clearWatch(runWatchId); runWatchId=null;
  runActive=false;

  const sec   = getRunElapsedSec();
  const pace  = runDistKm>0 ? sec/runDistKm : 0;
  const steps = Math.round(runDistKm*1350);
  const cals  = Math.round(runDistKm*70);

  if (runDistKm > 0.05) {
    const id = nextId(DB.runLog);
    const row = {
      ID:id, Date:TODAY, Day:dow(TODAY),
      Start_Timestamp: new Date(runStartedAt).toISOString(),
      End_Timestamp: new Date().toISOString(),
      Duration_Sec: sec, Distance_Km: +fmtDecimal(runDistKm,3),
      Avg_Pace: +fmtDecimal(pace,1), Steps: steps, Calories: cals,
      Path_JSON: JSON.stringify(runPoints)
    };
    DB.runLog.unshift(row);
    saveLocalDB();
    if (supabaseClient && currentUser) {
      await supabaseClient.from('run_log').insert({...row, user_id:currentUser.id}).catch(()=>{});
    }
    addXP(10); updateStreak();
    toast(`Run saved: ${fmtDecimal(runDistKm,2)}km`, true);
    spawnConfetti();
  }

  // Back to idle
  const idle   = document.getElementById('run-idle-view');
  const active = document.getElementById('run-active-view');
  idle.style.display   = '';
  active.style.display = 'none';
  if (runMap && runPolyline) { runMap.removeLayer(runPolyline); runPolyline=null }
  runDistKm=0; runPoints=[]; runStartedAt=null;
  renderRunIdle();
  renderHome();
}
function toggleGPS() {
  runGpsEnabled = !runGpsEnabled;
  const btn = document.getElementById('gps-toggle-btn');
  if (btn) btn.style.color = runGpsEnabled ? 'var(--lime)' : 'var(--t3)';
  toast('GPS '+(runGpsEnabled?'on':'off'), true);
}

// ─── STATS / PROGRESS ─────────────────────────────────────
function renderStats() {
  if (CUR_STAT_TAB === 'week')    renderWeekStats();
  if (CUR_STAT_TAB === 'body')    renderBodyStats();
  if (CUR_STAT_TAB === 'routine') renderRoutine();
}

function renderWeekStats() {
  const days7 = Array.from({length:7},(_,i)=>datePlusDays(TODAY,-6+i));

  // Weekly cals / training days
  let totalCal=0, trainDays=0;
  const dayData = days7.map(d=>{
    const cals = DB.calLog.filter(r=>r.Date===d).reduce((s,r)=>s+(r.Calories_kcal||0),0);
    const sets  = DB.workoutLog.filter(r=>r.Date===d).length;
    if (cals>0||sets>0) totalCal+=cals;
    if (sets>0) trainDays++;
    return { d, cals, sets };
  });
  const avgCal = Math.round(totalCal / days7.filter(d=>DB.calLog.some(r=>r.Date===d)).length) || 0;
  document.getElementById('wk-avg-cal').textContent   = avgCal ? fmt(avgCal) : '—';
  document.getElementById('wk-training-days').textContent = trainDays;
  document.getElementById('wk-total-cal').textContent  = fmt(totalCal)+' kcal';

  // Calorie bar chart
  const maxCal = Math.max(...dayData.map(d=>d.cals), 1);
  const calChartEl  = document.getElementById('cal-bar-chart');
  const calLabelEl  = document.getElementById('cal-bar-labels');
  if (calChartEl) calChartEl.innerHTML = dayData.map(dd=>{
    const h = Math.max(4, Math.round((dd.cals/maxCal)*72));
    const isToday = dd.d===TODAY;
    return `<div class="bar-col">
      <div class="bar ${isToday?'today':''}" style="height:80px">
        <div class="bar-fill amber ${isToday?'lime':''}" style="height:${h}px"></div>
      </div>
    </div>`;
  }).join('');
  if (calLabelEl) calLabelEl.innerHTML = dayData.map(dd=>
    `<div style="flex:1;text-align:center;font-size:9px;color:${dd.d===TODAY?'var(--lime)':'var(--t3)'}">${dow(dd.d)}</div>`
  ).join('');

  // Workout bar chart
  const maxSets = Math.max(...dayData.map(d=>d.sets), 1);
  const workChartEl = document.getElementById('work-bar-chart');
  const workLabelEl = document.getElementById('work-bar-labels');
  if (workChartEl) workChartEl.innerHTML = dayData.map(dd=>{
    const h = Math.max(4, Math.round((dd.sets/maxSets)*72));
    return `<div class="bar-col">
      <div class="bar" style="height:80px">
        <div class="bar-fill blue" style="height:${h}px"></div>
      </div>
    </div>`;
  }).join('');
  if (workLabelEl) workLabelEl.innerHTML = dayData.map(dd=>
    `<div style="flex:1;text-align:center;font-size:9px;color:var(--t3)">${dow(dd.d)}</div>`
  ).join('');

  // Heatmap (30 days)
  const grid = document.getElementById('heatmap-grid');
  if (grid) {
    const days30 = Array.from({length:30},(_,i)=>datePlusDays(TODAY,-29+i));
    grid.innerHTML = days30.map(d=>{
      const hasCal  = DB.calLog.some(r=>r.Date===d);
      const hasWork = DB.workoutLog.some(r=>r.Date===d);
      const cls = hasCal&&hasWork?'has-both':hasCal?'has-cal':hasWork?'has-work':'';
      const isT = d===TODAY?'today':'';
      return `<div class="cal-cell ${cls} ${isT}" title="${d}"></div>`;
    }).join('');
  }

  // Run list
  const runListEl = document.getElementById('stats-run-list');
  if (runListEl) {
    if (!DB.runLog.length) {
      runListEl.innerHTML = `<div style="color:var(--t2);font-size:13px;text-align:center;padding:12px 0">No runs yet</div>`;
    } else {
      runListEl.innerHTML = DB.runLog.slice(0,5).map(r=>`
        <div class="card" style="margin-bottom:8px;padding:12px 14px">
          <div class="flex-between">
            <div>
              <div style="font-size:14px;font-weight:600">${fmtDecimal(r.Distance_Km||0,2)}km</div>
              <div style="font-size:12px;color:var(--t2)">${formatDate(r.Date)}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:14px;font-weight:600;color:var(--teal)">${formatDuration(r.Duration_Sec||0)}</div>
              <div style="font-size:11px;color:var(--t2)">${formatPace(r.Avg_Pace||0)} /km</div>
            </div>
          </div>
        </div>`).join('');
    }
  }
}

// ─── ROUTINE ──────────────────────────────────────────────
const WEEKLY_ROUTINE = {
  mon: { tag:'Free day', blocks:[
    {id:'mon-1130pm',time:'11:30 PM',type:'sleep',title:'Bedtime (prev Sun)',subtitle:'target 8 hrs'},
    {id:'mon-730am', time:'7:30 AM', type:'wake', title:'Wake up',subtitle:'8 hrs sleep'},
    {id:'mon-745am', time:'7:45 AM', type:'meal', title:'Pre-workout meal — rice + chicken',subtitle:''},
    {id:'mon-815am', time:'8:15 AM', type:'cardio',title:'Cardio — 30 min treadmill/bike',subtitle:''},
    {id:'mon-850am', time:'8:50 AM', type:'gym',  title:'Abs circuit — 15 min',subtitle:''},
    {id:'mon-930am', time:'9:30 AM', type:'meal', title:'Post-workout shake (NitroTech + banana)',subtitle:''},
    {id:'mon-100pm', time:'1:00 PM', type:'meal', title:'Lunch — chicken curry + rice',subtitle:''},
    {id:'mon-430pm', time:'4:30 PM', type:'meal', title:'Snack — Greek yogurt + nuts',subtitle:''},
    {id:'mon-730pm', time:'7:30 PM', type:'meal', title:'Dinner — chicken + roti + dal',subtitle:''},
    {id:'mon-1100pm',time:'11:00 PM',type:'sleep',title:'Bedtime',subtitle:'target 8 hrs'}]},
  tue: { tag:'Work 4:30–9:30 PM', blocks:[
    {id:'tue-700am', time:'7:00 AM', type:'wake', title:'Wake up',subtitle:'8 hrs sleep'},
    {id:'tue-715am', time:'7:15 AM', type:'meal', title:'Pre-workout — oats + NitroTech shake',subtitle:''},
    {id:'tue-800am', time:'8:00 AM', type:'gym',  title:'Gym — Pull day (back + biceps) ~65 min',subtitle:''},
    {id:'tue-915am', time:'9:15 AM', type:'meal', title:'Post-workout shake',subtitle:''},
    {id:'tue-1230pm',time:'12:30 PM',type:'meal', title:'Lunch — chicken rice + veggies',subtitle:''},
    {id:'tue-330pm', time:'3:30 PM', type:'meal', title:'Pre-work meal — packed chicken box',subtitle:''},
    {id:'tue-430pm', time:'4:30 PM', type:'work', title:'Work shift starts',subtitle:''},
    {id:'tue-930pm', time:'9:30 PM', type:'work', title:'Work ends',subtitle:''},
    {id:'tue-1030pm',time:'10:30 PM',type:'meal', title:'Late dinner — light + protein',subtitle:''},
    {id:'tue-1100pm',time:'11:00 PM',type:'sleep',title:'Bedtime',subtitle:'8 hrs planned'}]},
  wed: { tag:'Work 12–9 PM', blocks:[
    {id:'wed-700am', time:'7:00 AM', type:'wake', title:'Wake up',subtitle:''},
    {id:'wed-715am', time:'7:15 AM', type:'meal', title:'Breakfast — oats + NitroTech shake',subtitle:''},
    {id:'wed-800am', time:'8:00 AM', type:'gym',  title:'Gym — Legs day ~75 min',subtitle:''},
    {id:'wed-930am', time:'9:30 AM', type:'meal', title:'Post-workout shake',subtitle:''},
    {id:'wed-1200pm',time:'12:00 PM',type:'work', title:'Work starts',subtitle:''},
    {id:'wed-300pm', time:'3:00 PM', type:'meal', title:'Work lunch — packed box',subtitle:''},
    {id:'wed-900pm', time:'9:00 PM', type:'work', title:'Work ends',subtitle:''},
    {id:'wed-930pm', time:'9:30 PM', type:'meal', title:'Dinner at home',subtitle:''},
    {id:'wed-1130pm',time:'11:30 PM',type:'sleep',title:'Bedtime',subtitle:'7.5 hrs'}]},
  thu: { tag:'Work 9–5 PM', blocks:[
    {id:'thu-700am', time:'7:00 AM', type:'wake', title:'Wake up',subtitle:''},
    {id:'thu-715am', time:'7:15 AM', type:'meal', title:'Breakfast — rice + egg + NitroTech',subtitle:''},
    {id:'thu-800am', time:'8:00 AM', type:'gym',  title:'Gym — Push day ~65 min',subtitle:''},
    {id:'thu-920am', time:'9:20 AM', type:'meal', title:'Post-workout shake',subtitle:''},
    {id:'thu-1230pm',time:'12:30 PM',type:'meal', title:'Work lunch — packed chicken box',subtitle:''},
    {id:'thu-500pm', time:'5:00 PM', type:'work', title:'Work ends',subtitle:''},
    {id:'thu-600pm', time:'6:00 PM', type:'meal', title:'Dinner — chicken + rice + salad',subtitle:''},
    {id:'thu-1130pm',time:'11:30 PM',type:'sleep',title:'Bedtime',subtitle:'8 hrs'}]},
  fri: { tag:'Work 9–3 PM', blocks:[
    {id:'fri-730am', time:'7:30 AM', type:'wake', title:'Wake up',subtitle:'8.5 hrs sleep'},
    {id:'fri-745am', time:'7:45 AM', type:'meal', title:'Breakfast — oats + NitroTech shake',subtitle:''},
    {id:'fri-900am', time:'9:00 AM', type:'work', title:'Work starts',subtitle:''},
    {id:'fri-1200pm',time:'12:00 PM',type:'meal', title:'Work lunch — packed chicken box',subtitle:''},
    {id:'fri-300pm', time:'3:00 PM', type:'work', title:'Work ends',subtitle:''},
    {id:'fri-330pm', time:'3:30 PM', type:'gym',  title:'Gym — Pull day + abs ~75 min',subtitle:'best session of week'},
    {id:'fri-500pm', time:'5:00 PM', type:'meal', title:'Post-gym shake + snack',subtitle:''},
    {id:'fri-800pm', time:'8:00 PM', type:'meal', title:'Dinner — chicken biriyani-style',subtitle:''},
    {id:'fri-1030pm',time:'10:30 PM',type:'sleep',title:'Bedtime',subtitle:'7 hrs'}]},
  sat: { tag:'Work 6 AM–3 PM', blocks:[
    {id:'sat-530am', time:'5:30 AM', type:'wake', title:'Wake up — quick meal, no gym',subtitle:''},
    {id:'sat-545am', time:'5:45 AM', type:'meal', title:'Early meal — banana + NitroTech shake',subtitle:''},
    {id:'sat-600am', time:'6:00 AM', type:'work', title:'Work starts (long shift)',subtitle:''},
    {id:'sat-1000am',time:'10:00 AM',type:'meal', title:'Mid-shift meal — packed chicken box',subtitle:''},
    {id:'sat-100pm', time:'1:00 PM', type:'meal', title:'Packed snack — yogurt + nuts',subtitle:''},
    {id:'sat-300pm', time:'3:00 PM', type:'work', title:'Work ends',subtitle:''},
    {id:'sat-600pm', time:'6:00 PM', type:'meal', title:'Big recovery dinner — chicken curry + rice',subtitle:''},
    {id:'sat-1000pm',time:'10:00 PM',type:'sleep',title:'Bedtime',subtitle:'9 hrs'}]},
  sun: { tag:'Free + prep day', blocks:[
    {id:'sun-700am', time:'7:00 AM', type:'wake', title:'Wake up — rest day',subtitle:''},
    {id:'sun-730am', time:'7:30 AM', type:'meal', title:'Big breakfast — eggs + oats + shake',subtitle:''},
    {id:'sun-1030am',time:'10:30 AM',type:'cardio',title:'Long walk or light jog outdoors',subtitle:''},
    {id:'sun-1200pm',time:'12:00 PM',type:'meal', title:'Lunch — chicken wrap or bowl',subtitle:''},
    {id:'sun-200pm', time:'2:00 PM', type:'rest', title:'Meal prep — cook for the week',subtitle:''},
    {id:'sun-600pm', time:'6:00 PM', type:'meal', title:'Dinner — something different, treat',subtitle:''},
    {id:'sun-1000pm',time:'10:00 PM',type:'sleep',title:'Bedtime',subtitle:'target 8+ hrs'}]}
};

function getTodayKey() {
  return ['sun','mon','tue','wed','thu','fri','sat'][new Date().getDay()];
}
function parseTime(str) {
  const [time,mod] = str.split(' ');
  let [h,m] = time.split(':').map(Number);
  if (h===12) h=0;
  if (mod==='PM') h+=12;
  return h*60+m;
}
function renderRoutineStrip() {
  const key = getTodayKey();
  const plan = WEEKLY_ROUTINE[key];
  const el   = document.getElementById('routine-strip');
  if (!el) return;
  const now  = new Date().getHours()*60+new Date().getMinutes();
  const upcoming = plan.blocks.filter(b=>Math.abs(parseTime(b.time)-now)<=120 && !completedBlocks.has(b.id)).slice(0,3);
  if (!upcoming.length) {
    el.innerHTML = `<div style="font-size:13px;color:var(--t2)">All blocks done ✓</div>`; return;
  }
  el.innerHTML = upcoming.map(b=>`
    <div style="display:flex;align-items:center;gap:8px;padding:4px 0">
      <div class="t-dot d-${b.type}" style="position:static;flex-shrink:0"></div>
      <div style="font-size:13px;flex:1">${esc(b.title)}</div>
      <div style="font-size:11px;color:var(--t2)">${b.time}</div>
    </div>`).join('');
}
function renderRoutine() {
  const key  = getTodayKey();
  const plan = WEEKLY_ROUTINE[key];
  if (!plan) return;

  const dayLbl = document.getElementById('routine-day-label');
  const dayDsc = document.getElementById('routine-day-desc');
  if (dayLbl) dayLbl.textContent = key.charAt(0).toUpperCase()+key.slice(1)+' Routine';
  if (dayDsc) dayDsc.textContent = plan.tag;

  if (new Date().getDay()===0) {
    const prepCard = document.getElementById('sunday-prep-card');
    if (prepCard) prepCard.style.display='block';
  }

  const now = new Date().getHours()*60+new Date().getMinutes();
  const wrap = document.getElementById('routine-timeline');
  if (!wrap) return;

  wrap.innerHTML = plan.blocks.map(b=>{
    const done = completedBlocks.has(b.id);
    const active = Math.abs(parseTime(b.time)-now)<=120;
    return `
      <div class="t-row ${done?'done':''} ${active?'active':''}" id="trow-${b.id}">
        <div class="t-time">${b.time}</div>
        <div class="t-dot d-${b.type}"></div>
        <div class="t-content" onclick="routineBlockClick('${b.id}','${b.type}','${esc(b.title)}')">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
            <div>
              <div class="t-title-txt">${esc(b.title)}</div>
              ${b.subtitle?`<div class="t-sub-txt">${esc(b.subtitle)}</div>`:''}
            </div>
            <button class="t-check-btn" onclick="toggleBlock(event,'${b.id}')">
              <i class="fa-solid fa-check"></i>
            </button>
          </div>
        </div>
      </div>`;
  }).join('');
}

function routineBlockClick(id, type, title) {
  if (type==='meal') { navTo('log'); switchLogTab('nutrition'); toast('Ready to log: '+title, true) }
  else if (type==='gym'||type==='cardio') { navTo('log'); switchLogTab('workout') }
  else toggleBlock({stopPropagation:()=>{}}, id);
}
function toggleBlock(e, id) {
  e.stopPropagation();
  if (completedBlocks.has(id)) completedBlocks.delete(id);
  else { completedBlocks.add(id); if(navigator.vibrate) navigator.vibrate(20) }
  localStorage.setItem('forge_blocks_'+TODAY, JSON.stringify([...completedBlocks]));
  renderRoutine();
  renderRoutineStrip();
}

// ─── PROFILE ──────────────────────────────────────────────
function renderProfile() {
  const name  = currentUser?.email?.split('@')[0] || 'Guest';
  const email = currentUser?.email || 'Not signed in';
  const initial = name.charAt(0).toUpperCase();

  document.getElementById('profile-avatar').textContent = initial;
  document.getElementById('profile-name').textContent   = name;
  document.getElementById('profile-email').textContent  = email;
  document.getElementById('profile-level').innerHTML    = `<i class="fa-solid fa-bolt" style="font-size:10px"></i> Level ${userStats.level}`;
  document.getElementById('xp-current').textContent     = fmt(userStats.xp)+' XP';
  document.getElementById('xp-next').textContent        = 'Next: '+fmt(userStats.level*XP_PER_LEVEL)+' XP';
  const xpPct = ((userStats.xp % XP_PER_LEVEL) / XP_PER_LEVEL * 100);
  document.getElementById('xp-bar').style.width = xpPct+'%';
  document.getElementById('profile-streak-num').textContent = userStats.streak;
  document.getElementById('profile-total-workouts').textContent = [...new Set(DB.workoutLog.map(r=>r.Date+r.Split_Name))].length;
  document.getElementById('profile-total-runs').textContent  = DB.runLog.length;
  document.getElementById('goal-setting-desc').textContent   = `${fmt(GOAL)} kcal / day`;

  fetchLeaderboard();
}

// ─── LEADERBOARD ──────────────────────────────────────────
async function fetchLeaderboard() {
  if (!supabaseClient || !currentUser) return;
  try {
    const { data:friends } = await supabaseClient.from('friends').select('friend_id').eq('user_id', currentUser.id);
    friendsList = friends||[];
    const uids = [currentUser.id, ...(friends||[]).map(f=>f.friend_id)];
    const { data:stats } = await supabaseClient.from('user_stats').select('user_id,email,xp,level,streak').in('user_id',uids).order('xp',{ascending:false});
    const el = document.getElementById('lb-list');
    if (!el) return;
    if (!stats?.length) { el.innerHTML=`<div style="font-size:13px;color:var(--t2);text-align:center;padding:12px">No data yet</div>`; return }
    const medals=['🥇','🥈','🥉'];
    el.innerHTML = stats.map((s,i)=>{
      const isMe = s.user_id===currentUser.id;
      const name = isMe?'YOU':(s.email?.split('@')[0]?.toUpperCase()||'Friend');
      return `
        <div class="lb-row">
          <div class="lb-rank ${isMe?'me':''}">${medals[i]||i+1}</div>
          <div class="lb-info">
            <div class="lb-name ${isMe?'me':''}">${esc(name)}</div>
            <div class="lb-sub">🔥 ${s.streak} day streak · Lv${s.level}</div>
          </div>
          <div class="lb-xp">${fmt(s.xp)}<small> XP</small></div>
        </div>`;
    }).join('');
  } catch(e){ console.warn('Leaderboard error',e) }
}

async function addFriend() {
  const input = document.getElementById('friend-email-input');
  const email = input?.value?.trim();
  if (!email||!email.includes('@')) { toast('Enter a valid email', false, true); return }
  if (!supabaseClient||!currentUser) { toast('Sign in to add friends', false, true); return }
  const { data:user } = await supabaseClient.from('user_stats').select('user_id').eq('email',email).single();
  if (!user) { toast('User not found', false, true); return }
  await supabaseClient.from('friends').upsert({ user_id:currentUser.id, friend_id:user.user_id }, { onConflict:'user_id,friend_id' });
  if (input) input.value='';
  toast('Friend added!', true);
  fetchLeaderboard();
}

// ─── REST TIMER ───────────────────────────────────────────
function setTimerPreset(sec) {
  const m=Math.floor(sec/60), s=sec%60;
  const mEl=document.getElementById('timer-min'), sEl=document.getElementById('timer-sec');
  if(mEl) mEl.value=m;
  if(sEl) sEl.value=s;
  timerRemainMs = sec*1000;
  updateTimerDisplay(sec);
  document.getElementById('timer-status-txt').textContent = 'Ready';
}
function toggleRestTimer() {
  const btn = document.getElementById('timer-toggle-btn');
  if (timerInterval) {
    clearInterval(timerInterval); timerInterval=null;
    if(btn) btn.textContent='Resume';
    document.getElementById('timer-display')?.classList.remove('running');
    document.getElementById('timer-status-txt').textContent = 'Paused';
  } else {
    const m = parseInt(document.getElementById('timer-min')?.value||'1',10);
    const s = parseInt(document.getElementById('timer-sec')?.value||'30',10);
    if (!timerRemainMs || timerRemainMs<=0) timerRemainMs = (m*60+s)*1000;
    timerEndAt = Date.now() + timerRemainMs;
    if(btn) btn.textContent='Pause';
    document.getElementById('timer-display')?.classList.add('running');
    document.getElementById('timer-status-txt').textContent = 'Running…';
    timerInterval = setInterval(timerTick, 200);
  }
}
function timerTick() {
  timerRemainMs = Math.max(0, timerEndAt - Date.now());
  updateTimerDisplay(Math.ceil(timerRemainMs/1000));
  if (timerRemainMs <= 0) {
    clearInterval(timerInterval); timerInterval=null;
    if(navigator.vibrate) navigator.vibrate([200,100,200,100,200]);
    document.getElementById('timer-display')?.classList.remove('running');
    document.getElementById('timer-status-txt').textContent = '✓ Rest done — go!';
    document.getElementById('timer-toggle-btn').textContent='Start';
    toast('Rest done — go!', true);
  }
}
function resetRestTimer() {
  clearInterval(timerInterval); timerInterval=null; timerRemainMs=0;
  updateTimerDisplay(0);
  document.getElementById('timer-display')?.classList.remove('running');
  document.getElementById('timer-status-txt').textContent='Ready';
  document.getElementById('timer-toggle-btn').textContent='Start';
}
function updateTimerDisplay(sec) {
  const m=Math.floor(sec/60),s=sec%60;
  const el=document.getElementById('timer-display');
  if(el) el.textContent=pad(m)+':'+pad(s);
}

// ─── APPLE HEALTH IMPORT ──────────────────────────────────
function importAppleHealth(e) {
  const file = e.target.files[0];
  if (!file) return;
  const statusEl = document.getElementById('health-import-status');
  if (statusEl) { statusEl.style.display='block'; statusEl.textContent='Parsing Apple Health export…' }

  const reader = new FileReader();
  reader.onload = async (evt) => {
    const xml = evt.target.result;
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xml, 'application/xml');

      const records = doc.querySelectorAll('Record');
      let calCount=0, weightCount=0, waterCount=0;

      records.forEach(rec => {
        const type  = rec.getAttribute('type') || '';
        const value = parseFloat(rec.getAttribute('value') || '0');
        const startDate = (rec.getAttribute('startDate')||'').slice(0,10);
        if (!startDate || isNaN(value)) return;

        if (type === 'HKQuantityTypeIdentifierDietaryEnergyConsumed') {
          const cals = Math.round(type.includes('kJ') ? value/4.184 : value);
          if (cals>0 && cals<5000) {
            const id = nextId(DB.calLog);
            DB.calLog.push({ ID:id, Date:startDate, Day:dow(startDate), Course:'Meal',
              Calories_kcal:cals, Timestamp:'12:00', Notes:'Apple Health import', Protein_g:0, Carbs_g:0, Fat_g:0 });
            calCount++;
          }
        } else if (type === 'HKQuantityTypeIdentifierBodyMass') {
          const kg = type.includes('lb') ? value*0.453592 : value;
          if (kg>20 && kg<300) {
            const id = nextId(DB.bodyWeightLog);
            DB.bodyWeightLog.push({ ID:id, Date:startDate, Weight_kg:+fmtDecimal(kg,1) });
            weightCount++;
          }
        } else if (type === 'HKQuantityTypeIdentifierDietaryWater') {
          const ml = type.includes('L') ? value*1000 : value;
          if (ml>0 && ml<5000) {
            const id = nextId(DB.waterLog);
            DB.waterLog.push({ ID:id, Date:startDate, Day:dow(startDate), Amount_ml:Math.round(ml), Timestamp:'12:00' });
            waterCount++;
          }
        }
      });

      saveLocalDB();
      if (statusEl) statusEl.textContent = `✓ Imported: ${calCount} cal entries, ${weightCount} weigh-ins, ${waterCount} water logs`;
      renderHome(); renderNutrition(); renderBodyStats();
      toast(`Apple Health synced!`, true);
    } catch(err) {
      console.warn('Apple Health import error', err);
      if (statusEl) statusEl.textContent = 'Error parsing file. Make sure it is export.xml from Apple Health.';
    }
  };
  reader.readAsText(file);
}

// ─── DATA EXPORT / IMPORT ─────────────────────────────────
function exportData() {
  try {
    const wb = XLSX.utils.book_new();
    const addSheet = (name, data) => {
      if (data.length) XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(data), name);
    };
    addSheet('CalorieLog',   DB.calLog);
    addSheet('WorkoutLog',   DB.workoutLog);
    addSheet('RunLog',       DB.runLog);
    addSheet('WaterLog',     DB.waterLog);
    addSheet('BodyWeight',   DB.bodyWeightLog);
    XLSX.writeFile(wb, 'forge_data_v5.xlsx');
    toast('Data exported!', true);
  } catch(e) { toast('Export failed', false, true) }
}
function importXLSX(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = evt => {
    try {
      const wb = XLSX.read(evt.target.result, { type:'binary' });
      const parse = name => {
        const sh = wb.Sheets[name];
        return sh ? XLSX.utils.sheet_to_json(sh) : [];
      };
      DB.calLog       = parse('CalorieLog');
      DB.workoutLog   = parse('WorkoutLog');
      DB.runLog       = parse('RunLog');
      DB.waterLog     = parse('WaterLog') || parse('Water');
      DB.bodyWeightLog= parse('BodyWeight');
      saveLocalDB();
      renderHome(); renderNutrition();
      closeAllSheets();
      toast('Data imported!', true);
    } catch(e) { toast('Import failed — check file', false, true) }
  };
  reader.readAsBinaryString(file);
}
function confirmClearData() {
  if (!confirm('Clear ALL data? This cannot be undone.')) return;
  DB = { calLog:[], workoutLog:[], runLog:[], waterLog:[], bodyWeightLog:[] };
  saveLocalDB();
  renderHome(); renderNutrition();
  closeAllSheets();
  toast('All data cleared', true);
}

// ─── SWIPE TO DELETE ──────────────────────────────────────
let swipeStartX=0, swipeEl=null;
function swipeStart(e, id) {
  swipeStartX = e.touches[0].clientX;
  swipeEl = id;
}
function swipeMove(e, id) {
  if (swipeEl!==id) return;
  const dx = e.touches[0].clientX - swipeStartX;
  const el = document.getElementById(id);
  if (dx<-30&&el) el.classList.add('reveal-del');
  else if (dx>10&&el) el.classList.remove('reveal-del');
}
function swipeEnd(e, id) { swipeEl=null }

// ─── CONFETTI ─────────────────────────────────────────────
function spawnConfetti() {
  const layer = document.getElementById('confetti-layer');
  if (!layer) return;
  const colors = ['#B8FF00','#4DA8FF','#FF5FA8','#FFB347','#00FFBF'];
  for (let i=0;i<40;i++) {
    const p = document.createElement('div');
    p.className = 'confetti-piece';
    p.style.cssText = `left:${Math.random()*100}vw;background:${colors[Math.floor(Math.random()*colors.length)]};width:${6+Math.random()*6}px;height:${6+Math.random()*6}px;animation-duration:${1.5+Math.random()*2}s;animation-delay:${Math.random()*0.5}s`;
    layer.appendChild(p);
    setTimeout(()=>p.remove(), 4000);
  }
}

// ─── ESCAPE HTML ──────────────────────────────────────────
function esc(str) {
  return String(str||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─── PERIODIC REFRESH ─────────────────────────────────────
setInterval(()=>{
  if (CUR_SCREEN==='home') renderHome();
  if (CUR_SCREEN==='stats'&&CUR_STAT_TAB==='routine') renderRoutineStrip();
}, 60000);

// ─── INIT ─────────────────────────────────────────────────
function initApp() {
  TODAY = todayStr();
  loadLocalDB();
  completedBlocks = new Set(JSON.parse(localStorage.getItem('forge_blocks_'+TODAY)||'[]'));
  renderHome();

  // Register service worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(()=>{});
  }
}

async function boot() {
  // Check if already signed in
  if (HAS_SUPABASE) {
    await initSupabase();
    // If not signed in after a moment, show auth
    setTimeout(()=>{
      if (!currentUser) {
        const authEl = document.getElementById('auth-screen');
        if (authEl) authEl.style.display='flex';
      }
    }, 800);
  } else {
    document.getElementById('auth-screen').classList.add('hidden');
    initApp();
  }
}

// Kick off
boot();
