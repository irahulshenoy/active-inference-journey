// LocalStorage keys
const STORAGE_KEY = 'active-inference-progress';
const STREAK_KEY = 'active-inference-streak';
const LAST_CHECK_KEY = 'active-inference-last-check';

// State management
let state = {
    checked: new Set(),
    currentPhase: 'tonight',
    streak: 0,
    lastCheckDate: null
};

// Initialize the app
function init() {
    loadState();
    renderContent();
    setupEventListeners();
    updateStats();
    calculateExamDays();
    updateStreak();
}

// Load state from localStorage
function loadState() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
        const data = JSON.parse(saved);
        state.checked = new Set(data.checked);
    }
    
    const savedStreak = localStorage.getItem(STREAK_KEY);
    if (savedStreak) {
        state.streak = parseInt(savedStreak);
    }
    
    const savedDate = localStorage.getItem(LAST_CHECK_KEY);
    if (savedDate) {
        state.lastCheckDate = savedDate;
    }
}

// Save state to localStorage
function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
        checked: Array.from(state.checked)
    }));
    localStorage.setItem(STREAK_KEY, state.streak.toString());
    if (state.lastCheckDate) {
        localStorage.setItem(LAST_CHECK_KEY, state.lastCheckDate);
    }
}

// Update streak
function updateStreak() {
    const today = new Date().toDateString();
    
    if (state.lastCheckDate !== today) {
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        
        if (state.lastCheckDate === yesterday.toDateString()) {
            // Continue streak
        } else if (state.lastCheckDate !== null && state.lastCheckDate !== today) {
            // Streak broken
            state.streak = 0;
        }
    }
    
    document.getElementById('current-streak').textContent = state.streak;
}

// Check a task
function checkTask(taskId) {
    if (state.checked.has(taskId)) {
        state.checked.delete(taskId);
    } else {
        state.checked.add(taskId);
        
        // Update streak on first check of the day
        const today = new Date().toDateString();
        if (state.lastCheckDate !== today) {
            state.streak++;
            state.lastCheckDate = today;
        }
    }
    
    saveState();
    renderContent();
    updateStats();
    updateStreak();
}

// Calculate days until exam
function calculateExamDays() {
    const examDate = new Date('2026-09-01');
    const today = new Date();
    const diffTime = examDate - today;
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    document.getElementById('days-until').textContent = diffDays;
}

// Calculate progress
function calculateProgress(tasks) {
    if (tasks.length === 0) return 0;
    const completed = tasks.filter(t => state.checked.has(t.id)).length;
    return Math.round((completed / tasks.length) * 100);
}

// Get all tasks for a phase
function getPhaseTasks(phase) {
    if (phase === 'tonight') {
        return learningPlan.tonight.tasks.map((task, i) => ({
            id: `tonight-${i}`,
            text: task
        }));
    }
    
    const phaseData = learningPlan[phase];
    const tasks = [];
    
    phaseData.weeks.forEach((week, weekIdx) => {
        week.days.forEach((day, dayIdx) => {
            tasks.push({
                id: `${phase}-week${weekIdx}-day${dayIdx}`,
                weekIdx,
                dayIdx,
                day: day.day,
                text: day.task
            });
        });
    });
    
    return tasks;
}

// Update overall stats
function updateStats() {
    const allTasks = [
        ...getPhaseTasks('tonight'),
        ...getPhaseTasks('phase1'),
        ...getPhaseTasks('phase2'),
        ...getPhaseTasks('phase3')
    ];
    
    const progress = calculateProgress(allTasks);
    document.getElementById('total-progress').textContent = `${progress}%`;
}

// Render content based on current phase
function renderContent() {
    const content = document.getElementById('content');
    
    if (state.currentPhase === 'tonight') {
        renderTonight(content);
    } else {
        renderPhase(content, state.currentPhase);
    }
}

// Render tonight's tasks
function renderTonight(container) {
    const tasks = getPhaseTasks('tonight');
    
    container.innerHTML = `
        <div class="tonight-special">
            <h2>${learningPlan.tonight.title}</h2>
            <p style="color: white; margin-bottom: 20px;">${learningPlan.tonight.description}</p>
            ${tasks.map(task => `
                <div class="day-item ${state.checked.has(task.id) ? 'completed' : ''}">
                    <div class="checkbox ${state.checked.has(task.id) ? 'checked' : ''}" 
                         onclick="checkTask('${task.id}')">
                    </div>
                    <div class="day-text">${task.text}</div>
                </div>
            `).join('')}
        </div>
    `;
}

// Render a phase
function renderPhase(container, phase) {
    const phaseData = learningPlan[phase];
    
    let html = `
        <div class="phase-intro">
            <h2>${phaseData.title}</h2>
            <div class="phase-meta">
                <strong>${phaseData.dates}</strong> | ${phaseData.duration}
            </div>
        </div>
    `;
    
    phaseData.weeks.forEach((week, weekIdx) => {
        const weekTasks = [];
        week.days.forEach((day, dayIdx) => {
            weekTasks.push({
                id: `${phase}-week${weekIdx}-day${dayIdx}`,
                day: day.day,
                text: day.task
            });
        });
        
        const weekProgress = calculateProgress(weekTasks);
        
        html += `
            <div class="week-section">
                <div class="week-header" onclick="toggleWeek(this)">
                    <div class="week-title">${week.title}</div>
                    <div class="week-progress">${weekProgress}%</div>
                </div>
                <div class="week-dates" style="color: var(--text-muted); margin-bottom: 10px;">${week.dates}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${weekProgress}%"></div>
                </div>
                <div class="week-content">
                    ${weekTasks.map(task => `
                        <div class="day-item ${state.checked.has(task.id) ? 'completed' : ''}">
                            <div class="checkbox ${state.checked.has(task.id) ? 'checked' : ''}" 
                                 onclick="checkTask('${task.id}')">
                            </div>
                            <div class="day-text">
                                <span class="day-label">Day ${task.day}:</span> ${task.text}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Toggle week expansion
function toggleWeek(header) {
    const content = header.parentElement.querySelector('.week-content');
    content.classList.toggle('expanded');
}

// Setup event listeners
function setupEventListeners() {
    // Phase navigation
    document.querySelectorAll('.phase-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentPhase = btn.dataset.phase;
            renderContent();
        });
    });
    
    // Reset button
    document.getElementById('reset-btn').addEventListener('click', () => {
        if (confirm('Are you sure you want to reset ALL progress? This cannot be undone!')) {
            localStorage.clear();
            state.checked = new Set();
            state.streak = 0;
            state.lastCheckDate = null;
            renderContent();
            updateStats();
            updateStreak();
        }
    });
    
    // Export button
    document.getElementById('export-btn').addEventListener('click', () => {
        const data = {
            progress: Array.from(state.checked),
            streak: state.streak,
            lastCheck: state.lastCheckDate,
            exportDate: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `active-inference-progress-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
    });
}

// Make functions globally available
window.checkTask = checkTask;
window.toggleWeek = toggleWeek;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
