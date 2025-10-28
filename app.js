// LocalStorage keys
const STORAGE_KEY = 'active-inference-progress';
const STREAK_KEY = 'active-inference-streak';
const LAST_CHECK_KEY = 'active-inference-last-check';

// State management
let state = {
    checked: new Set(),
    currentPhase: 'today',  // Default to today's view
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
    updateTodayInfo();
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

// Calculate current day number (Day 1 = Oct 28, 2025)
function getCurrentDayNumber() {
    const startDate = new Date('2025-10-28');
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    startDate.setHours(0, 0, 0, 0);
    
    const diffTime = today - startDate;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    // Day 1 starts on Oct 28, so add 1
    return diffDays + 1;
}

// Find today's task
function getTodaysTask() {
    const dayNum = getCurrentDayNumber();
    
    // Before start date
    if (dayNum < 1) {
        return {
            phase: 'tonight',
            dayNum: dayNum,
            found: true,
            beforeStart: true,
            message: "Get ready! Complete tonight's starter tasks before Day 1 (Oct 28)!"
        };
    }
    
    // Search through all phases
    let currentDay = 1;
    const phases = ['phase1', 'phase2', 'phase3'];
    
    for (const phaseName of phases) {
        const phase = learningPlan[phaseName];
        
        for (let weekIdx = 0; weekIdx < phase.weeks.length; weekIdx++) {
            const week = phase.weeks[weekIdx];
            
            for (let dayIdx = 0; dayIdx < week.days.length; dayIdx++) {
                if (currentDay === dayNum) {
                    return {
                        phase: phaseName,
                        weekIdx: weekIdx,
                        dayIdx: dayIdx,
                        week: week,
                        day: week.days[dayIdx],
                        dayNum: dayNum,
                        found: true,
                        taskId: `${phaseName}-week${weekIdx}-day${dayIdx}`,
                        message: `${week.title}, Day ${dayIdx + 1} of 7`
                    };
                }
                currentDay++;
            }
        }
    }
    
    // After all 266 days
    if (dayNum > 266) {
        return {
            found: false,
            afterEnd: true,
            dayNum: dayNum,
            message: "🎉 Journey Complete! You've finished all 266 days! Time to ace that exam!"
        };
    }
    
    return {
        found: false,
        dayNum: dayNum,
        message: `Day ${dayNum} - Something went wrong`
    };
}

// Update today info in header
function updateTodayInfo() {
    const todayTask = getTodaysTask();
    const todayInfoEl = document.getElementById('today-info');
    
    if (todayInfoEl) {
        if (todayTask.beforeStart) {
            todayInfoEl.innerHTML = `📅 <strong>Before Day 1</strong> - Complete starter tasks!`;
        } else if (todayTask.afterEnd) {
            todayInfoEl.innerHTML = `🎓 <strong>Journey Complete!</strong> - Exam time!`;
        } else if (todayTask.found) {
            todayInfoEl.innerHTML = `📅 <strong>Day ${todayTask.dayNum} of 266</strong> - ${todayTask.message}`;
        } else {
            todayInfoEl.innerHTML = `📅 Day ${todayTask.dayNum}`;
        }
    }
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
    
    if (state.currentPhase === 'today') {
        renderToday(content);
    } else if (state.currentPhase === 'tonight') {
        renderTonight(content);
    } else {
        renderPhase(content, state.currentPhase);
    }
}

// Render today's task view
function renderToday(container) {
    const todayTask = getTodaysTask();
    
    if (todayTask.beforeStart) {
        // Show tonight's tasks
        const tasks = getPhaseTasks('tonight');
        container.innerHTML = `
            <div class="today-view">
                <div class="today-header">
                    <h2>🌟 Get Ready for Day 1!</h2>
                    <p class="today-subtitle">Complete these starter tasks before October 28, 2025</p>
                </div>
                <div class="tonight-special">
                    <h3>${learningPlan.tonight.title}</h3>
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
            </div>
        `;
    } else if (todayTask.afterEnd) {
        container.innerHTML = `
            <div class="today-view">
                <div class="completion-message">
                    <h2>🎉 Congratulations!</h2>
                    <p>You've completed all 266 days of your Active Inference journey!</p>
                    <p>Your qualifying exam is coming up - you're ready! 🎓</p>
                </div>
            </div>
        `;
    } else if (todayTask.found) {
        const isCompleted = state.checked.has(todayTask.taskId);
        const phaseData = learningPlan[todayTask.phase];
        
        container.innerHTML = `
            <div class="today-view">
                <div class="today-header">
                    <h2>📅 Today: Day ${todayTask.dayNum} of 266</h2>
                    <p class="today-subtitle">${todayTask.message}</p>
                    <p class="today-dates">${todayTask.week.dates}</p>
                </div>
                
                <div class="today-task-card">
                    <div class="today-phase-label">${phaseData.title}</div>
                    <div class="day-item today-highlight ${isCompleted ? 'completed' : ''}">
                        <div class="checkbox ${isCompleted ? 'checked' : ''}" 
                             onclick="checkTask('${todayTask.taskId}')">
                        </div>
                        <div class="day-text">
                            <span class="day-label">Day ${todayTask.dayIdx + 1}:</span> ${todayTask.day.task}
                        </div>
                    </div>
                </div>
                
                <div class="today-context">
                    <h3>This Week: ${todayTask.week.title}</h3>
                    <p class="week-dates-small">${todayTask.week.dates}</p>
                    <div class="week-preview">
                        ${todayTask.week.days.map((day, idx) => {
                            const taskId = `${todayTask.phase}-week${todayTask.weekIdx}-day${idx}`;
                            const completed = state.checked.has(taskId);
                            const isToday = idx === todayTask.dayIdx;
                            return `
                                <div class="preview-day ${completed ? 'completed' : ''} ${isToday ? 'current' : ''}">
                                    <div class="preview-num">Day ${idx + 1}</div>
                                    <div class="preview-check">${completed ? '✓' : ''}</div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
                
                <div class="navigation-hint">
                    <p>💡 Use the buttons above to browse all phases and weeks</p>
                </div>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="today-view">
                <div class="today-header">
                    <h2>❓ Day ${todayTask.dayNum}</h2>
                    <p>Unable to find today's task. Check your system date.</p>
                </div>
            </div>
        `;
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
    const todayTask = getTodaysTask();
    
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
                text: day.task,
                isToday: todayTask.found && todayTask.phase === phase && 
                         todayTask.weekIdx === weekIdx && todayTask.dayIdx === dayIdx
            });
        });
        
        const weekProgress = calculateProgress(weekTasks);
        const hasToday = weekTasks.some(t => t.isToday);
        
        html += `
            <div class="week-section ${hasToday ? 'has-today' : ''}">
                <div class="week-header" onclick="toggleWeek(this)">
                    <div class="week-title">
                        ${week.title}
                        ${hasToday ? '<span class="today-badge">📍 TODAY</span>' : ''}
                    </div>
                    <div class="week-progress">${weekProgress}%</div>
                </div>
                <div class="week-dates" style="color: var(--text-muted); margin-bottom: 10px;">${week.dates}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${weekProgress}%"></div>
                </div>
                <div class="week-content ${hasToday ? 'expanded' : ''}">
                    ${weekTasks.map(task => `
                        <div class="day-item ${state.checked.has(task.id) ? 'completed' : ''} ${task.isToday ? 'today-highlight' : ''}">
                            <div class="checkbox ${state.checked.has(task.id) ? 'checked' : ''}" 
                                 onclick="checkTask('${task.id}')">
                            </div>
                            <div class="day-text">
                                <span class="day-label">Day ${task.day}:</span> ${task.text}
                                ${task.isToday ? '<span class="today-marker">← TODAY</span>' : ''}
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
