# Active Inference Learning Journey Tracker 🧠

An interactive web-based tracker for your 10-month journey to mastering Active Inference and acing your qualifying exam on **September 1, 2026**!

## ✨ NEW: Today's Tasks Feature!

**Your tracker now automatically knows what day you're on!** Every time you open it, you'll see:
- 📅 **Current day number** (Day X of 266)
- 🎯 **Today's specific task** highlighted and ready to go
- 📊 **Week progress** showing which days you've completed
- 🔥 **No more hunting** - just click "TODAY" and start working!

## Features

✅ **Auto-Daily Updates** - Tracker automatically shows today's task based on Oct 28 start date  
✅ **Interactive Checkboxes** - Click to mark tasks complete, progress saves automatically  
📊 **Progress Tracking** - See your progress for each week and overall  
🔥 **Streak Counter** - Track your daily learning streak  
📱 **Mobile Friendly** - Works great on phone, tablet, or desktop  
🎯 **Smart Navigation** - Jump to today, or browse all phases  
💾 **Auto-Save** - Your progress is saved in your browser automatically  
📥 **Export Progress** - Download your progress as JSON  
🌟 **Smart Highlighting** - Today's task is highlighted, week auto-expands

## Quick Start

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **+** icon in the top right, select **New repository**
3. Name it something like `active-inference-journey` or `learning-tracker`
4. Make it **Public** (required for free GitHub Pages)
5. Click **Create repository**

### Step 2: Upload Your Files

**Option A: Upload via GitHub Website (Easiest)**

1. On your new repository page, click **uploading an existing file**
2. Drag and drop all these files:
   - `index.html`
   - `styles.css`
   - `app.js`
   - `data.js`
   - `README.md`
3. Click **Commit changes**

**Option B: Use Git Command Line**

```bash
cd learning-tracker
git init
git add .
git commit -m "Initial commit with Today feature"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

### Step 3: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (gear icon)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select **main** branch
5. Click **Save**
6. Wait 1-2 minutes, then refresh the page
7. You'll see: "Your site is live at `https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/`"

### Step 4: Access Your Tracker!

Visit: `https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/`

Bookmark it! Add it to your home screen on mobile!

## How to Use

### Daily Workflow
1. Open your tracker every morning
2. Click **"📍 TODAY"** button (it's the first one!)
3. See exactly what you need to do today
4. Check off tasks as you complete them
5. Watch your streak grow! 🔥

### Navigation
- **📍 TODAY** - Automatically shows today's task (updates daily!)
- **Phase 1-3** - Browse all weeks and tasks
- **Tonight's Tasks ⭐** - The 4 starter tasks to complete before Day 1

### How Today's Feature Works
- **Oct 27, 2025 (before start)**: Shows starter tasks
- **Oct 28, 2025 (Day 1)**: Shows "Day 1 - Python Basics"
- **Each day after**: Automatically updates to show current day's task
- **After Sept 1, 2026**: Shows completion message 🎉

The tracker calculates your current day based on the start date (Oct 28, 2025) and automatically displays the right task. No manual input needed!

### Understanding the Display

When you click "TODAY", you'll see:
- **Day counter**: "Day 15 of 266"
- **Week info**: Which week and day within that week
- **Today's task**: Highlighted in blue with specific instructions
- **Week preview**: All 7 days of current week with completion status
- **Navigation hint**: Reminder you can browse other phases

### Progress Tracking
- **Overall Progress**: Shows your total completion across all 266 days
- **Streak**: Tracks consecutive days you've checked off tasks
- **Days Until Exam**: Countdown to September 1, 2026
- **Week Progress**: Each week shows its own completion percentage

### Backup Your Progress
- Click **Export Progress** to download your data
- Save this file somewhere safe (Google Drive, Dropbox)
- Your progress is also saved in your browser automatically
- Export regularly as a backup!

## Timeline Overview

- **Tonight (Oct 27)**: Complete 4 starter tasks ✅
- **Phase 1** (12 weeks): Oct 28 - Dec 22, 2025
  - Python, NumPy, Math, PyTorch, Neural Networks, RL
- **Phase 2** (16 weeks): Dec 23, 2025 - May 11, 2026
  - Active Inference theory, Driver behavior, Lab integration
- **Phase 3** (16 weeks): May 12 - Aug 31, 2026
  - Literature review, Implementation, Presentation prep
- **Exam Week**: Aug 25 - Sept 1, 2026 🎓

**Total: 266 days of focused learning**

## Customization

### Change Colors
Edit `styles.css` and modify the `:root` variables:
```css
:root {
    --primary: #6366f1;  /* Main accent color */
    --success: #10b981;  /* Completion color */
    --bg: #0f172a;       /* Background */
}
```

### Update Start Date
If you need to change the start date, edit `app.js`, around line 103:
```javascript
function getCurrentDayNumber() {
    const startDate = new Date('2025-10-28');  // Change this date
    // ...
}
```

### Adjust Exam Date
In `app.js`, around line 95:
```javascript
function calculateExamDays() {
    const examDate = new Date('2026-09-01');  // Change exam date
    // ...
}
```

### Update Tasks
Edit `data.js` to modify the learning plan structure. Each day has:
- `day`: Day number within the week (1-7)
- `task`: The specific task with resources and links

## Mobile Usage

### iOS (iPhone/iPad)
1. Open your tracker in Safari
2. Tap the Share button (square with arrow)
3. Scroll down and tap **Add to Home Screen**
4. Name it and tap **Add**
5. Now you have an app icon! 📱

### Android
1. Open your tracker in Chrome
2. Tap the three dots menu (⋮)
3. Tap **Add to Home screen**
4. Confirm and you're done! 📱

## Troubleshooting

### My progress disappeared!
- Progress is saved per-browser. If you switch browsers or clear browser data, progress resets
- Use **Export Progress** regularly to backup
- Consider using the same browser consistently

### Today's task isn't showing correctly
- Check your computer's date and time are set correctly
- The tracker calculates based on system date
- If Oct 28, 2025 hasn't arrived yet, it will show starter tasks

### Can I use this on multiple devices?
- Progress is stored locally in each browser
- You can manually sync by exporting from one device and keeping backup
- For true multi-device sync, you'd need to add a backend (advanced)

### GitHub Pages not working?
- Make sure your repository is **Public**
- Check that you selected the **main** branch in Settings → Pages
- Wait 2-5 minutes after enabling Pages
- Try a hard refresh (Ctrl+Shift+R or Cmd+Shift+R)
- Check for errors in browser console (F12)

### The dates seem wrong in the tracker
- The tracker starts on **October 28, 2025** (Day 1)
- Make sure you're viewing it on or after that date
- If you want to test it early, you can temporarily change the start date in `app.js`

## Technical Details

### Tech Stack
- **Pure JavaScript** - No frameworks needed
- **LocalStorage** - Saves progress in your browser
- **GitHub Pages** - Free hosting
- **Responsive CSS** - Works on all devices
- **Date Calculations** - Automatic daily task detection

### How Auto-Detection Works
1. Gets current date from your system
2. Calculates days since Oct 28, 2025
3. Maps day number to specific task in data structure
4. Highlights and displays that task
5. Shows week context and progress

### Files Structure
- `index.html` - Main page structure with navigation
- `styles.css` - All styling including today's view
- `app.js` - Logic for day calculation and rendering
- `data.js` - Complete 266-day curriculum with resources
- `README.md` - This file!

## Privacy & Data

- **All data stays on your device** - Nothing is sent to any server
- Your progress is stored in browser's LocalStorage
- No tracking, no analytics, no cookies
- Export feature creates local JSON file only
- Completely private and offline-capable (after first load)

## Tips for Success

### Daily Habits
✅ Open tracker first thing in morning  
✅ Complete today's task before moving on  
✅ Check off tasks immediately when done  
✅ Review week progress on Sundays  
✅ Export progress monthly as backup  

### Staying Motivated
- Celebrate your streak! 🔥 Try to never break it
- Share progress with lab mates
- Use the week preview to see your progress
- Remember: Consistency beats intensity
- 2 hours a day × 266 days = Qualifying exam SUCCESS! 🎉

### If You Fall Behind
- Don't panic! Life happens
- Use phase navigation to catch up
- Focus on understanding, not just checking boxes
- Adjust pace if needed (talk to McDonald)
- The tracker is a guide, not a strict schedule

## Support

### Questions or Issues?
- Check this README first
- Review Troubleshooting section above
- Check browser console for errors (F12)
- Verify all files are uploaded correctly
- Make sure GitHub Pages is enabled

### Want to Contribute?
This is your personal tracker, but if you improve it and want to share:
- Fork the repo
- Make your changes
- Document what you added
- Share with lab mates!

## License

Free to use, modify, and share! Good luck on your journey! 🚀

---

## Quick Start Checklist

- [ ] Create GitHub repository
- [ ] Upload all 5 files (index.html, styles.css, app.js, data.js, README.md)
- [ ] Enable GitHub Pages in Settings
- [ ] Visit your live site
- [ ] Bookmark it in browser
- [ ] Add to phone home screen 📱
- [ ] Click "📍 TODAY" to see what to do!
- [ ] Complete today's task! ✅
- [ ] Check it off and watch your streak begin! 🔥

---

**Start Date**: October 28, 2025 (Tomorrow!)  
**Exam Date**: September 1, 2026  
**Total Days**: 266 days of focused learning  
**Your Goal**: Pass qualifying exam with flying colors! 🎓

**Remember**: The journey of a thousand miles begins with a single step. Or in this case, 266 days begins with Day 1. You've got this! 💪

*Last Updated: October 27, 2025 - Added automatic daily task detection!*
