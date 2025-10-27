# Active Inference Learning Journey Tracker 🧠

An interactive web-based tracker for your 9-month journey to mastering Active Inference and acing your qualifying exam!

## Features

✅ **Interactive Checkboxes** - Click to mark tasks complete, progress saves automatically  
📊 **Progress Tracking** - See your progress for each week and overall  
🔥 **Streak Counter** - Track your daily learning streak  
📱 **Mobile Friendly** - Works great on phone, tablet, or desktop  
🎯 **Phase Navigation** - Easy navigation between learning phases  
💾 **Auto-Save** - Your progress is saved in your browser automatically  
📥 **Export Progress** - Download your progress as JSON

## How to Deploy to GitHub Pages

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
git commit -m "Initial commit"
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

## Usage Tips

### Daily Workflow
1. Open your tracker every morning
2. Check off tasks as you complete them
3. Watch your streak grow! 🔥

### Week-by-Week
- Click on each week section to expand/collapse
- Progress bars show your completion for each week
- Focus on one week at a time to avoid overwhelm

### Tonight's Tasks
- Start with the "Tonight's Task ⭐" button
- Complete these 4 simple tasks to get started RIGHT NOW

### Progress Tracking
- **Overall Progress**: Shows your total completion across all phases
- **Streak**: Tracks consecutive days you've checked off tasks
- **Days Until Exam**: Countdown to September 1, 2026

### Backup Your Progress
- Click **Export Progress** to download your data
- Save this file somewhere safe (Google Drive, Dropbox)
- Your progress is also saved in your browser automatically

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

### Update Tasks
Edit `data.js` to modify the learning plan structure.

### Adjust Exam Date
In `app.js`, line 62, change the exam date:
```javascript
const examDate = new Date('2026-09-01');
```

## Mobile Usage

### iOS (iPhone/iPad)
1. Open your tracker in Safari
2. Tap the Share button
3. Scroll down and tap **Add to Home Screen**
4. Name it and tap **Add**
5. Now you have an app icon!

### Android
1. Open your tracker in Chrome
2. Tap the three dots menu
3. Tap **Add to Home screen**
4. Confirm and you're done!

## Troubleshooting

### My progress disappeared!
- Progress is saved per-browser. If you switch browsers or clear browser data, progress resets
- Use **Export Progress** regularly to backup
- Consider using the same browser consistently

### Can I use this on multiple devices?
- Progress is stored locally in each browser
- You can manually sync by exporting from one device and keeping a copy

### GitHub Pages not working?
- Make sure your repository is **Public**
- Check that you selected the **main** branch in Settings → Pages
- Wait 2-5 minutes after enabling Pages
- Try a hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

## Tech Stack

- **Pure JavaScript** - No frameworks needed
- **LocalStorage** - Saves progress in your browser
- **GitHub Pages** - Free hosting
- **Responsive CSS** - Works on all devices

## License

Free to use, modify, and share! Good luck on your journey! 🚀

---

## Quick Start Checklist

- [ ] Create GitHub repository
- [ ] Upload all files
- [ ] Enable GitHub Pages
- [ ] Visit your live site
- [ ] Bookmark it
- [ ] Add to phone home screen
- [ ] Complete Tonight's Tasks! ⭐
- [ ] Start Week 1 tomorrow!

**Remember:** Consistency beats intensity. 2 hours a day, every day, for 9 months = Qualifying Exam SUCCESS! 🎉
