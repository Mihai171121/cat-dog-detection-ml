# 🚀 GitHub Repository Setup Guide

Your local Git repository is ready! Follow these steps to publish it on GitHub:

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `cat-dog-detection-ml` (or your preferred name)
   - **Description**: "AI-powered Cat vs Dog detection using YOLO with graphical interface"
   - **Visibility**: Choose Public or Private
   - ⚠️ **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Connect and Push to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Navigate to your project
cd "D:\Curs Python\ML Cats vs Dogs"

# Add GitHub as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (GitHub's default)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

### Example (replace with your info):
```bash
git remote add origin https://github.com/yourusername/cat-dog-detection-ml.git
git branch -M main
git push -u origin main
```

## Step 3: Configure Your GitHub Repository

After pushing, configure these settings on GitHub:

### Add Topics (for discoverability)
Go to repository → Settings → About section → Add topics:
- `machine-learning`
- `deep-learning`
- `yolo`
- `object-detection`
- `pytorch`
- `computer-vision`
- `cat-dog-classification`
- `gui`
- `python`

### Set Repository Image
- Upload a screenshot of your GUI as the social preview image
- Go to Settings → General → Social preview → Upload image

### Enable GitHub Pages (optional)
If you want to showcase your project documentation:
- Settings → Pages → Source → Deploy from main branch

## Step 4: Create Sample Images Folder (Optional)

If you want to include sample images in the repository:

```bash
# Create a samples folder with small demo images
mkdir samples
# Copy 2-3 small demo images there (< 1MB each)
# Then:
git add samples/
git commit -m "Add sample detection images"
git push
```

## Step 5: Add Badges and Links

Your README.md already includes badges. After pushing, you can add more:

- **GitHub Stars**: `![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/REPO_NAME)`
- **GitHub Issues**: `![GitHub issues](https://img.shields.io/github/issues/YOUR_USERNAME/REPO_NAME)`
- **Last Commit**: `![GitHub last commit](https://img.shields.io/github/last-commit/YOUR_USERNAME/REPO_NAME)`

## 📝 Important Notes

### What's NOT Pushed (via .gitignore):
- ✅ Large dataset files (Data_set_Cat_vs_Dog/)
- ✅ Trained models (*.pt files - too large)
- ✅ Training outputs (runs/)
- ✅ Images and videos
- ✅ Virtual environment files

### What IS Pushed:
- ✅ Source code
- ✅ Configuration files
- ✅ Documentation (README, PowerPoint presentation)
- ✅ Requirements file
- ✅ Training scripts

## 🔄 Daily Workflow

After initial setup, use these commands for updates:

```bash
# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "Your descriptive message"

# Push to GitHub
git push
```

## 🛠️ Troubleshooting

### Authentication Issues
If GitHub asks for authentication:

**Option 1: Personal Access Token**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` permissions
3. Use token as password when pushing

**Option 2: SSH Keys**
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
3. Change remote to SSH: `git remote set-url origin git@github.com:USERNAME/REPO.git`

### Large Files Warning
If git warns about large files:
```bash
# Remove from staging
git rm --cached path/to/large/file

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Commit
git add .gitignore
git commit -m "Fix: Remove large files"
```

## 🎉 You're All Set!

Your project is now ready for GitHub! The repository includes:
- ✅ Complete source code
- ✅ Professional README with badges
- ✅ MIT License
- ✅ Contributing guidelines
- ✅ Proper .gitignore
- ✅ PowerPoint presentation

Share your repository link with the community! 🚀

---

**Quick Reference:**
```bash
git add .                          # Stage changes
git commit -m "message"            # Commit changes
git push                           # Push to GitHub
git pull                           # Pull updates
git status                         # Check status
git log --oneline                  # View commit history
```

