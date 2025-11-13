@echo off
chcp 65001 >nul
color 0A
title GitHub Repository Publisher

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║        🚀 GitHub Repository Publisher for ML Project        ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/4] Checking Git status...
git status >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Git repository not found!
    pause
    exit /b 1
)

echo ✅ Git repository found
echo.

echo [2/4] Opening GitHub to create repository...
echo.
echo 👉 In your browser:
echo    1. Log in to GitHub
echo    2. Click "New repository" or use the opened link
echo    3. Repository name: cat-dog-detection-ml
echo    4. Description: AI-powered Cat vs Dog detection using YOLO with GUI
echo    5. Choose Public or Private
echo    6. ⚠️  DO NOT add README, .gitignore, or license
echo    7. Click "Create repository"
echo.

start https://github.com/new

echo Press any key AFTER you created the repository on GitHub...
pause >nul

echo.
echo [3/4] Please enter your GitHub information:
echo.

set /p GITHUB_USERNAME="Enter your GitHub username: "
set /p REPO_NAME="Enter repository name [cat-dog-detection-ml]: "

if "%REPO_NAME%"=="" set REPO_NAME=cat-dog-detection-ml

echo.
echo [4/4] Pushing to GitHub...
echo.

echo Adding remote...
git remote remove origin 2>nul
git remote add origin https://github.com/%GITHUB_USERNAME%/%REPO_NAME%.git

echo Renaming branch to main...
git branch -M main

echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ⚠️  Push failed! This is usually due to authentication.
    echo.
    echo 💡 Solution:
    echo    1. GitHub will prompt for login
    echo    2. Use your username: %GITHUB_USERNAME%
    echo    3. For password, use a Personal Access Token:
    echo       - Go to: https://github.com/settings/tokens
    echo       - Generate new token (classic)
    echo       - Select 'repo' permissions
    echo       - Copy the token and paste as password
    echo.
    echo Opening token creation page...
    start https://github.com/settings/tokens/new
    echo.
    echo After creating token, let's try again...
    pause
    echo.
    git push -u origin main
)

echo.
if errorlevel 0 (
    echo ╔═══════════════════════════════════════════════════════════════╗
    echo ║                    ✅ SUCCESS! 🎉                            ║
    echo ╚═══════════════════════════════════════════════════════════════╝
    echo.
    echo Your repository is now live at:
    echo 🌐 https://github.com/%GITHUB_USERNAME%/%REPO_NAME%
    echo.
    echo Opening your repository...
    start https://github.com/%GITHUB_USERNAME%/%REPO_NAME%
    echo.
    echo 🎯 Next steps:
    echo    - Add topics: machine-learning, yolo, pytorch, object-detection
    echo    - Upload a screenshot as social preview image
    echo    - Share your project with the community!
    echo.
) else (
    echo ❌ Failed to push to GitHub
    echo.
    echo Please check:
    echo 1. Repository exists: https://github.com/%GITHUB_USERNAME%/%REPO_NAME%
    echo 2. You have correct permissions
    echo 3. Your credentials are correct
    echo.
)

pause

