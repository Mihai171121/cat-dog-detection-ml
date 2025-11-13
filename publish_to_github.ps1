# GitHub Repository Publisher
# PowerShell script to publish your ML project to GitHub

$Host.UI.RawUI.WindowTitle = "GitHub Repository Publisher"
Write-Host "`n"
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║        🚀 GitHub Repository Publisher for ML Project        ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "`n"

# Change to script directory
Set-Location -Path $PSScriptRoot

# Check Git status
Write-Host "[1/4] Checking Git status..." -ForegroundColor Cyan
try {
    git status *>$null
    Write-Host "✅ Git repository found`n" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Git repository not found!" -ForegroundColor Red
    pause
    exit 1
}

# Open GitHub
Write-Host "[2/4] Opening GitHub to create repository..." -ForegroundColor Cyan
Write-Host "`n👉 In your browser:" -ForegroundColor Yellow
Write-Host "   1. Log in to GitHub" -ForegroundColor White
Write-Host "   2. Click 'New repository' or use the opened link" -ForegroundColor White
Write-Host "   3. Repository name: " -NoNewline -ForegroundColor White
Write-Host "cat-dog-detection-ml" -ForegroundColor Cyan
Write-Host "   4. Description: " -NoNewline -ForegroundColor White
Write-Host "AI-powered Cat vs Dog detection using YOLO with GUI" -ForegroundColor Cyan
Write-Host "   5. Choose Public or Private" -ForegroundColor White
Write-Host "   6. ⚠️  DO NOT add README, .gitignore, or license" -ForegroundColor Yellow
Write-Host "   7. Click 'Create repository'" -ForegroundColor White
Write-Host "`n"

Start-Process "https://github.com/new"

Write-Host "Press any key AFTER you created the repository on GitHub..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Get user input
Write-Host "`n[3/4] GitHub information:" -ForegroundColor Cyan
Write-Host "`n"

$githubUsername = Read-Host "Enter your GitHub username"
$repoName = Read-Host "Enter repository name [cat-dog-detection-ml]"

if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "cat-dog-detection-ml"
}

# Push to GitHub
Write-Host "`n[4/4] Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "`n"

Write-Host "Adding remote..." -ForegroundColor Gray
git remote remove origin 2>$null
git remote add origin "https://github.com/$githubUsername/$repoName.git"

Write-Host "Renaming branch to main..." -ForegroundColor Gray
git branch -M main

Write-Host "Pushing to GitHub..." -ForegroundColor Gray
Write-Host "`n⚠️  GitHub will ask for authentication:" -ForegroundColor Yellow
Write-Host "   Username: $githubUsername" -ForegroundColor Cyan
Write-Host "   Password: Use Personal Access Token (NOT your password!)" -ForegroundColor Cyan
Write-Host "`n"

$pushResult = git push -u origin main 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  Push failed! This is usually due to authentication.`n" -ForegroundColor Yellow
    Write-Host "💡 Solution:" -ForegroundColor Cyan
    Write-Host "   1. Create a Personal Access Token:" -ForegroundColor White
    Write-Host "      - Go to: https://github.com/settings/tokens" -ForegroundColor Gray
    Write-Host "      - Click 'Generate new token (classic)'" -ForegroundColor Gray
    Write-Host "      - Select 'repo' permissions" -ForegroundColor Gray
    Write-Host "      - Copy the token" -ForegroundColor Gray
    Write-Host "   2. When prompted for password, paste the token" -ForegroundColor White
    Write-Host "`n"

    Write-Host "Opening token creation page..." -ForegroundColor Cyan
    Start-Process "https://github.com/settings/tokens/new?scopes=repo`&description=ML-Project-Upload"

    Write-Host "`nPress any key after creating the token to try again..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

    Write-Host "`nTrying again...`n" -ForegroundColor Cyan
    git push -u origin main
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    ✅ SUCCESS! 🎉                            ║" -ForegroundColor Green
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host "`n"

    $repoUrl = "https://github.com/$githubUsername/$repoName"
    Write-Host "Your repository is now live at:" -ForegroundColor Cyan
    Write-Host "🌐 $repoUrl`n" -ForegroundColor Yellow

    Write-Host "Opening your repository..." -ForegroundColor Cyan
    Start-Process $repoUrl

    Write-Host "`n🎯 Next steps:" -ForegroundColor Cyan
    Write-Host "   - Add topics: machine-learning, yolo, pytorch, object-detection" -ForegroundColor White
    Write-Host "   - Upload a screenshot as social preview image" -ForegroundColor White
    Write-Host "   - Share your project with the community!" -ForegroundColor White
    Write-Host "`n"
} else {
    Write-Host "`n❌ Failed to push to GitHub`n" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "1. Repository exists: https://github.com/$githubUsername/$repoName" -ForegroundColor White
    Write-Host "2. You have correct permissions" -ForegroundColor White
    Write-Host "3. Your credentials are correct" -ForegroundColor White
    Write-Host "`n"
}

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

