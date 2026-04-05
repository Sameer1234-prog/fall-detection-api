@echo off
echo Replace YOUR_USERNAME with your actual GitHub username
echo Then run this file

set GITHUB_URL=https://github.com/YOUR_USERNAME/fall-detection-api.git

git remote add origin %GITHUB_URL%
git branch -M main
git push -u origin main

echo Done! Now go to railway.app to deploy
pause
