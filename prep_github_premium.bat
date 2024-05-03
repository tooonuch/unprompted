@ECHO OFF
SETLOCAL
SET "sourcedir=T:\code\python\automatic-stable-diffusion-webui\extensions-builtin\_unprompted"
SET "gitdir=T:\code\github\unprompted_premium"
SET "keepfile=.gitignore"
SET "keepdir=.git"

echo %~dp0

FOR /d %%a IN ("%gitdir%\*") DO IF /i NOT "%%~nxa"=="%keepdir%" RD /S /Q "%%a"
FOR %%a IN ("%gitdir%\*") DO IF /i NOT "%%~nxa"=="%keepfile%" DEL "%%a"

robocopy "%sourcedir%" "%gitdir%" *.* /XD "__pycache__" "%~dp0lib_unprompted\stable_diffusion\clipseg\weights" "%~dp0models" "%~dp0user" "%~dp0lib_unprompted\stable_diffusion\controlnet\models" "%~dp0lib_unprompted\stable_diffusion\controlnet\annotator\ckpts" "%~dp0templates" /XF "prep_github_premium.bat" "config_user.json" "mthesaur.txt" "devcode.py" "dev_preset.txt" /S

robocopy "%sourcedir%\templates\common" "%gitdir%\templates\common" *.* /XF "dev_preset.txt" /S

robocopy "%sourcedir%\templates\pro" "%gitdir%\templates\pro" *.* /XF "dev_preset.txt" /S

pause