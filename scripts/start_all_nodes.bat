@echo off
REM Lance les 3 noeuds Fed-BioMed dans des fenetres separees.
REM A executer APRES avoir cree le researcher (cellule 1 du notebook 03).

set VENV=%~dp0..\.venv\Scripts\python.exe
set SCRIPT=%~dp0start_node.py

echo Demarrage des 3 noeuds Fed-BioMed...
start "FBM Node 1 - Hopital 1" cmd /k "%VENV%" "%SCRIPT%" --node-id 1
timeout /t 2 /nobreak >nul
start "FBM Node 2 - Hopital 2" cmd /k "%VENV%" "%SCRIPT%" --node-id 2
timeout /t 2 /nobreak >nul
start "FBM Node 3 - Hopital 3" cmd /k "%VENV%" "%SCRIPT%" --node-id 3

echo Noeuds en cours de demarrage dans 3 fenetres separees.
echo Attendez que les 3 noeuds soient connectes avant de continuer dans le notebook.
