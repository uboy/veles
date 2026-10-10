@echo off
rem Установка veles в один клик: двойной клик по этому файлу.
rem cmd не подчиняется политике выполнения PowerShell, поэтому запускает
rem install.ps1 с явным разрешением; прав администратора не требуется.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
