# 01-enable-wsl.ps1 —— 必须【以管理员身份】运行
# 作用：启用 WSL2 相关可选功能 + 安装 Ubuntu-24.04（不自动启动，避免交互卡住）
# 日志：setup\01-enable-wsl.log

$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot '01-enable-wsl.log'
Start-Transcript -Path $log -Force | Out-Null

function Section($t) { Write-Host "`n===== $t =====" -ForegroundColor Cyan }

Section "0) 当前状态"
Write-Host "管理员: $(if (([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {'是'} else {'否'})"
wsl.exe --version 2>&1 | Select-Object -First 5

Section "1) 启用 虚拟机平台 与 WSL 可选功能"
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

Section "2) 设置 WSL 默认版本为 2"
wsl.exe --set-default-version 2

Section "3) 安装 Ubuntu-24.04（--no-launch：先不进入首次交互）"
wsl.exe --install -d Ubuntu-24.04 --no-launch
if ($LASTEXITCODE -ne 0) {
    Write-Host "(--no-launch 不可用，改用普通安装)" -ForegroundColor Yellow
    wsl.exe --install -d Ubuntu-24.04
}

Section "4) 最终状态"
wsl.exe --status
wsl.exe -l -v

Section "完成"
Write-Host "若上面提示需要重启，请重启 Windows，然后继续 02 步。" -ForegroundColor Green
Write-Host "如需把 WSL 发行版迁到 D 盘（省 C 盘空间），可执行 setup\01b-move-wsl-to-d.ps1" -ForegroundColor Green

Stop-Transcript | Out-Null
