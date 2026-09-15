# 01b-move-wsl-to-d.ps1 —— 可选，【以管理员身份】运行
# 作用：把 Ubuntu 发行版从 C 盘迁到 D 盘（当前 C 盘有 241GB 空闲，非必需）
# 注意：迁移后默认用户名需要重新指定（见脚本末尾提示）

$ErrorActionPreference = 'Stop'
$Distro = 'Ubuntu-24.04'
$Target = 'D:\wsl\Ubuntu-24.04'
$Tar    = 'D:\wsl\Ubuntu-24.04.tar'

New-Item -ItemType Directory -Force -Path 'D:\wsl' | Out-Null

Write-Host "== 1) 关闭 WSL ==" -ForegroundColor Cyan
wsl.exe --shutdown

Write-Host "== 2) 导出到 $Tar（可能几分钟，体积数 GB）==" -ForegroundColor Cyan
wsl.exe --export $Distro $Tar
if ($LASTEXITCODE -ne 0) { throw "导出失败：请确认发行版名称为 $Distro（用 wsl -l -v 查看）" }

Write-Host "== 3) 注销原发行版 ==" -ForegroundColor Cyan
wsl.exe --unregister $Distro

Write-Host "== 4) 从 D 盘导入 ==" -ForegroundColor Cyan
wsl.exe --import $Distro $Target $Tar --version 2

Write-Host "== 5) 清理 tar 包（确认导入成功后手动删除更稳妥）==" -ForegroundColor Yellow
Write-Host "   确认无误后可执行: Remove-Item '$Tar'"

Write-Host "== 6) 状态 ==" -ForegroundColor Cyan
wsl.exe -l -v

Write-Host @"

导入后默认用户会变成 root，恢复普通用户：
  1) wsl -d $Distro           # 以 root 进入
  2) 编辑 /etc/wsl.conf 追加：
       [user]
       default=<你的Linux用户名>
  3) Windows 执行 wsl --shutdown 后重新进入
"@ -ForegroundColor Green
