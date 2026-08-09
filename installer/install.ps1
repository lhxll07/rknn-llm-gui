$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RequestedDistro = $env:RKLLM_WORKBENCH_WSL_DISTRO

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "未检测到 WSL2。请先安装 WSL2，再重新运行此安装器。"
}

$distros = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($RequestedDistro) {
    $Distro = $RequestedDistro
} elseif ($distros -contains "Ubuntu") {
    $Distro = "Ubuntu"
} elseif ($distros.Count -eq 1) {
    $Distro = $distros[0]
} else {
    throw "请设置 RKLLM_WORKBENCH_WSL_DISTRO 指定要使用的 WSL 发行版，例如 Arch。"
}

if ($distros -notcontains $Distro) {
    throw "未找到 WSL 发行版 '$Distro'。请先安装它，再重新运行此安装器。"
}

$WslRoot = (& wsl.exe -d $Distro -- wslpath -a ($RootDir -replace '\\', '/')).Trim()
if (-not $WslRoot) {
    throw "无法将项目路径转换为 WSL 路径。"
}

Write-Host "正在 WSL 发行版 '$Distro' 中安装 RKLLM 工作台……"
& wsl.exe -d $Distro -- bash -lc "cd '$WslRoot' && bash installer/install.sh"
if ($LASTEXITCODE -ne 0) {
    throw "WSL 安装器失败，退出码：$LASTEXITCODE。"
}
