$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RequestedDistro = $env:RKLLM_WORKBENCH_WSL_DISTRO

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "未检测到 WSL2。"
}

$distros = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($RequestedDistro) {
    $Distro = $RequestedDistro
} elseif ($distros -contains "Ubuntu") {
    $Distro = "Ubuntu"
} elseif ($distros.Count -eq 1) {
    $Distro = $distros[0]
} else {
    throw "请设置 RKLLM_WORKBENCH_WSL_DISTRO 指定要使用的 WSL 发行版。"
}

if ($distros -notcontains $Distro) {
    throw "未找到 WSL 发行版 '$Distro'。"
}

$WslRoot = (& wsl.exe -d $Distro -- wslpath -a ($RootDir -replace '\\', '/')).Trim()
if (-not $WslRoot) {
    throw "无法将项目路径转换为 WSL 路径。"
}

& wsl.exe -d $Distro -- bash -lc "cd '$WslRoot' && bash installer/uninstall.sh"
if ($LASTEXITCODE -ne 0) {
    throw "卸载失败，退出码：$LASTEXITCODE。"
}
