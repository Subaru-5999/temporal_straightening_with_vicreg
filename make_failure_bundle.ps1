# Builds VICREG_FAILURE_BUNDLE.zip: full code tree + failure log + results
# docs + bundle index. Excludes .git, caches, third-party research_papers,
# CAD meshes, giant old logs, and large binaries.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$staging = Join-Path $root "bundle_failure_staging\VICREG_FAILURE_BUNDLE"
$zip = Join-Path $root "VICREG_FAILURE_BUNDLE.zip"

if (Test-Path (Join-Path $root "bundle_failure_staging")) {
    Remove-Item (Join-Path $root "bundle_failure_staging") -Recurse -Force
}
if (Test-Path $zip) { Remove-Item $zip -Force }

# directories to skip entirely
$excludeDirs = @(
    ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    ".kiro", "research_papers", "checkpoints", "bundle_failure_staging"
)
# env subdirs with large unrelated assets
$excludePaths = @(
    (Join-Path $root "env\deformable_env")
)
# root-level files to skip (noise / huge / third-party)
$excludeFiles = @(
    "make_failure_bundle.ps1", "make_teaching_bundle.ps1",
    "TEACHING_BUNDLE.zip", "VICREG_FAILURE_BUNDLE.zip",
    "temporal_straightening_original.zip", "2603.12231v2.pdf",
    "arXiv-2603.12231v2.tar.gz", "_paper.txt", ".train_pid",
    "train_20260707_090936.log", "train_channel_off_lr1e6.log",
    "train_channel_on.log", "train_pusht_channel_off.log",
    "train_pusht_channel_on.log"
)

function Is-ExcludedDir([string]$name) { $excludeDirs -contains $name }
function Is-ExcludedPath([string]$p) {
    foreach ($x in $excludePaths) { if ($p.StartsWith($x, [System.StringComparison]::OrdinalIgnoreCase)) { return $true } }
    return $false
}

$count = 0
Get-ChildItem $root -Recurse -File | ForEach-Object {
    $f = $_
    $rel = $f.FullName.Substring($root.Length + 1)
    # skip by directory names in path
    foreach ($d in $excludeDirs) {
        if ($rel -match ("^" + [regex]::Escape($d) + "\\")) { return }
        if ($rel -match ("\\" + [regex]::Escape($d) + "\\")) { return }
    }
    if (Is-ExcludedPath $f.FullName) { return }
    if ($f.Directory.FullName -eq $root -and ($excludeFiles -contains $f.Name)) { return }
    $dest = Join-Path $staging $rel
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Copy-Item $f.FullName $dest
    $count++
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory((Resolve-Path (Join-Path $root "bundle_failure_staging")), $zip)
Remove-Item (Join-Path $root "bundle_failure_staging") -Recurse -Force
Write-Output ("created {0} with {1} files, {2:N0} bytes" -f $zip, $count, (Get-Item $zip).Length)
