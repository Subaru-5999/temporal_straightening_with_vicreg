$files = @(
 "BUNDLE_README.md",
 "ICLR_TEACHING_GUIDE.md",
 "models\visual_world_model.py",
 "models\sigreg.py",
 "models\diagnostics.py",
 "models\dino.py",
 "models\vit.py",
 "models\proprio.py",
 "planning\gd.py",
 "planning\cem.py",
 "planning\mpc.py",
 "planning\objectives.py",
 "planning\evaluator.py",
 "planning\base_planner.py",
 "conf\train.yaml",
 "env\pusht\pusht_wrapper.py",
 "experiments\verify_stop_grad.py",
 "experiments\verify_encoder_trains.py",
 "experiments\metric_alignment.py",
 "experiments\curvature_incentive.py",
 "experiments\rollout_drift.py",
 "experiments\planning_landscape.py",
 "experiments\planning_jacobian.py",
 "experiments\violation_of_expectation.py",
 "train.py",
 "reproduce_table1.py",
 "PROGRESS_SIGREG_E2E.md",
 "_paper.txt"
)
$staging = "bundle_staging"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
foreach ($f in $files) {
    $dest = Join-Path $staging $f
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Copy-Item $f $dest
}
$zip = "TEACHING_BUNDLE.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory((Resolve-Path $staging), $zip)
Remove-Item $staging -Recurse -Force
Write-Output ("created {0} with {1} files, {2:N0} bytes" -f $zip, $files.Count, (Get-Item $zip).Length)
