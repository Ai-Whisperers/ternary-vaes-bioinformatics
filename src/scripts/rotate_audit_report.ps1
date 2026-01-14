# Audit Report Rotation Script
# Automates the full code health audit pipeline.

$ErrorActionPreference = "Stop"

Write-Host "=================================================="
Write-Host "Code Health Audit Rotation"
Write-Host "=================================================="

# Define script paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
$AnalysisDir = Join-Path $ScriptDir "analysis"

# 1. Audit Repo (Repo Structure, File counts)
Write-Host "`n[1/3] Running Repository Audit..."
$AuditScript = Join-Path $AnalysisDir "audit_repo.py"
python $AuditScript
if ($LASTEXITCODE -ne 0) { throw "Repository Audit Failed" }

# 2. Run Metrics (Complexity, Lines of Code)
Write-Host "`n[2/3] Running Code Metrics Analysis..."
$MetricsScript = Join-Path $AnalysisDir "run_metrics.py"
python $MetricsScript
if ($LASTEXITCODE -ne 0) { throw "Metrics Analysis Failed" }

# 3. Generate Final Report (Markdown Compilation)
Write-Host "`n[3/3] Generating Final Technical Debt Report..."
$ReportScript = Join-Path $AnalysisDir "generate_final_report.py"
python $ReportScript
if ($LASTEXITCODE -ne 0) { throw "Report Generation Failed" }

Write-Host "`n=================================================="
Write-Host "Audit Rotation Complete."
Write-Host "=================================================="
