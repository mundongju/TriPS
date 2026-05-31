# Render every figure PDF in static\pdfs\ to a crisp PNG in static\images\.
# The page works without this (it falls back to embedding the PDFs); this just
# produces sharp, mobile-friendly rasters.
#   powershell -ExecutionPolicy Bypass -File convert_figures.ps1 [-Dpi 200]
param([int]$Dpi = 200)

Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "static\images" | Out-Null

function Find-Tool($names) { foreach ($n in $names) { $c = Get-Command $n -ErrorAction SilentlyContinue; if ($c) { return $c.Source } }; return $null }
$pdftocairo = Find-Tool @('pdftocairo')
$pdftoppm   = Find-Tool @('pdftoppm')
$magick     = Find-Tool @('magick')
$gs         = Find-Tool @('gswin64c','gswin32c','gs')

if (-not ($pdftocairo -or $pdftoppm -or $magick -or $gs)) {
  Write-Error "Install poppler (pdftocairo/pdftoppm), ImageMagick (magick), or Ghostscript (gswin64c)."; exit 1
}

Get-ChildItem "static\pdfs\*.pdf" | ForEach-Object {
  $name = $_.BaseName
  if ($name -eq "TriPS_arxiv") { return }
  $pdf = $_.FullName
  $pre = "static\images\$name"
  $out = "$pre.png"
  Write-Output "[render] $name"
  if     ($pdftocairo) { & $pdftocairo -png -r $Dpi -singlefile $pdf $pre }
  elseif ($pdftoppm)   { & $pdftoppm  -png -r $Dpi -singlefile $pdf $pre }
  elseif ($magick)     { & $magick -density $Dpi "$pdf[0]" -background white -flatten $out }
  elseif ($gs)         { & $gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r$Dpi -dFirstPage=1 -dLastPage=1 -o $out $pdf }
}
Write-Output "Done -> static\images\*.png"
