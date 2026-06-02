# Render Table 1 (paper quantitative comparison) to static/images/Table1.png
#   powershell -ExecutionPolicy Bypass -File make_table1.ps1
# Markers in the data:  '#'=best (bold),  '~'=second best (underline).
Set-Location -Path $PSScriptRoot
Add-Type -AssemblyName System.Drawing
$out = "static\images\Table1.png"
$tasks=@('Super-Resolution x8','Super-Resolution x12','Motion Deblurring','Gaussian Deblurring')
$metrics=@('PSNR','SSIM','FID','LPIPS','MUSIQ'); $arrows=@([char]0x2191,[char]0x2191,[char]0x2193,[char]0x2193,[char]0x2191)
$ffhq=@(
 @('ReSample','24.65','0.708','98.07','0.196','32.06','22.96','0.632','172.19','0.357','21.09','25.54','0.747','96.62','0.216','31.65','26.37','0.746','97.47','0.160','31.50'),
 @('FlowChef','27.53','0.759','57.24','0.147','49.05','26.38','0.731','110.15','0.209','38.70','24.88','0.716','63.48','0.237','39.22','27.30','0.754','46.40','0.152','44.26'),
 @('FlowDPS','27.92','~0.772','~23.80','0.120','~54.36','26.84','0.745','30.54','~0.156','48.33','25.15','0.721','43.18','0.222','48.51','26.02','0.731','45.00','0.204','47.11'),
 @('FLAIR','~28.88','0.768','55.51','0.123','52.26','27.25','0.752','~29.31','0.158','46.63','28.80','0.695','21.57','0.095','52.48','28.60','0.729','#18.41','0.090','~54.71'),
 @('TriPS-T (Ours)','#29.03','#0.789','26.66','~0.113','53.65','#27.45','~0.754','35.13','0.161','~52.63','#31.20','~0.809','~17.28','~0.060','~63.52','#29.95','#0.804','25.02','~0.084','51.12'),
 @('TriPS-G (Ours)','28.55','0.762','#22.18','#0.107','#61.69','~27.38','#0.762','#28.22','#0.154','#53.92','#31.20','#0.813','#15.89','#0.059','#64.37','~29.60','~0.782','~21.21','#0.074','#61.92'))
$div2k=@(
 @('ReSample','20.55','0.535','75.67','0.238','26.82','19.54','0.459','119.22','0.372','20.57','21.47','0.588','61.79','0.222','31.17','21.74','0.559','80.79','0.221','24.22'),
 @('FlowChef','22.08','0.561','47.47','0.213','41.70','20.88','0.508','84.28','0.297','35.94','19.62','0.482','74.01','0.366','41.24','21.57','0.532','52.84','0.251','39.52'),
 @('FlowDPS','22.14','0.545','35.18','0.175','39.87','21.07','0.469','48.36','#0.251','33.47','19.88','0.473','52.23','0.322','40.01','20.46','0.473','58.56','0.307','36.80'),
 @('FLAIR','~22.90','0.592','41.23','0.167','42.30','21.12','~0.520','~42.16','0.256','46.23','23.90','0.614','22.17','0.129','52.24','22.70','0.561','32.26','0.157','~44.96'),
 @('TriPS-T (Ours)','#23.05','#0.607','~31.80','#0.158','~45.24','#21.27','0.518','43.74','~0.255','~50.14','#26.29','#0.728','~15.49','#0.066','~59.16','~23.94','#0.646','~28.57','~0.123','44.47'),
 @('TriPS-G (Ours)','22.78','~0.594','#27.84','~0.163','#50.14','~21.13','#0.531','#37.48','0.257','#52.26','~26.19','~0.715','#14.94','#0.066','#59.89','#23.97','~0.644','#26.88','#0.121','#45.19'))
$mw=168;$cw=86;$rowH=34;$hdr1=38;$hdr2=36;$grp=32
$W=$mw+$cw*20+2; $H=[int]($hdr1+$hdr2+$rowH+($grp+6*$rowH)*2+6)
$bmp=New-Object System.Drawing.Bitmap $W,$H; $g=[System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias; $g.TextRenderingHint=[System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit; $g.Clear([System.Drawing.Color]::White)
$reg=[System.Drawing.FontStyle]::Regular; $bld=[System.Drawing.FontStyle]::Bold
$fReg=New-Object System.Drawing.Font('Segoe UI',[single]10.5,$reg); $fBold=New-Object System.Drawing.Font('Segoe UI',[single]10.5,$bld); $fGrp=New-Object System.Drawing.Font('Segoe UI',[single]11,$bld)
$brInk=New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(31,39,51))
$brHdr=New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(241,244,248))
$brGrp=New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(232,236,242))
$brT=New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(252,235,235))
$brG=New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(234,247,238))
$penLine=New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(225,229,235))
$penThick=New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(120,128,140),[single]1.4)
$penU=New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(31,39,51),[single]1.1)
$sfC=New-Object System.Drawing.StringFormat; $sfC.Alignment='Center'; $sfC.LineAlignment='Center'
$sfL=New-Object System.Drawing.StringFormat; $sfL.Alignment='Near'; $sfL.LineAlignment='Center'
function Rect($x,$y,$w,$h){ return [System.Drawing.RectangleF]::new([single]$x,[single]$y,[single]$w,[single]$h) }
function CellX($i){ return [int]($mw+$cw*$i) }
function DrawMark($txt,$x,$y,$w,$h){
  $f=$fReg;$u=$false
  if($txt.StartsWith('#')){$f=$fBold;$txt=$txt.Substring(1)} elseif($txt.StartsWith('~')){$u=$true;$txt=$txt.Substring(1)}
  $g.DrawString($txt,$f,$brInk,(Rect $x $y $w $h),$sfC)
  if($u){ $sz=$g.MeasureString($txt,$f); $lx=$x+($w-$sz.Width)/2; $ly=$y+$h/2+$sz.Height/2-2; $g.DrawLine($penU,[single]$lx,[single]$ly,[single]($lx+$sz.Width),[single]$ly) }
}
$y=0
$g.FillRectangle($brGrp,0,$y,$W,$hdr1); $g.DrawString('Flow Matching Model (SD3.5-M)',$fGrp,$brInk,(Rect 0 $y $W $hdr1),$sfC); $y+=$hdr1
$g.FillRectangle($brHdr,0,$y,$W,$hdr2); $g.DrawString('Method',$fBold,$brInk,(Rect 8 $y ($mw-8) $hdr2),$sfL)
for($t=0;$t -lt 4;$t++){ $g.DrawString($tasks[$t],$fBold,$brInk,(Rect (CellX ($t*5)) $y ($cw*5) $hdr2),$sfC) }
$y+=$hdr2
$g.FillRectangle($brHdr,0,$y,$W,$rowH)
for($t=0;$t -lt 4;$t++){ for($m=0;$m -lt 5;$m++){ DrawMark ($metrics[$m]+$arrows[$m]) (CellX ($t*5+$m)) $y $cw $rowH } }
$y+=$rowH
function DrawGroup($label,$rows){
  $g.FillRectangle($brGrp,0,$script:y,$W,$grp); $g.DrawString($label,$fGrp,$brInk,(Rect 0 $script:y $W $grp),$sfC); $script:y+=$grp
  foreach($r in $rows){
    $isT=$r[0].StartsWith('TriPS-T'); $isG=$r[0].StartsWith('TriPS-G')
    if($isT){$g.FillRectangle($brT,0,$script:y,$W,$rowH)} elseif($isG){$g.FillRectangle($brG,0,$script:y,$W,$rowH)}
    $mf=$fReg; if($isT -or $isG){$mf=$fBold}
    $g.DrawString($r[0],$mf,$brInk,(Rect 8 $script:y ($mw-8) $rowH),$sfL)
    for($c=1;$c -le 20;$c++){ DrawMark $r[$c] (CellX ($c-1)) $script:y $cw $rowH }
    $g.DrawLine($penLine,0,$script:y,$W,$script:y); $script:y+=$rowH
  }
}
DrawGroup 'FFHQ (768 x 768)' $ffhq
DrawGroup 'DIV2K (768 x 768)' $div2k
$g.DrawLine($penThick,$mw,$hdr1,$mw,$H)
for($t=1;$t -lt 4;$t++){ $g.DrawLine($penThick,(CellX ($t*5)),$hdr1,(CellX ($t*5)),$H) }
$g.DrawRectangle($penThick,0,0,$W-1,$H-1)
$g.Dispose(); $bmp.Save($out,[System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
Write-Output "Saved $out"
