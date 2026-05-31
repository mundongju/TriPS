#!/usr/bin/env bash
# Render every figure PDF in static/pdfs/ to a crisp PNG in static/images/.
# The page works without this step (it falls back to embedding the PDFs), but
# running it once gives sharp, mobile-friendly raster figures.
#
#   bash convert_figures.sh          # default 200 DPI
#   DPI=300 bash convert_figures.sh  # higher resolution
#
# Uses whichever tool is available: pdftocairo / pdftoppm (poppler) > magick/convert (ImageMagick) > gs (Ghostscript).
set -e
cd "$(dirname "$0")"
mkdir -p static/images
DPI="${DPI:-200}"

render() {  # $1 = pdf, $2 = out.png
  local f="$1" out="$2" pre="${2%.png}"
  if   command -v pdftocairo >/dev/null 2>&1; then pdftocairo -png -r "$DPI" -singlefile "$f" "$pre"
  elif command -v pdftoppm  >/dev/null 2>&1; then pdftoppm  -png -r "$DPI" -singlefile "$f" "$pre"
  elif command -v magick    >/dev/null 2>&1; then magick -density "$DPI" "${f}[0]" -background white -flatten "$out"
  elif command -v convert   >/dev/null 2>&1; then convert -density "$DPI" "${f}[0]" -background white -flatten "$out"
  elif command -v gs        >/dev/null 2>&1; then gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r"$DPI" -dFirstPage=1 -dLastPage=1 -o "$out" "$f"
  else echo "ERROR: install poppler (pdftocairo), ImageMagick (magick), or Ghostscript (gs)."; exit 1
  fi
}

for f in static/pdfs/*.pdf; do
  name="$(basename "$f" .pdf)"
  [ "$name" = "TriPS_arxiv" ] && continue          # skip the full paper
  echo "[render] $name"
  render "$f" "static/images/$name.png"
done
echo "Done -> static/images/*.png"
