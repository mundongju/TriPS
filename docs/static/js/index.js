/* TriPS project page — behaviour */

// If a figure PNG is not present yet (run convert_figures.* to generate them),
// gracefully fall back to embedding the source PDF so the page is never broken.
function pdfFallback(img){
  var pdf = img.getAttribute('data-pdf');
  if(!pdf){ return; }
  var emb = document.createElement('embed');
  emb.src = pdf + '#toolbar=0&navpanes=0&view=FitH';
  emb.type = 'application/pdf';
  img.parentNode.replaceChild(emb, img);
}

document.addEventListener('DOMContentLoaded', function(){
  // attach fallback to every figure image
  document.querySelectorAll('img[data-pdf]').forEach(function(img){
    img.addEventListener('error', function(){ pdfFallback(img); });
    // also trigger fallback if the PNG is a 0-byte placeholder that loaded but has no size
    img.addEventListener('load', function(){
      if(img.naturalWidth === 0){ pdfFallback(img); }
    });
  });

  // bulma carousel (loaded from CDN)
  if(window.bulmaCarousel){
    bulmaCarousel.attach('.carousel', {
      slidesToShow: 1, slidesToScroll: 1, loop: true, infinite: true,
      autoplay: true, autoplaySpeed: 5000, pauseOnHover: true, navigation: true, pagination: true
    });
  }

  // copy bibtex
  var btn = document.getElementById('copy-bib');
  if(btn){
    btn.addEventListener('click', function(){
      var t = document.getElementById('bibtex-text').innerText;
      navigator.clipboard.writeText(t).then(function(){
        var old = btn.innerHTML; btn.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span><span>Copied</span>';
        setTimeout(function(){ btn.innerHTML = old; }, 1500);
      });
    });
  }
});
