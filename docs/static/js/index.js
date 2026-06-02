/* TriPS project page behaviour */

/* ---- figure PDF fallback (if a rendered image is missing) ---- */
function pdfFallback(img){
  var pdf = img.getAttribute('data-pdf'); if(!pdf) return;
  var emb = document.createElement('embed');
  emb.src = pdf + '#toolbar=0&navpanes=0&view=FitH'; emb.type = 'application/pdf';
  img.parentNode.replaceChild(emb, img);
}

/* ---- qualitative samples ----
   Each card shows two before/after sliders: top = Measurement vs Ours,
   bottom = FlowDPS vs Ours. Several samples are shown per page (FLAIR style). */
var QBASE = 'static/quali/';
function S(dir, idx, ours){ return {
  meas: QBASE+dir+'/'+idx+'_meas.jpg',
  flow: QBASE+dir+'/'+idx+'_flow.jpg',
  ours: QBASE+dir+'/'+idx+'_ours.jpg',
  oursLabel: ours }; }
var SAMPLES = [
  S('mod','0012','TriPS-G (Ours)'),
  S('srx8','0026','TriPS-G (Ours)'),
  S('mod','0064','TriPS-G (Ours)'),
  S('gaussian_deblur','0038','TriPS-G (Ours)'),
  S('gaussian_deblur','0004','TriPS-G (Ours)'),
  S('gaussian_deblur','0187','TriPS-G (Ours)'),
  S('motion_deblur','0002','TriPS-G (Ours)'),
  S('motion_deblur','0011','TriPS-G (Ours)'),
  S('motion_deblur','0015','TriPS-G (Ours)'),
  S('srx12','0030','TriPS-G (Ours)'),
  S('srx12','0052','TriPS-G (Ours)'),
  S('mod','0240','TriPS-G (Ours)'),
  S('inpainting','0054','TriPS-T (Ours)'),
  S('inpainting','0407','TriPS-T (Ours)'),
  S('inpainting','0120','TriPS-T (Ours)')
];
var PER_PAGE = 3;

/* build one before/after comparison slider into a container */
function buildBA(container, beforeSrc, afterSrc, leftLabel, rightLabel){
  container.innerHTML = '';
  var after  = new Image(); after.className='after';  after.src=afterSrc;
  var before = new Image(); before.className='before'; before.src=beforeSrc;
  var handle = document.createElement('div'); handle.className='handle';
  var knob = document.createElement('div'); knob.className='knob'; knob.innerHTML='&#8596;'; handle.appendChild(knob);
  var ll = document.createElement('div'); ll.className='lbl lbl-l'; ll.textContent=leftLabel;
  var rl = document.createElement('div'); rl.className='lbl lbl-r'; rl.textContent=rightLabel;
  container.appendChild(after); container.appendChild(before);
  container.appendChild(handle); container.appendChild(ll); container.appendChild(rl);

  function setPct(p){ p=Math.max(0,Math.min(100,p));
    before.style.clipPath='inset(0 '+(100-p)+'% 0 0)';
    before.style.webkitClipPath='inset(0 '+(100-p)+'% 0 0)';
    handle.style.left=p+'%'; }
  setPct(50);
  var dragging=false;
  function pctFromEvent(e){ var r=container.getBoundingClientRect();
    var x=(e.touches?e.touches[0].clientX:e.clientX)-r.left; return (x/r.width)*100; }
  function down(e){ dragging=true; setPct(pctFromEvent(e)); e.preventDefault(); }
  function move(e){ if(dragging) setPct(pctFromEvent(e)); }
  function up(){ dragging=false; }
  container.addEventListener('mousedown',down); container.addEventListener('touchstart',down,{passive:false});
  window.addEventListener('mousemove',move);   container.addEventListener('touchmove',move,{passive:false});
  window.addEventListener('mouseup',up);        container.addEventListener('touchend',up);
}

function renderPage(p){
  var grid = document.getElementById('q-grid');
  grid.innerHTML='';
  var start=p*PER_PAGE, end=Math.min(start+PER_PAGE, SAMPLES.length);
  for(var i=start;i<end;i++){
    var s=SAMPLES[i];
    var card=document.createElement('div'); card.className='q-card';
    var top=document.createElement('div'); top.className='ba';
    var bot=document.createElement('div'); bot.className='ba';
    card.appendChild(top); card.appendChild(bot); grid.appendChild(card);
    buildBA(top, s.meas, s.ours, 'Measurement', s.oursLabel);
    buildBA(bot, s.flow, s.ours, 'FlowDPS', s.oursLabel);
  }
  document.getElementById('q-count').textContent = (start+1)+'–'+end+' / '+SAMPLES.length;
  var dots=document.querySelectorAll('.dot-btn');
  dots.forEach(function(d,k){ d.classList.toggle('active', k===p); });
}

document.addEventListener('DOMContentLoaded', function(){
  // figure fallbacks
  document.querySelectorAll('img[data-pdf]').forEach(function(img){
    img.addEventListener('error', function(){ pdfFallback(img); });
    img.addEventListener('load', function(){ if(img.naturalWidth===0) pdfFallback(img); });
  });

  // Table 1 image: if missing, reveal the fallback note
  var t1 = document.getElementById('table1-img');
  if(t1){ t1.addEventListener('error', function(){
    t1.style.display='none';
    var fb=document.getElementById('table1-fallback'); if(fb) fb.style.display='block';
  }); }

  // qualitative gallery (paged, several samples per page)
  if(document.getElementById('quali-viewer')){
    var pages=Math.ceil(SAMPLES.length/PER_PAGE), cur=0;
    var dotsWrap=document.getElementById('q-dots');
    for(var k=0;k<pages;k++){ (function(kk){
      var b=document.createElement('button'); b.className='dot-btn';
      b.addEventListener('click',function(){ cur=kk; renderPage(cur); }); dotsWrap.appendChild(b);
    })(k); }
    document.getElementById('q-prev').addEventListener('click',function(){ cur=(cur-1+pages)%pages; renderPage(cur); });
    document.getElementById('q-next').addEventListener('click',function(){ cur=(cur+1)%pages; renderPage(cur); });
    document.addEventListener('keydown',function(e){ if(e.key==='ArrowLeft') document.getElementById('q-prev').click();
      else if(e.key==='ArrowRight') document.getElementById('q-next').click(); });
    renderPage(0);
  }

  // bibtex copy
  var btn=document.getElementById('copy-bib');
  if(btn){ btn.addEventListener('click',function(){
    navigator.clipboard.writeText(document.getElementById('bibtex-text').innerText).then(function(){
      var o=btn.innerHTML; btn.innerHTML='<span class="icon"><i class="fas fa-check"></i></span><span>Copied</span>';
      setTimeout(function(){ btn.innerHTML=o; },1500);
    });
  }); }
});
