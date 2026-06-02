/* TriPS project page behaviour */

/* ---- figure PDF fallback (if a rendered image is missing) ---- */
function pdfFallback(img){
  var pdf = img.getAttribute('data-pdf'); if(!pdf) return;
  var emb = document.createElement('embed');
  emb.src = pdf + '#toolbar=0&navpanes=0&view=FitH'; emb.type = 'application/pdf';
  img.parentNode.replaceChild(emb, img);
}

/* ---- qualitative samples (FLAIR-style) ----
   task order: srx8, gaussian_deblur, motion_deblur, srx12, inpainting
   each row of the viewer: top = Measurement vs Ours, bottom = FlowDPS vs Ours */
var QBASE = 'static/quali/';
function S(task, label, idx, ours){ return {task:task, label:label, idx:idx, ours:ours,
  meas:QBASE+task+'/'+idx+'_meas.jpg', flow:QBASE+task+'/'+idx+'_flow.jpg', oursImg:QBASE+task+'/'+idx+'_ours.jpg'}; }
var SAMPLES = [
  S('srx8','Super-Resolution x8','0023','TriPS-G (Ours)'),
  S('srx8','Super-Resolution x8','0026','TriPS-G (Ours)'),
  S('srx8','Super-Resolution x8','0042','TriPS-G (Ours)'),
  S('gaussian_deblur','Gaussian Deblurring','0038','TriPS-G (Ours)'),
  S('gaussian_deblur','Gaussian Deblurring','0004','TriPS-G (Ours)'),
  S('gaussian_deblur','Gaussian Deblurring','0187','TriPS-G (Ours)'),
  S('motion_deblur','Motion Deblurring','0002','TriPS-G (Ours)'),
  S('motion_deblur','Motion Deblurring','0011','TriPS-G (Ours)'),
  S('motion_deblur','Motion Deblurring','0015','TriPS-G (Ours)'),
  S('srx12','Super-Resolution x12','0030','TriPS-G (Ours)'),
  S('srx12','Super-Resolution x12','0052','TriPS-G (Ours)'),
  S('srx12','Super-Resolution x12','0183','TriPS-G (Ours)'),
  S('inpainting','Inpainting','0054','TriPS-T (Ours)'),
  S('inpainting','Inpainting','0407','TriPS-T (Ours)'),
  S('inpainting','Inpainting','0120','TriPS-T (Ours)')
];

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

function renderSample(i){
  var s = SAMPLES[i];
  document.getElementById('q-task').textContent = s.label;
  document.getElementById('q-count').textContent = (i+1)+' / '+SAMPLES.length;
  buildBA(document.getElementById('ba-top'), s.meas, s.oursImg, 'Measurement', s.ours);
  buildBA(document.getElementById('ba-bot'), s.flow, s.oursImg, 'FlowDPS', s.ours);
  var dots = document.querySelectorAll('.dot-btn');
  dots.forEach(function(d,k){ d.classList.toggle('active', k===i); });
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

  // qualitative gallery
  if(document.getElementById('quali-viewer')){
    var cur=0;
    var dotsWrap=document.getElementById('q-dots');
    SAMPLES.forEach(function(_,k){ var b=document.createElement('button'); b.className='dot-btn';
      b.addEventListener('click',function(){ cur=k; renderSample(cur); }); dotsWrap.appendChild(b); });
    document.getElementById('q-prev').addEventListener('click',function(){ cur=(cur-1+SAMPLES.length)%SAMPLES.length; renderSample(cur); });
    document.getElementById('q-next').addEventListener('click',function(){ cur=(cur+1)%SAMPLES.length; renderSample(cur); });
    document.addEventListener('keydown',function(e){ if(e.key==='ArrowLeft') document.getElementById('q-prev').click();
      else if(e.key==='ArrowRight') document.getElementById('q-next').click(); });
    renderSample(0);
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
