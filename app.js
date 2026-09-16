const models = [
  {name:'Qwen3-TTS', file:1}, {name:'MOSS', file:2}, {name:'Finetune', file:3},
  {name:'VoxCPM2', file:4}, {name:'Ours', file:5}, {name:'Ming-omni-tts', file:6}
];

const samples = [
  {id:'S01',dataset:'InstructTTSEval',condition:'APS',text:'你自己不是有家吗？你老在我这门口转悠什么呀？',instruction:'性别：男性。音高：男性低沉，句末上扬。语速：较快。音量：谈话音量，略显响亮。年龄：中年。清晰度：发音清晰。流畅度：表达流畅。口音：带有北方口音。音色质感：略显粗犷。情绪：略带不耐烦与质问。语调：质问语气，句末上扬明显。性格：略显急躁，态度直接。'},
  {id:'S02',dataset:'InstructTTSEval',condition:'APS',text:'都有可能，但是现在不可能。这就是我们这代人值得自豪而又严峻的历史使命。',instruction:'性别：男性。音高：沉稳中低音，强调处上扬。语速：舒缓庄重，停顿分明有力。音量：洪亮饱满，关键处着重发力。年龄：中年至老年。清晰度：发音清晰标准，字字铿锵。音色浑厚共鸣，情绪严肃郑重，充满历史使命感。'},
  {id:'S03',dataset:'InstructTTSEval',condition:'APS',text:'改变是可以的，但是集体个体的事情，那是原则问题。',instruction:'性别：女性。音高：中高音域，句末下降。语速：由平稳转为强调性放缓。音量：常规对话，强调处增强。年龄：中年。吐字清晰精准，标准普通话。音色清亮、质感坚实。态度严肃坚决，关键语句以降调强调。'},
  {id:'S04',dataset:'InstructTTSEval',condition:'DSD',text:'你怎么还不明白呀，妈。他是一个人活生生的人。他没招我，没惹我，现在要死在我面前了，我一伸手我就能救他。我能不救吗？',instruction:'体现出青年女性的声线，音色明亮，声音在表达焦急时变得高亢。从急迫到激愤，语速加快，语调急切且多用升调，显得更加坚定和直率。'},
  {id:'S05',dataset:'InstructTTSEval',condition:'DSD',text:'你还挺有理的，还你们老家，还祖祖辈辈，这是你们老家吗？',instruction:'表达态度时展现出直率泼辣的特征，声音中带有明显的激动感和质问语气。使用标准普通话，吐字流畅无停顿。'},
  {id:'S06',dataset:'InstructTTSEval',condition:'DSD',text:'机器人时代已经来临，他们或许还不能拿着鸡毛掸子打扫你的家，但是他们或多或少已经占领了工厂，生产各种东西，从灯泡、汽车到起重机。不过有趣的是，机器人还是有件事做不来，就是生产机器人。这究竟是怎么办到的？',instruction:'口音为标准普通话，音量充沛而稳定，音色醇厚富有磁性，语速舒缓平稳，节奏感明确，语气沉稳干练，吐字清晰，字正腔圆。'},
  {id:'S07',dataset:'InstructTTSEval',condition:'RP',text:'我们是国营企业，从进货到生产到服务部门，都要经过检查的。',instruction:'用一个企业管理者在会议上发言那样的语气来朗读。'},
  {id:'S08',dataset:'InstructTTSEval',condition:'RP',text:'靠着普通网友把网页翻成自己的母语，这个网站现在有超过70个语言版本。',instruction:'选用稳定而明确的声调传达清晰的信息，为观众提供最新动态。'},
  {id:'S09',dataset:'InstructTTSEval',condition:'RP',text:'哎，不行，我看见照片我就会想起灵儿那张恶心的脸。我待不下去了嘛。',instruction:'用急促而高昂的声音抱怨一整天的工作疲惫，声音中透露出无法掩饰的厌倦和无奈。'},
  {id:'S10',dataset:'Mint',condition:'Hard',text:'你一定要答应我，无论发生什么事，都要先保护好自己。多穿点衣服，晚上早点回家，千万别一个人走夜路。记住，你的安全比什么都重要，听到了吗？',instruction:'你是一个温暖的人，正在用一种恳切、催促的语气劝说你关心的人注意安全。在温暖的表面之下，隐藏着你对他们安危的深深恐惧。'}
];

const $ = id => document.getElementById(id);
let current = 0, filter = 'ALL', activeAudio = null;
const formatTime = value => Number.isFinite(value) ? `${Math.floor(value/60)}:${String(Math.floor(value%60)).padStart(2,'0')}` : '0:00';
const bars = seed => Array.from({length:60},(_,i)=>18+((Math.sin((i+seed)*1.71)+1)*16+((i*seed)%13))).map(h=>`<i class="bar" style="--h:${Math.min(94,h)}%"></i>`).join('');

function renderFilters(){
  $('filters').innerHTML=['ALL','APS','DSD','RP','HARD'].map(x=>`<button class="filter ${filter===x?'active':''}" data-filter="${x}">${x}</button>`).join('');
  $('filters').querySelectorAll('button').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;const first=samples.findIndex(s=>filter==='ALL'||s.condition.toUpperCase()===filter);if(first>=0)current=first;render();});
}
function renderNav(){
  const visible=samples.map((s,i)=>({s,i})).filter(({s})=>filter==='ALL'||s.condition.toUpperCase()===filter);
  $('sampleList').innerHTML=visible.map(({s,i})=>`<button class="sample-button ${i===current?'active':''}" data-index="${i}" title="${s.text}"><span class="sample-no">${String(i+1).padStart(2,'0')}</span><span class="sample-title">${s.text}</span><span class="sample-type">${s.condition}</span></button>`).join('');
  $('sampleList').querySelectorAll('button').forEach(b=>b.onclick=()=>{current=Number(b.dataset.index);stopAudio();render();document.querySelector('.prompt-card').scrollIntoView({behavior:'smooth',block:'start'});});
}
function stopAudio(){if(activeAudio){activeAudio.pause();activeAudio=null;}}
function player(model,index,sample){
  const wave=bars((current+2)*(index+5));
  return `<article class="player" data-player="${index}"><div class="model-name">${model.name}<span class="model-tag">MODEL ${String(index+1).padStart(2,'0')}</span></div><button class="play" aria-label="播放 ${model.name}"><svg viewBox="0 0 20 20"><path d="M5 3.5v13l11-6.5z"/></svg></button><div class="wave-wrap" role="slider" tabindex="0" aria-label="${model.name} 播放进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="wave">${wave}</div><div class="progress-fill" style="--progress:0%">${wave}</div></div><div class="time"><span>0:00</span> / <span>0:00</span></div><audio preload="metadata" src="audio/${sample.id}_model_${model.file}.wav"></audio></article>`;
}
function bindPlayers(){
  document.querySelectorAll('.player').forEach(row=>{
    const audio=row.querySelector('audio'),button=row.querySelector('.play'),track=row.querySelector('.wave-wrap'),times=row.querySelectorAll('.time span');
    const sync=()=>{const p=audio.duration?audio.currentTime/audio.duration*100:0;row.querySelector('.progress-fill').style.setProperty('--progress',`${p}%`);track.setAttribute('aria-valuenow',Math.round(p));times[0].textContent=formatTime(audio.currentTime);times[1].textContent=formatTime(audio.duration)};
    audio.onloadedmetadata=sync;audio.ontimeupdate=sync;audio.onended=()=>{row.classList.remove('playing');button.innerHTML='<svg viewBox="0 0 20 20"><path d="M5 3.5v13l11-6.5z"/></svg>';sync()};
    button.onclick=()=>{if(audio.paused){if(activeAudio&&activeAudio!==audio)activeAudio.pause();document.querySelectorAll('.player').forEach(x=>x.classList.remove('playing'));document.querySelectorAll('.play').forEach(x=>x.innerHTML='<svg viewBox="0 0 20 20"><path d="M5 3.5v13l11-6.5z"/></svg>');audio.play();activeAudio=audio;row.classList.add('playing');button.innerHTML='<svg viewBox="0 0 20 20"><path d="M5 4h3v12H5zm7 0h3v12h-3z"/></svg>'}else{audio.pause();row.classList.remove('playing');button.innerHTML='<svg viewBox="0 0 20 20"><path d="M5 3.5v13l11-6.5z"/></svg>'}};
    audio.onpause=()=>{row.classList.remove('playing');button.innerHTML='<svg viewBox="0 0 20 20"><path d="M5 3.5v13l11-6.5z"/></svg>'};
    const seek=e=>{const r=track.getBoundingClientRect();audio.currentTime=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*audio.duration};track.onclick=seek;track.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();audio.currentTime=Math.max(0,Math.min(audio.duration,audio.currentTime+(e.key==='ArrowRight'?5:-5)))}};
  });
}
function render(){
  renderFilters();renderNav();const s=samples[current];$('sampleCount').textContent=`${String(current+1).padStart(2,'0')} / ${String(samples.length).padStart(2,'0')}`;$('conditionBadge').textContent=s.condition.toUpperCase();$('datasetLabel').textContent=s.dataset;$('promptIndex').textContent=s.id;$('targetText').textContent=s.text;$('instruction').textContent=s.instruction;$('players').innerHTML=models.map((m,i)=>player(m,i,s)).join('');bindPlayers();
}
render();
