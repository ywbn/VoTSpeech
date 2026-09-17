const models = [
  {name:'VoTspeech', key:'ours'},
  {name:'Ming-Omni-TTS-0.5B', key:'ming'},
  {name:'Qwen3-TTS-12Hz-1.7B-VD', key:'qwen3_tts'},
  {name:'MOSS-VoiceGenerator', key:'moss_zh'},
  {name:'VoxCPM2', key:'voxcpm2'},
  {name:'Finetune', key:'finetune'}
];

const samples = [
  {
    "id": "S01",
    "sourceId": "zh_11",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "俗话说，风风雨雨一家女嘛，可我从小风里来浪里去，不过是为了一网鱼，从来没想到淋一场大雨是为国为人民，这不是一个好机会吗？",
    "instruction": "性别: 女性嗓音特征. 音高: 女性中高音区，语调起伏较大. 语速: 初始语速平稳，后段逐渐加快. 音量: 音量清晰洪亮，随情感增强. 年龄: 青年女性. 清晰度: 吐字清晰准确，发音标准. 流畅度: 言语流畅连贯，无迟疑. 口音: 标准普通话，略带舞台腔. 音色质感: 音色清澈明亮，富有力度. 情绪: 由沉稳转向激昂，充满信念. 语调: 抑扬顿挫分明，极富表现力. 性格: 果敢坚定，充满理想与热忱."
  },
  {
    "id": "S02",
    "sourceId": "zh_29",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "对啊就是你的茶头要吃米啦，有听到吗？",
    "instruction": "性别: 男性. 音高: 男性常规音高，句末略有上扬. 语速: 初始从容，后半句略有加快. 音量: 日常对话音量，略带强调感. 年龄: 中年男性. 清晰度: 发音清晰，吐字有力. 流畅度: 表达流畅，一气呵成. 口音: 带有闽南语区口音特征. 音色质感: 略带颗粒感，声音坚实. 情绪: 轻松提醒，略带期待. 语调: 陈述中带有提醒意味，结尾疑问略带确认. 性格: 直接爽朗，略显随和."
  },
  {
    "id": "S03",
    "sourceId": "zh_387",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "刚刚长大了，一定会开飞船的。",
    "instruction": "性别: 女性童声. 音高: 典型女童高音. 语速: 语速舒缓平稳. 音量: 音量初始轻柔，后趋响亮清晰. 年龄: 幼年儿童. 清晰度: 吐字清晰圆润. 流畅度: 言语流畅自然. 口音: 标准普通话. 音色质感: 音色清脆甜美，略带稚气. 情绪: 由初始沉吟转为坚定自信与向往. 语调: 起始平缓，后半句语调上扬，强调肯定. 性格: 天真乐观，富有梦想."
  },
  {
    "id": "S04",
    "sourceId": "zh_417",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "请教彭长官，胡主任的二十万人马，为何消灭不了共军的两万人？请问您使的是什么战术？",
    "instruction": "性别: 男性. 音高: 男性低音域，疑问句末尾上扬. 语速: 语速先稳后略快，强调质问. 音量: 音量初始平稳，情绪激动时显著增强. 年龄: 中年. 清晰度: 吐字清晰标准. 流畅度: 表达流畅无阻. 口音: 标准普通话，正式语境. 音色质感: 音质浑厚，坚实有力. 情绪: 由郑重趋向激昂与不解. 语调: 质问语调，情绪起伏鲜明. 性格: 果敢自信，态度鲜明."
  },
  {
    "id": "S05",
    "sourceId": "zh_634",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "就算你学的不是相关专业，诚实做人，认真做事的基本道德应该有吧。",
    "instruction": "性别: 男性. 音高: 男性中音，句尾音调上扬. 语速: 语速适中，关键处加重. 音量: 音量较高，语气有力. 年龄: 中青年. 清晰度: 发音清晰，字句分明. 流畅度: 表达流畅，无明显卡顿. 口音: 标准普通话. 音色质感: 音质坚实，略显硬朗. 情绪: 严肃说理，略带责备. 语调: 语调肯定，句末反问. 性格: 自信坚定，态度鲜明."
  },
  {
    "id": "S06",
    "sourceId": "zh_755",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "按您说的，就把这个马家给开除了。他刚多大呀，这么点儿小孩就给推向社会当代理青年。那对社会也不负责呀。",
    "instruction": "性别: 男性. 音高: 男性中音区，语调高昂且起伏大. 语速: 语速偏快，带有急切感. 音量: 音量较大，富有力度. 年龄: 中老年. 清晰度: 吐字清晰，字字铿锵. 流畅度: 语言流畅，一气呵成. 口音: 标准普通话. 音色质感: 音色略带沙哑，但饱满有力. 情绪: 情绪激动，表达强烈不满与忧虑. 语调: 质问语气强烈，句末多上扬. 性格: 耿直刚正，易于激动."
  },
  {
    "id": "S07",
    "sourceId": "zh_833",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "啊，其实特别简单，我有一个详细的教程，我可以发给你啊。这样吧，我也帮你准备一份，到时候你看是用你自己的还是我这个枪手做的，都一样，反正啊心意在那儿了。",
    "instruction": "性别: 女性. 音高: 女性高音区，声线明亮. 语速: 语速偏快，节奏流畅. 音量: 音量清晰，日常交流水平. 年龄: 青年女性. 清晰度: 发音清晰，吐字标准. 流畅度: 表达流畅，一气呵成. 口音: 标准普通话. 音色质感: 音色清亮，略带甜美. 情绪: 热情友好，积极主动. 语调: 语调轻快，富有感染力. 性格: 外向开朗，乐于助人."
  },
  {
    "id": "S08",
    "sourceId": "zh_838",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "这事儿跟你没关系。这么跟你说吧，我呢是张兰香，左请右请，请回他们家住的一位重要级人物。我说啥他就听啥，这么跟你说，你明白吗？",
    "instruction": "性别: 年轻女性. 音高: 女性高音域，音调清亮. 语速: 语速较快，节奏紧凑. 音量: 正常谈话音量，略带强调语气. 年龄: 少女. 清晰度: 吐字清晰准确. 流畅度: 表达流畅，一气呵成. 口音: 标准普通话，无明显地域特征. 音色质感: 音色清脆，略显单薄尖细. 情绪: 自信得意，略带不耐与傲慢. 语调: 多用上扬语调，语气肯定且具挑战性. 性格: 自信张扬，略显骄纵急躁."
  },
  {
    "id": "S09",
    "sourceId": "zh_888",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "谷志勇，你自己不求上进，别耽误你女儿。",
    "instruction": "性别: 女性. 音高: 女性声线偏高，句末音调上扬. 语速: 语速较快，略显急促. 音量: 音量较大，充满力量. 年龄: 中年女性. 清晰度: 吐字清晰准确. 流畅度: 表达流畅，一气呵成. 口音: 标准普通话. 音色质感: 音色清亮，略带锐利. 情绪: 情绪激动，带有不满与指责. 语调: 语调高亢有力，强调意味明显. 性格: 性格直爽，言辞急切."
  },
  {
    "id": "S10",
    "sourceId": "zh_977",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "他们先跟人家编剧把合同签了，然后呢就拖着不付钱，哄人家编剧出东西。那编剧就想我这合同都签了，付钱还不是早晚的事儿，就把他们要的东西给他们了，他们就拿这东西偷偷的去找熟悉的平台看，平台要是认了他们就付款，平台要是不认，他们就耍赖。",
    "instruction": "性别: 典型成年男性语音. 音高: 男性常态音高，偶因强调而上扬. 语速: 整体语速偏快，节奏鲜明. 音量: 正常谈话音量，强调处略有增强. 年龄: 青壮年阶段. 清晰度: 吐字清晰准确，发音标准. 流畅度: 言语高度流畅，无明显停顿. 口音: 标准普通话，几无口音痕迹. 音色质感: 音质清亮坚实，富有穿透力. 情绪: 带有批判意味，略显不满与愤慨. 语调: 陈述语气坚定，语流富于顿挫. 性格: 自信果断，表达直接有力."
  },
  {
    "id": "S11",
    "sourceId": "zh_505",
    "dataset": "InstructTTSEval",
    "condition": "APS",
    "text": "自从你爸爸去世以后，我把所有的心血都放在你们兄弟俩身上，艰难的经营着这家医院。我不允许任何人来破坏它。",
    "instruction": "性别: 女性. 音高: 女性常态音高，结尾激动拔高. 语速: 语速由平缓渐快，结尾急促. 音量: 音量由正常渐强，结尾高亢. 年龄: 中老年. 清晰度: 吐字清晰，字字铿锵. 流畅度: 表达流畅，一气呵成. 口音: 标准普通话，地域特征不明显. 音色质感: 音色坚实，激动时略显锐利. 情绪: 由隐忍悲情转为愤慨坚决. 语调: 语调由平缓转激昂，富含警告意味. 性格: 意志刚强，决断不容挑战."
  },
  {
    "id": "S12",
    "sourceId": "zh_11",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "俗话说，风风雨雨一家女嘛，可我从小风里来浪里去，不过是为了一网鱼，从来没想到淋一场大雨是为国为人民，这不是一个好机会吗？",
    "instruction": "展现出果敢坚定、充满理想与热忱的青年女性音色特征,音高保持中高音区,语调抑扬顿挫分明,发音标准且吐字清晰,语句显得流畅连贯,表现出由沉稳转向激昂的情绪。"
  },
  {
    "id": "S13",
    "sourceId": "zh_29",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "对啊就是你的茶头要吃米啦，有听到吗？",
    "instruction": "展现出带颗粒感且坚实的男声特质,语速先从容后略加快,随着语末略有上扬,轻松且略带期待,表达语句时声音略强调显得更为有力。"
  },
  {
    "id": "S14",
    "sourceId": "zh_387",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "刚刚长大了，一定会开飞船的。",
    "instruction": "以天真乐观和充满梦想的女性童声为基础,音色清脆甜美,流畅地表达出未来探索者的信念。语音从轻柔开始,并随着情绪的变化变得响亮清晰,语调通过明显的尾音上扬来强调肯定性。"
  },
  {
    "id": "S15",
    "sourceId": "zh_417",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "请教彭长官，胡主任的二十万人马，为何消灭不了共军的两万人？请问您使的是什么战术？",
    "instruction": "声音采用标准普通话发音,清晰度高且发音明确。语音的整体语调要质问性强,随着情绪的起伏鲜明变化,展现出果敢自信的态度。音量在初始时保持平稳,并在情绪激动时明显增大。"
  },
  {
    "id": "S16",
    "sourceId": "zh_634",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "就算你学的不是相关专业，诚实做人，认真做事的基本道德应该有吧。",
    "instruction": "声音应具备中青年男性的特质,以坚实而略显硬朗的音质表述。保持严肃说理的情绪基调,语气中带有轻微的责备。音量要充足,语调肯定并在句尾部分上扬。表达流畅,清晰度高。"
  },
  {
    "id": "S17",
    "sourceId": "zh_755",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "按您说的，就把这个马家给开除了。他刚多大呀，这么点儿小孩就给推向社会当代理青年。那对社会也不负责呀。",
    "instruction": "语音表现需富有力度,中音区的男性声线卷积着明显的情绪波动,音量较大,使用标准普通话表达质询时的激动和刚正不阿的性格。"
  },
  {
    "id": "S18",
    "sourceId": "zh_833",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "啊，其实特别简单，我有一个详细的教程，我可以发给你啊。这样吧，我也帮你准备一份，到时候你看是用你自己的还是我这个枪手做的，都一样，反正啊心意在那儿了。",
    "instruction": "以标准普通话为基础,音色需显得清亮且略带甜美,语速保持偏快但自然流畅,语气外向开朗,富有感染力。"
  },
  {
    "id": "S19",
    "sourceId": "zh_838",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "这事儿跟你没关系。这么跟你说吧，我呢是张兰香，左请右请，请回他们家住的一位重要级人物。我说啥他就听啥，这么跟你说，你明白吗？",
    "instruction": "保持语气的自信与傲慢,采用正常的音量,但在强调时稍显突出,运用标准的普通话音,清晰表达中不失略显骄纵的色彩,语速快速但准确。"
  },
  {
    "id": "S20",
    "sourceId": "zh_888",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "谷志勇，你自己不求上进，别耽误你女儿。",
    "instruction": "保持中年女性的直爽风格,语音需体现急切与焦急,在关键句末使用偏高的女性声线上扬,强调部分语句时音质略带锐利,充满不满与指责的情绪,音量较大。"
  },
  {
    "id": "S21",
    "sourceId": "zh_977",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "他们先跟人家编剧把合同签了，然后呢就拖着不付钱，哄人家编剧出东西。那编剧就想我这合同都签了，付钱还不是早晚的事儿，就把他们要的东西给他们了，他们就拿这东西偷偷的去找熟悉的平台看，平台要是认了他们就付款，平台要是不认，他们就耍赖。",
    "instruction": "展现青壮年男性语音特质,以标准普通话为基础,发音清晰且流畅,音色清亮且有穿透力,在陈述时保持正常音量但偶尔在强调时有所增强,语气直接且果断。"
  },
  {
    "id": "S22",
    "sourceId": "zh_505",
    "dataset": "InstructTTSEval",
    "condition": "DSD",
    "text": "自从你爸爸去世以后，我把所有的心血都放在你们兄弟俩身上，艰难的经营着这家医院。我不允许任何人来破坏它。",
    "instruction": "声音应展现出中老年女性的特质,吐字清晰,前半段语速平缓,随着情感变化逐渐加快;音量正常开始,语气由隐忍转为愤慨,最终达到激昂高亢的状态。"
  },
  {
    "id": "S23",
    "sourceId": "zh_11",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "俗话说，风风雨雨一家女嘛，可我从小风里来浪里去，不过是为了一网鱼，从来没想到淋一场大雨是为国为人民，这不是一个好机会吗？",
    "instruction": "在辩论赛中陈述观点,从冷静分析到激昂陈述,充满对未来的期盼。"
  },
  {
    "id": "S24",
    "sourceId": "zh_29",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "对啊就是你的茶头要吃米啦，有听到吗？",
    "instruction": "结尾略微上扬,带着轻松笑意提醒对方别忘记,犹如一个老朋友的体贴关心。"
  },
  {
    "id": "S25",
    "sourceId": "zh_387",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "刚刚长大了，一定会开飞船的。",
    "instruction": "小公主满怀憧憬地描述她心中的理想国度。"
  },
  {
    "id": "S26",
    "sourceId": "zh_417",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "请教彭长官，胡主任的二十万人马，为何消灭不了共军的两万人？请问您使的是什么战术？",
    "instruction": "在法庭上据理力争的律师。"
  },
  {
    "id": "S27",
    "sourceId": "zh_634",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "就算你学的不是相关专业，诚实做人，认真做事的基本道德应该有吧。",
    "instruction": "男人在和对方讨论问题时,声音坚定有力,适中语速,声音清晰,每个字都分明,有时带着一种略微责备的态度。"
  },
  {
    "id": "S28",
    "sourceId": "zh_755",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "按您说的，就把这个马家给开除了。他刚多大呀，这么点儿小孩就给推向社会当代理青年。那对社会也不负责呀。",
    "instruction": "就像是一个义愤填膺的人在陈情时,情绪高涨且语速快,音调多有起伏。"
  },
  {
    "id": "S29",
    "sourceId": "zh_833",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "啊，其实特别简单，我有一个详细的教程，我可以发给你啊。这样吧，我也帮你准备一份，到时候你看是用你自己的还是我这个枪手做的，都一样，反正啊心意在那儿了。",
    "instruction": "动漫少女开心地帮助朋友解决困难。"
  },
  {
    "id": "S30",
    "sourceId": "zh_838",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "这事儿跟你没关系。这么跟你说吧，我呢是张兰香，左请右请，请回他们家住的一位重要级人物。我说啥他就听啥，这么跟你说，你明白吗？",
    "instruction": "在班级辩论赛上,用一种自信张扬又略带不耐的口吻发言。"
  },
  {
    "id": "S31",
    "sourceId": "zh_888",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "谷志勇，你自己不求上进，别耽误你女儿。",
    "instruction": "教师严厉地指导学生重新审视问题。"
  },
  {
    "id": "S32",
    "sourceId": "zh_977",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "他们先跟人家编剧把合同签了，然后呢就拖着不付钱，哄人家编剧出东西。那编剧就想我这合同都签了，付钱还不是早晚的事儿，就把他们要的东西给他们了，他们就拿这东西偷偷的去找熟悉的平台看，平台要是认了他们就付款，平台要是不认，他们就耍赖。",
    "instruction": "说话像情感揭示专家,快节奏中带点愤慨和不满,但始终保持清晰流畅。"
  },
  {
    "id": "S33",
    "sourceId": "zh_505",
    "dataset": "InstructTTSEval",
    "condition": "RP",
    "text": "自从你爸爸去世以后，我把所有的心血都放在你们兄弟俩身上，艰难的经营着这家医院。我不允许任何人来破坏它。",
    "instruction": "权威女性领导者在发布家族重要声明。"
  }
];

const conditionOptions = [
  {key:'APS', label:'属性明细', note:'音高 · 音色 · 情绪'},
  {key:'DSD', label:'声音描述', note:'整体表达与质感'},
  {key:'RP', label:'角色场景', note:'身份 · 场景 · 目的'}
];
const cases = samples.filter(sample => sample.condition === 'APS');
const sampleMap = new Map(samples.map(sample => [`${sample.sourceId}:${sample.condition}`, sample]));
const $ = id => document.getElementById(id);
let currentCase = 0, condition = 'APS', activeAudio = null;
const formatTime = value => Number.isFinite(value) ? `${Math.floor(value/60)}:${String(Math.floor(value%60)).padStart(2,'0')}` : '0:00';
const bars = seed => Array.from({length:60},(_,i)=>18+((Math.sin((i+seed)*1.71)+1)*16+((i*seed)%13))).map(h=>`<i class="bar" style="--h:${Math.min(94,h)}%"></i>`).join('');
const selectedSample = () => sampleMap.get(`${cases[currentCase].sourceId}:${condition}`);

function renderConditions(){
  $('conditionTabs').innerHTML=conditionOptions.map(item=>`<button class="condition-tab ${condition===item.key?'active':''}" role="tab" aria-selected="${condition===item.key}" data-condition="${item.key}"><span class="condition-code">${item.key}</span><span class="condition-label">${item.label}</span><span class="condition-note">${item.note}</span></button>`).join('');
  $('conditionTabs').querySelectorAll('button').forEach(button=>button.onclick=()=>{
    if(condition===button.dataset.condition)return;
    stopAudio();condition=button.dataset.condition;render();
  });
}
function renderNav(){
  $('sampleList').innerHTML=cases.map((sample,index)=>`<button class="sample-button ${index===currentCase?'active':''}" data-index="${index}" title="${sample.text}"><span class="sample-no">${String(index+1).padStart(2,'0')}</span><span class="sample-title">${sample.text}</span><span class="sample-type">3×</span></button>`).join('');
  $('sampleList').querySelectorAll('button').forEach(button=>button.onclick=()=>{
    currentCase=Number(button.dataset.index);stopAudio();render();
    document.querySelector('.prompt-card').scrollIntoView({behavior:'smooth',block:'start'});
  });
}
function stopAudio(){if(activeAudio){activeAudio.pause();activeAudio=null;}}
function player(model,index,sample){
  const wave=bars((currentCase+2)*(index+5)+condition.charCodeAt(0));
  return `<article class="player" data-player="${index}"><div class="model-name">${model.name}</div><button class="play" aria-label="播放 ${model.name}"><svg viewBox="0 0 20 20"><path d="M5 3.5v13l11-6.5z"/></svg></button><div class="wave-wrap" role="slider" tabindex="0" aria-label="${model.name} 播放进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="wave">${wave}</div><div class="progress-fill" style="--progress:0%">${wave}</div></div><div class="time"><span>0:00</span> / <span>0:00</span></div><audio preload="metadata" src="audio/${model.key}/${sample.condition}/${sample.sourceId}.wav"></audio></article>`;
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
  renderNav();renderConditions();const s=selectedSample();$('sampleCount').textContent=`${String(currentCase+1).padStart(2,'0')} / ${String(cases.length).padStart(2,'0')}`;$('conditionBadge').textContent=condition;$('datasetLabel').textContent=s.dataset;$('promptIndex').textContent=`CASE ${String(currentCase+1).padStart(2,'0')}`;$('targetText').textContent=s.text;$('instruction').textContent=s.instruction;$('players').innerHTML=models.map((m,i)=>player(m,i,s)).join('');bindPlayers();
  document.querySelector('.prompt-card').dataset.condition=condition;
  document.querySelectorAll('.prompt-grid,.players').forEach(node=>{node.classList.remove('content-enter');void node.offsetWidth;node.classList.add('content-enter')});
}
render();
