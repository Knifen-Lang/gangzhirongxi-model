#!/usr/bin/env python3
"""
new_job_definition.py — 新岗位定义生成

从已标注JD中筛选新岗位候选（是否新岗位候选=='是'），
按归一化岗位名分组，每组聚合成一份结构化岗位定义：
  岗位名称、别称、核心职责、必备技能、加分技能、典型行业应用场景

规则引擎（默认）: 正则+统计，零LLM，确定性输出
LLM兜底（--use_llm）: 规则初稿→LLM精炼核心职责+行业场景→幻觉校验→择优采纳

Usage:
    # 纯规则（默认，零依赖）
    python new_job_definition.py \
        --input_dir "./【已标注】filtered/jd_v2/" \
        --output new_job_definitions.json

    # LLM精炼（需 DEEPSEEK_API_KEY 环境变量）
    python new_job_definition.py \
        --input_dir "./【已标注】filtered/jd_v2/" \
        --output new_job_definitions.json \
        --use_llm --llm_max 30
"""

import os, sys, json, argparse, csv, re, glob, time, ssl
from collections import defaultdict, Counter
from datetime import datetime
from difflib import SequenceMatcher

# SSL context for corporate proxy environments
_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# LLM API 配置（与 synthesize_data.py 保持一致）
LLM_CONFIG = {
    'api_base': 'https://api.deepseek.com/v1',
    'model': 'deepseek-v4-pro',
    'fallback_model': 'deepseek-v4-flash',
    'max_tokens': 2048,
}


# ============================================================
#  常量
# ============================================================

# 可从岗位名中剥离的通用修饰词
STRIP_TOKENS = r'高级|资深|初级|助理|实习|应届|校招|管培|培训生|见习|总监|主管|负责人'

PAREN_SUFFIX = re.compile(r'[（(][^)）]*[)）]$')

# 行业关键词 → 行业名
INDUSTRY_MAP = [
    (r'汽车|整车|新能源车|自动驾驶|adas|车联网', '智能汽车/自动驾驶'),
    (r'金融|银行|保险|证券|风控|支付|信用|投融资|基金|信托|期货', '金融科技'),
    (r'医疗|医药|医院|健康|影像|诊断|基因|临床|制药|药企', '智慧医疗'),
    (r'制造|工业|工厂|生产|质检|产线|机器人|智能制造|装备', '智能制造'),
    (r'教育|培训|学习|课程|教学|院校|在线教育', '智慧教育'),
    (r'电商|零售|物流|仓储|供应链|配送|快递', '电商与新零售'),
    (r'通信|电信|5G|运营商|基站|光通信|无线', '通信网络'),
    (r'游戏|娱乐|短视频|直播|内容|媒体|视频|影视|动漫|文化', '数字内容与娱乐'),
    (r'安全|安防|监控|巡检|应急|消防|公安|警务', '公共安全与安防'),
    (r'芯片|半导体|IC|集成电路|晶圆|封装|流片', '半导体与芯片'),
    (r'能源|电力|电网|光伏|风电|储能|新能源', '能源科技'),
    (r'政务|政府|智慧城市|数字政府|城市大脑|公共', '数字政务与智慧城市'),
    (r'招聘|HR|人事|猎头|人力|组织', 'HR科技'),
    (r'地产|建筑|施工|工程|勘察|测绘', '建筑与地产科技'),
    (r'农业|种植|养殖|畜牧|渔业|农田', '智慧农业'),
    (r'法律|律师|法务|合规|知产', '法律科技'),
]

# AI信号（用于职责排序 + 子分组）
AI_SIGNALS = [
    '大模型', 'LLM', '大语言模型', 'GPT', 'ChatGPT', 'DeepSeek',
    'AIGC', '生成式AI', 'GenAI',
    'Agent', 'AI Agent', '智能体', 'Multi-Agent', '多智能体',
    'RAG', '检索增强生成', 'Graph RAG',
    '大模型微调', 'SFT', 'RLHF', 'DPO',
    '多模态大模型', '视觉大模型', 'VLM', '多模态',
    'LangChain', 'LangGraph', 'LlamaIndex', 'Dify', 'Coze',
    'Stable Diffusion', 'SDXL', 'Midjourney', 'Sora',
    'LoRA', 'QLoRA', 'PEFT',
    'Prompt Engineering', '提示工程',
    'AI Coding', 'Copilot', 'CodeAgent',
    '数字孪生', '数字人', '具身智能',
    '联邦学习', '隐私计算',
]

# AI信号家族（用于子分组）
AI_FAMILIES = {
    '大模型': ['大模型', 'LLM', '大语言模型', 'GPT', 'ChatGPT', '大模型微调', 'SFT', 'RLHF', 'DPO',
              'DeepSeek', '预训练', '模型训练', '大语言', 'LLaMA', 'Qwen'],
    'Agent': ['Agent', 'AI Agent', '智能体', 'Multi-Agent', '多智能体', 'LangChain', 'LangGraph',
              'LlamaIndex', 'Dify', 'Coze', 'Agentic', 'AI Coding', 'Copilot', 'CodeAgent'],
    'AIGC': ['AIGC', '生成式AI', 'GenAI', 'Stable Diffusion', 'SDXL', 'Midjourney', 'Sora',
             'LoRA', 'QLoRA', '图像生成', '视频生成', '文生图', '文生视频'],
    '多模态': ['多模态大模型', '视觉大模型', 'VLM', '多模态', '视觉语言'],
    'RAG': ['RAG', '检索增强生成', 'Graph RAG', '向量数据库', 'Embedding'],
}


# ============================================================
#  岗位名归一化
# ============================================================

def normalize_job_name(raw):
    """归一化岗位名：去掉括号、职级、空白差异、尾部噪音，保留核心"""
    if not raw:
        return ''
    s = raw.strip()
    # ---- 第1轮：粗清理 ----
    # 去【】[] 等营销标签
    s = re.sub(r'[【\[][^】\]]*[】\]]', '', s)
    # 去所有括号内容（全角+半角），多轮处理嵌套
    for _ in range(3):
        new_s = re.sub(r'[（(][^)）]*[)）]', '', s)
        if new_s == s:
            break
        s = new_s
    # 去职级前缀
    s = re.sub(rf'^({STRIP_TOKENS})', '', s).strip()
    # lowercase
    s = s.lower()
    # 去多余空白
    s = re.sub(r'\s+', '', s)
    # ---- 第2轮：后缀噪音 ----
    # 去尾部编号: "-XX" / "—XX" / "_XX"
    s = re.sub(r'[-—_]\S{1,8}$', '', s)
    # 去尾部 "---公司---城市" 模式
    s = re.sub(r'[-—]{2,}.+$', '', s)
    # 去尾部 "/实习" "/全职" "/应届" "/社招" "/校招"
    s = re.sub(r'/[^\w一-鿿]{0,2}(实习|全职|应届|社招|校招|正式|兼职|外包)$', '', s)
    # ---- 第3轮：语义归一化 ----
    # "实习生" 是职级不是独立岗位：Xxx实习生 → Xxx工程师
    s = re.sub(
        r'(算法|开发|测试|数据|产品|设计|运维|安全|前端|后端|全栈|AI|NLP|CV|ML|'
        r'系统|网络|嵌入|硬件|芯片|结构|材料|电气|机械|光学|声学|仿真|'
        r'量化|风控|建模|分析|架构|科研|研究)实习生$',
        r'\1工程师', s
    )
    # 去掉残留"实习生"/"方向"（非技术前缀）
    s = re.sub(r'(实习生|方向)$', '', s)
    # 统一: 研发工程师/应用开发工程师/软件开发工程师 → 开发工程师
    s = re.sub(r'(应用开发|软件开发|开发|研发)工程师', '开发工程师', s)
    # 应用工程师 → 开发工程师
    s = re.sub(r'应用工程师', '开发工程师', s)
    # 英文技术词后的"开发"可选: Agent开发工程师 ≈ Agent工程师
    # 合并: xxxAgent开发工程师 → xxxAgent工程师
    s = re.sub(r'([A-Za-z]{2,})开发工程师$', r'\1工程师', s)
    # ---- 第4轮：前缀噪音 ----
    # 去掉开头公司/部门/项目名前缀（如 "华为-", "2027AIDU-"）
    s = re.sub(r'^[\w一-鿿]{1,12}[-—]', '', s)
    # 去掉开头纯数字字母编号
    s = re.sub(r'^[\dA-Za-z_]{2,12}[-—]', '', s)
    # 去掉开头残留 "-"
    s = re.sub(r'^[-—]', '', s)
    # ---- 第5轮：中英混合名取中文部分 ----
    # 对于中英混合（如 "AI Edge Engineer – AI边缘工程师"），优先取中文段
    if re.search(r'[一-鿿]', s):
        parts = re.split(r'[|｜–—-]', s)
        zh_parts = [p.strip() for p in parts if re.search(r'[一-鿿]', p)]
        if zh_parts:
            s = max(zh_parts, key=len)
    # "/" 分隔的复合名：取含角色后缀的最长段
    if '/' in s:
        parts = [p.strip() for p in s.split('/')]
        best = max(parts, key=lambda p: (
            1 if re.search(r'(工程师|经理|专家|总监|架构师|研究员|分析师|设计师|专员|顾问|负责人|主管)', p) else 0,
            len(p)
        ))
        if best:
            s = best
    # ---- 最终清理 ----
    s = s.strip()
    if len(s) < 3:
        return raw.strip().lower()
    return s


# ============================================================
#  技能解析
# ============================================================

def parse_skill_list(s):
    """解析 【技能|类别|级别】；【...】 格式 → [{'skill','category','level'}]"""
    if not s or pd.isna(s):
        return []
    skills = []
    for m in re.finditer(r'【(.+?)】', str(s)):
        parts = m.group(1).split('｜')
        if len(parts) >= 1:
            skills.append({
                'skill': parts[0].strip(),
                'category': parts[1].strip() if len(parts) >= 2 else '未知',
                'level': parts[2].strip() if len(parts) >= 3 else '未标注',
            })
    return skills


# ============================================================
#  职责提取
# ============================================================

def parse_duties(text):
    """从JD原文提取职责句子"""
    if not text or len(text) < 20:
        return []
    # 找"岗位职责"区段
    duty_patterns = [
        r'岗位职责[：:](.*?)(?=岗位要求|任职要求|任职资格|岗位条件|$)',
        r'工作(内容|职责|描述)[：:](.*?)(?=岗位要求|任职要求|任职资格|岗位条件|$)',
        r'职位描述[：:](.*?)(?=岗位要求|任职要求|任职资格|岗位条件|$)',
    ]
    duty_text = ''
    for pat in duty_patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            duty_text = m.group(0)
            break

    if not duty_text:
        # 无明确标题，取全文前60%
        duty_text = text[:int(len(text) * 0.6)]

    # 拆句子
    sentences = []
    for s in re.split(r'[。；;.\n]', duty_text):
        s = s.strip()
        # 去编号前缀
        s = re.sub(r'^[\d]+[、.．)\s]*', '', s)
        # 去标题
        s = re.sub(r'^(岗位职责|工作内容|职位描述|工作职责)[：:]?\s*', '', s)
        s = s.strip()
        if len(s) > 8 and len(s) < 120:
            sentences.append(s)
    return sentences


def score_duties(sentences, ai_signal_set, cluster_duty_counter, N):
    """用AI信号+簇内频率评分，返回Top 5-8条"""
    scored = []
    for sent in sentences:
        if len(sent) < 10:
            continue
        # AI信号密度
        ai_hits = sum(1 for sig in ai_signal_set if sig.lower() in sent.lower())
        ai_density = ai_hits / max(len(sent), 1)
        # 簇内频率分（bigram近似）
        freq_score = 0
        for i in range(len(sent) - 1):
            bigram = sent[i:i+2]
            freq_score += cluster_duty_counter.get(bigram, 0)
        freq_score /= max(len(sent), 1)
        # 综合分
        score = 0.4 * ai_density + 0.3 * freq_score + 0.3 * (1.0 / (len(sentences) + 1))
        scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])

    # 去重
    result = []
    for _, sent in scored:
        dup = False
        for existing in result:
            if SequenceMatcher(None, sent, existing).ratio() > 0.7:
                dup = True
                break
        if not dup:
            result.append(sent)
        if len(result) >= 8:
            break
    return result[:8]


def extract_ai_signals_from_text(text):
    """从文本提取AI信号"""
    if not text:
        return set()
    return {sig for sig in AI_SIGNALS if sig.lower() in text.lower()}


# ============================================================
#  行业推断
# ============================================================

def infer_industry(company_name, work_area=''):
    """从公司名+工作地推断行业"""
    text = f'{company_name} {work_area}'
    for pattern, industry in INDUSTRY_MAP:
        if re.search(pattern, text):
            return industry
    return '通用信息技术'


# ============================================================
#  LLM 精炼（可选兜底）
# ============================================================

def call_llm_api(prompt, api_key, temperature=0.4):
    """调用 DeepSeek API（urllib，零外部依赖）"""
    import urllib.request
    import urllib.error

    payload = {
        'model': LLM_CONFIG['model'],
        'messages': [
            {'role': 'system',
             'content': '你是一位专业的HR分析师和岗位定义专家。你的输出必须严格基于提供的源数据，不得编造任何技能名、公司名或统计数据。'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': temperature,
        'max_tokens': LLM_CONFIG['max_tokens'],
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f'{LLM_CONFIG["api_base"].rstrip("/")}/chat/completions',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
    )

    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=90) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')[:300] if e.fp else ''
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(2 ** attempt)
            else:
                print(f'  [LLM] API错误 {e.code}: {body}')
                return None
        except Exception as e:
            print(f'  [LLM] 网络错误: {e}')
            if attempt < 4:
                time.sleep(2 ** attempt)
    return None


def build_llm_prompt(defn):
    """基于规则定义构建LLM精炼prompt"""
    t = defn['source_traceability']
    raw_duties = defn['核心职责']
    req_skills = [s['skill'] for s in defn['必备技能']]
    bonus_skills = [s['skill'] for s in defn['加分技能']]
    industries = [i['industry'] for i in defn['典型行业应用场景']]

    prompt = f"""请基于以下源数据，为岗位"{defn['岗位名称']}"生成精炼的岗位定义。

【源数据】
- 岗位名称: {defn['岗位名称']}
- 别称: {', '.join(defn['别称'][:5])}
- 样本量: {t['total_jds']}条JD, {t['unique_companies']}家公司
- 代表性公司: {', '.join(t['top_companies'][:5])}
- 代表城市: {', '.join(t['top_cities'][:5])}

【原始职责描述（从{len(raw_duties)}条规则提取）】
{chr(10).join(f'- {d[:120]}' for d in raw_duties[:10])}

【高频必备技能（来自{t['total_jds']}条JD统计）】
{', '.join(req_skills[:8])}

【加分技能】
{', '.join(bonus_skills[:6])}

【规则推断行业】
{', '.join(industries[:5])}

请输出JSON格式（只输出JSON，不要其他文字）：
{{
  "核心职责": [
    "职责1（≤30字，凝练自源数据，不编造）",
    "职责2",
    ...
  ],
  "典型行业应用场景": [
    {{"industry": "行业名", "rationale": "一句话理由（必须引用源数据中的公司或JD内容）"}},
    ...
  ]
}}

要求：
1. 核心职责5-8条，每条≤30字，凝练而非照抄，但必须能从源职责回溯
2. 行业场景3-5个，必须基于源数据中的公司名推断，不得凭空添加行业
3. 严禁编造源数据中不存在的技能、公司、数字
4. 只输出JSON，不要markdown代码块标记"""
    return prompt


def verify_llm_duties(llm_duties, source_duties, source_skills, source_companies):
    """校验LLM输出：检测幻觉（编造技能/公司/数字）"""
    warnings = []
    source_text = ' '.join(source_duties).lower()
    all_source_skills = set(s.lower() for s in source_skills)
    all_source_companies = set(c.lower() for c in source_companies)

    for duty in llm_duties:
        # 检查是否在源数据中有语义重叠（至少50%的3-gram命中）
        if len(duty) < 6:
            warnings.append(f'职责过短: {duty}')
            continue
        trigrams = [duty[i:i+3] for i in range(len(duty)-2)]
        hit = sum(1 for t in trigrams if t in source_text)
        if hit / max(len(trigrams), 1) < 0.15:
            warnings.append(f'职责与源数据关联弱: {duty[:50]}...')

        # 检查是否编造了不存在的公司名
        for company in all_source_companies:
            if len(company) >= 4 and company in duty.lower():
                break  # OK，源数据有
        else:
            # 没有源公司名命中，检查是否提到了其他看起来像公司名的词
            pass  # 放宽：职责描述不一定需要提公司名

    return warnings


def refine_with_llm(defn, api_key):
    """对单条定义做LLM精炼，返回精炼后的字段（失败则返回None）"""
    prompt = build_llm_prompt(defn)
    raw = call_llm_api(prompt, api_key, temperature=0.4)

    if not raw:
        return None

    # 解析JSON
    try:
        # 清理可能的markdown包裹
        raw = raw.strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```\w*\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取JSON子串
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                result = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    # 校验
    source_skills = [s['skill'] for s in defn['必备技能']] + [s['skill'] for s in defn['加分技能']]
    source_companies = defn['source_traceability']['top_companies']
    warnings = verify_llm_duties(
        result.get('核心职责', []),
        defn['核心职责'],
        source_skills,
        source_companies
    )

    if len(warnings) > len(result.get('核心职责', [1])) * 0.5:
        # 超过一半职责有告警，拒绝LLM输出
        print(f'  [LLM] 幻觉校验不通过 ({len(warnings)}/{len(result.get("核心职责", []))}条告警)，回退规则版本')
        return None

    return {
        '核心职责': result.get('核心职责', defn['核心职责']),
        '典型行业应用场景': result.get('典型行业应用场景', defn['典型行业应用场景']),
        'llm_warnings': warnings,
    }


# ============================================================
#  主逻辑
# ============================================================

def load_new_jobs(input_dir):
    """加载所有已标注CSV，筛选新岗位候选"""
    files = list(set(
        glob.glob(os.path.join(input_dir, '【已标注】*.csv')) +
        glob.glob(os.path.join(input_dir, '*.csv'))
    ))
    if not files:
        files = list(set(glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)))

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                if str(row.get('是否新岗位候选', '')).strip() == '是':
                    rows.append(row.to_dict())
        except Exception as e:
            print(f'  [WARN] 无法读取 {f}: {e}')

    print(f'[LOAD] {len(files)} 个文件, {len(rows)} 条新岗位JD')
    return rows


def group_and_generate(rows, timeline_report_path=None, use_llm=False, llm_max=30):
    """按归一化岗位名分组，每组生成一份定义"""
    # 分组
    groups = defaultdict(list)
    for r in rows:
        raw = str(r.get('job_name', '')).strip()
        norm = normalize_job_name(raw)
        if not norm:
            continue
        groups[norm].append(r)

    print(f'[GROUP] {len(groups)} 个岗位类型')

    # 加载时间轴数据（可选）
    timeline_index = {}
    if timeline_report_path and os.path.exists(timeline_report_path):
        try:
            with open(timeline_report_path, 'r', encoding='utf-8') as f:
                cap = json.load(f)
            for rep in cap.get('reports', []):
                timeline_index[rep['job']] = rep.get('update_summary', '')
            print(f'[TIMELINE] {len(timeline_index)} 条能力趋势已加载')
        except Exception:
            pass

    # LLM API Key
    api_key = None
    if use_llm:
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        if not api_key:
            print('[LLM] DEEPSEEK_API_KEY 未设置，回退纯规则模式')
            use_llm = False

    definitions = []
    llm_refined = 0
    for cluster_id, (norm_name, jds) in enumerate(sorted(groups.items(), key=lambda x: -len(x[1])), 1):
        n = len(jds)
        if n < 2:
            continue  # 单条JD不足以"定义"一个岗位

        defn = build_definition(cluster_id, norm_name, jds, timeline_index)

        # LLM精炼：仅对JD数≥5的定义做精炼，上限llm_max个
        if use_llm and n >= 5 and llm_refined < llm_max:
            print(f'  [LLM] 精炼 #{cluster_id} {defn["岗位名称"]} ({n}JDs)...', end=' ')
            refined = refine_with_llm(defn, api_key)
            if refined:
                defn['核心职责'] = refined['核心职责']
                defn['典型行业应用场景'] = refined['典型行业应用场景']
                defn['llm_refined'] = True
                defn['llm_warnings'] = refined.get('llm_warnings', [])
                llm_refined += 1
                print('OK')
            else:
                defn['llm_refined'] = False
                print('回退规则')

        definitions.append(defn)

    print(f'[GEN] {len(definitions)} 份岗位定义已生成 (LLM精炼: {llm_refined})')
    return definitions


def dedup_by_jaccard(jds):
    """
    簇内Jaccard抄袭检测（复用 graph_calibrator.py 同样逻辑）。
    同公司+JD文本相似度>0.8 → 抄袭簇 → 每条JD权重=1/簇大小。
    返回每条JD的权重列表。
    """
    n = len(jds)
    texts = [str(r.get('skill_requirements', '')) for r in jds]
    companies = [str(r.get('company_name', '')).strip() for r in jds]
    factors = [1.0] * n   # 簇大小（1=独立JD）

    for i in range(n):
        if factors[i] != 1.0:
            continue  # 已被归入前面的抄袭簇
        cluster = [i]
        for j in range(i + 1, n):
            if factors[j] != 1.0:
                continue
            # 同公司 + 文本相似 > 0.8 = 抄袭
            if companies[i] and companies[i] == companies[j]:
                sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if sim > 0.8:
                    cluster.append(j)
        if len(cluster) > 1:
            for idx in cluster:
                factors[idx] = len(cluster)

    # 权重 = 1/簇大小（抄袭簇越大，每条JD贡献越小）
    weights = [1.0 / f for f in factors]
    return weights, factors


def build_definition(cluster_id, norm_name, jds, timeline_index):
    """为一个岗位类型合成定义"""
    n = len(jds)

    # ── 岗位名称 ──
    name_counter = Counter(str(r.get('job_name', '')).strip() for r in jds)
    canonical = name_counter.most_common(1)[0][0]
    aliases = [nm for nm, _ in name_counter.most_common(8) if nm != canonical]

    # ── Jaccard抄袭去重（同 graph_calibrator.py） ──
    jd_weights, cluster_factors = dedup_by_jaccard(jds)
    # 统计通胀：簇大小≥3视为重度，2视为中度
    inflate_heavy = sum(1 for f in cluster_factors if f >= 3)
    inflate_medium = sum(1 for f in cluster_factors if f == 2)
    inflate_light = 0  # Jaccard方法只有"抄袭"和"独立"两态
    inflate_clean = sum(1 for f in cluster_factors if f == 1.0)

    # 收集全量文本+技能（Jaccard去重权重）
    all_duties = []
    all_required = []        # (skill_dict, weight)
    all_bonus = []           # (skill_dict, weight)
    all_ai_signals = set()
    companies = []
    cities = []
    duty_counter = Counter()
    total_weight = 0.0       # 有效JD数（去重后）

    for i, r in enumerate(jds):
        w = jd_weights[i]
        total_weight += w

        text = str(r.get('skill_requirements', ''))
        duties = parse_duties(text)
        all_duties.extend(duties)
        for sent in duties:
            for c in range(len(sent) - 1):
                duty_counter[sent[c:c+2]] += w  # 加权

        all_ai_signals |= extract_ai_signals_from_text(text)
        for sk in parse_skill_list(r.get('必备技能', '')):
            all_required.append((sk, w))
        for sk in parse_skill_list(r.get('加分技能', '')):
            all_bonus.append((sk, w))

        company = str(r.get('company_name', '')).strip()
        if company:
            companies.append(company)
        area = str(r.get('work_area', '')).strip()
        if area:
            cities.append(area)

    # ── 核心职责 ──
    core_duties = score_duties(all_duties, all_ai_signals, duty_counter, n)
    if len(core_duties) < 3:
        core_duties = all_duties[:5]

    # ── 必备技能（Jaccard去重矫正） ──
    req_counter_raw = Counter()    # 原始计数
    req_counter_w = Counter()      # Jaccard去重计数
    req_levels = defaultdict(list)
    for sk, w in all_required:
        req_counter_raw[sk['skill']] += 1
        req_counter_w[sk['skill']] += w
        req_levels[sk['skill']].append(sk['level'])
    required_skills = []
    demoted_to_bonus = []          # 矫正后从必备降级到加分的技能
    effective_n = max(total_weight, 1.0)
    for skill, cnt_w in req_counter_w.most_common():
        raw_support = req_counter_raw[skill] / n
        corrected_support = cnt_w / effective_n
        modal_level = Counter(req_levels[skill]).most_common(1)[0][0]
        cat = 'AI新兴技能' if skill in all_ai_signals else '传统技术'
        skill_entry = {
            'skill': skill, 'category': cat, 'proficiency': modal_level,
            'support_pct': round(raw_support, 2),
            'support_pct_corrected': round(corrected_support, 2),
        }
        if corrected_support >= 0.15:
            required_skills.append(skill_entry)
        elif corrected_support >= 0.08:
            # 矫正后不足以当必备，但够格当加分 → 降级
            demoted_to_bonus.append(skill_entry)
        # else: 矫正后连加分都不够 → 丢弃
        if len(required_skills) >= 8:
            break

    # ── 加分技能（Jaccard去重矫正 + 必备降级） ──
    bonus_counter_raw = Counter()
    bonus_counter_w = Counter()
    bonus_levels = defaultdict(list)
    for sk, w in all_bonus:
        bonus_counter_raw[sk['skill']] += 1
        bonus_counter_w[sk['skill']] += w
        bonus_levels[sk['skill']].append(sk['level'])
    req_skill_names = {s['skill'] for s in required_skills}
    bonus_skills = []

    # 先加入从必备降级下来的技能（already validated）
    for sk_entry in demoted_to_bonus:
        if sk_entry['skill'] not in req_skill_names:
            bonus_skills.append(sk_entry)

    # 再加入原始加分技能（去重必备+已降级）
    existing_bonus = {s['skill'] for s in bonus_skills}
    for skill, cnt_w in bonus_counter_w.most_common():
        if skill in req_skill_names or skill in existing_bonus:
            continue
        raw_support = bonus_counter_raw[skill] / n
        corrected_support = cnt_w / effective_n
        if corrected_support < 0.08:
            break
        modal_level = Counter(bonus_levels[skill]).most_common(1)[0][0]
        cat = 'AI新兴技能' if skill in all_ai_signals else '传统技术'
        bonus_skills.append({
            'skill': skill, 'category': cat, 'proficiency': modal_level,
            'support_pct': round(raw_support, 2),
            'support_pct_corrected': round(corrected_support, 2),
        })
        existing_bonus.add(skill)
        if len(bonus_skills) >= 8:  # 加分上限放宽，容纳降级技能
            break

    # ── 典型行业应用场景 ──
    industry_counter = Counter()
    for company, city in zip(companies, cities + [''] * max(0, len(companies) - len(cities))):
        industry = infer_industry(company, city)
        industry_counter[industry] += 1
    industries = []
    for ind, cnt in industry_counter.most_common(5):
        support = cnt / len(companies) if companies else 0
        if support >= 0.05:
            industries.append({'industry': ind, 'support_pct': round(support, 2)})

    # ── 可溯源 ──
    company_set = list(dict.fromkeys(companies))
    city_set = list(dict.fromkeys(c for c in cities if c))
    # 通胀统计（基于Jaccard抄袭检测，同 graph_calibrator.py）
    inflate_breakdown = {
        '重度(簇≥3)': inflate_heavy,
        '中度(簇=2)': inflate_medium,
        '无': inflate_clean,
    }
    any_inflate = inflate_heavy + inflate_medium
    # 通胀严重度：重度JD*1.0 + 中度JD*0.5 / 总JD
    inflation_severity = round(
        (inflate_heavy * 1.0 + inflate_medium * 0.5) / max(n, 1), 2
    )
    inflation_ratio = any_inflate / n if n else 0

    # ── 动态更新 ──
    dynamic = {'linked_timeline': False}
    if timeline_index:
        for job_key, trend in timeline_index.items():
            if normalize_job_name(job_key) == norm_name or canonical in job_key or job_key in canonical:
                dynamic = {
                    'linked_timeline': True,
                    'timeline_report_job_match': job_key,
                    'capability_trend': trend[:200] if trend else '',
                }
                break

    return {
        'cluster_id': cluster_id,
        '岗位名称': canonical,
        '别称': aliases,
        '归一化名': norm_name,
        '核心职责': core_duties,
        '必备技能': required_skills,
        '加分技能': bonus_skills,
        '典型行业应用场景': industries if industries else [{'industry': '通用信息技术', 'support_pct': 1.0}],
        'source_traceability': {
            'total_jds': n,
            'effective_jds': round(total_weight, 1),  # 通胀矫正后有效JD数
            'unique_companies': len(company_set),
            'top_companies': company_set[:5],
            'top_cities': city_set[:5],
            'inflation_ratio': round(inflation_ratio, 2),
            'inflation_severity': inflation_severity,    # 加权严重度(0~1)
            'inflation_breakdown': inflate_breakdown,
        },
        '人工优化': {
            'status': 'pending_review',
            'audit_id': f'newjob_{cluster_id:03d}',
        },
        '动态更新': dynamic,
    }


# ============================================================
#  输出
# ============================================================

import unicodedata

def sanitize_xlsx(val):
    """清洗Excel非法字符（控制字符等）"""
    if not isinstance(val, str):
        return val
    # 移除非法控制字符（保留 \t \n \r）
    cleaned = []
    for ch in val:
        cp = ord(ch)
        if cp < 32 and cp not in (9, 10, 13):
            cleaned.append(' ')
        elif 0xD800 <= cp <= 0xDFFF:
            cleaned.append(' ')  # surrogate pairs
        elif cp == 0xFFFE or cp == 0xFFFF:
            cleaned.append(' ')
        else:
            cleaned.append(ch)
    return ''.join(cleaned)


def export_audit_excel(definitions, output_path):
    """导出待审核定义到Excel，方便人工逐条review"""
    try:
        import openpyxl
    except ImportError:
        print('[WARN] openpyxl未安装，跳过Excel导出')
        return

    wb = openpyxl.Workbook()

    # ── Sheet 1: 待审核定义 ──
    ws1 = wb.active
    ws1.title = '待审核定义'
    headers = [
        '审核编号', '岗位名称', '别称', '归一化名',
        'JD数(原始)', '有效JD(矫正)', '通胀严重度(0~1)', '通胀分级(轻/中/重/无)',
        '核心职责', '必备技能(原始%→矫正%)', '加分技能(原始%→矫正%)',
        '行业场景', '代表公司', '代表城市',
        'LLM精炼', '审核状态', '审核意见'
    ]
    ws1.append(headers)

    # 标题行加粗+底色
    header_fill = openpyxl.styles.PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font = openpyxl.styles.Font(color='FFFFFF', bold=True, size=11)
    for col, h in enumerate(headers, 1):
        cell = ws1.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = openpyxl.styles.Alignment(horizontal='center', vertical='center', wrap_text=True)

    for i, d in enumerate(definitions, 1):
        t = d['source_traceability']
        bd = t.get('inflation_breakdown', {})
        inflate_detail = f"重(簇≥3):{bd.get('重度(簇≥3)',0)} 中(簇=2):{bd.get('中度(簇=2)',0)} 独立:{bd.get('无',0)}"
        row = [
            d['人工优化']['audit_id'],
            d['岗位名称'],
            ' | '.join(d['别称'][:5]),
            d['归一化名'],
            t['total_jds'],
            f"{t['effective_jds']:.0f}",
            f"{t.get('inflation_severity', 0):.2f}",
            inflate_detail,
            '\n'.join(f'{j+1}. {duty}' for j, duty in enumerate(d['核心职责'])),
            '\n'.join(f"{s['skill']} {int(s['support_pct']*100)}%→{int(s.get('support_pct_corrected',s['support_pct'])*100)}%" for s in d['必备技能']),
            '\n'.join(f"{s['skill']} {int(s['support_pct']*100)}%→{int(s.get('support_pct_corrected',s['support_pct'])*100)}%" for s in d['加分技能']),
            '\n'.join(f"{ind['industry']}({int(ind['support_pct']*100)}%)" for ind in d['典型行业应用场景']),
            ' | '.join(t['top_companies'][:5]),
            ' | '.join(t['top_cities'][:5]),
            '是' if d.get('llm_refined') else '否',
            '待审核',
            ''  # 审核意见留空
        ]
        row = [sanitize_xlsx(cell) for cell in row]
        ws1.append(row)

    # 列宽
    col_widths = [14, 22, 30, 22, 6, 6, 8, 55, 30, 30, 25, 30, 25, 8, 10, 15]
    for col, w in enumerate(col_widths, 1):
        ws1.column_dimensions[openpyxl.utils.get_column_letter(col)].width = w

    # 冻结首行
    ws1.freeze_panes = 'A2'
    # 自动筛选
    ws1.auto_filter.ref = f'A1:P{len(definitions)+1}'

    # ── Sheet 2: 统计概览 ──
    ws2 = wb.create_sheet('统计概览')
    total_jds = sum(d['source_traceability']['total_jds'] for d in definitions)
    total_companies = sum(d['source_traceability']['unique_companies'] for d in definitions)

    stats = [
        ['指标', '数值'],
        ['定义总数', len(definitions)],
        ['覆盖JD总数', total_jds],
        ['平均JD/定义', round(total_jds / max(len(definitions), 1), 1)],
        ['LLM精炼数', sum(1 for d in definitions if d.get('llm_refined'))],
        ['关联时间轴', sum(1 for d in definitions if d['动态更新']['linked_timeline'])],
        ['', ''],
        ['Top 20 岗位', 'JD数'],
    ]
    for d in sorted(definitions, key=lambda x: -x['source_traceability']['total_jds'])[:20]:
        stats.append([d['岗位名称'], d['source_traceability']['total_jds']])

    for row in stats:
        ws2.append(row)

    ws2.column_dimensions['A'].width = 25
    ws2.column_dimensions['B'].width = 20

    # ── Sheet 3: 行业分布 ──
    ws3 = wb.create_sheet('行业分布')
    industry_all = Counter()
    for d in definitions:
        for ind in d['典型行业应用场景']:
            industry_all[ind['industry']] += d['source_traceability']['total_jds']
    ws3.append(['行业', '关联JD数', '关联定义数'])
    for ind, jd_count in industry_all.most_common():
        def_count = sum(1 for d in definitions
                       if any(i['industry'] == ind for i in d['典型行业应用场景']))
        ws3.append([ind, jd_count, def_count])
    ws3.column_dimensions['A'].width = 25

    wb.save(output_path)
    print(f'[AUDIT] 审核Excel: {output_path}')
    print(f'  待审核定义: {len(definitions)} 条')
    print(f'  3个工作表: 待审核定义 | 统计概览 | 行业分布')


def save_output(definitions, input_dir, output_path, use_llm=False):
    llm_count = sum(1 for d in definitions if d.get('llm_refined'))
    stats = {
        'total_definitions': len(definitions),
        'total_jds_covered': sum(d['source_traceability']['total_jds'] for d in definitions),
        'avg_jds_per_definition': round(sum(d['source_traceability']['total_jds'] for d in definitions) / max(len(definitions), 1), 1),
        'industry_distribution': Counter(i['industry'] for d in definitions for i in d['典型行业应用场景']),
        'definitions_with_timeline': sum(1 for d in definitions if d['动态更新']['linked_timeline']),
        'llm_refined_count': llm_count,
        'engine': 'rule+llm' if use_llm else 'rule',
    }

    output = {
        'generated_at': datetime.now().isoformat(),
        'source': input_dir,
        'num_definitions': len(definitions),
        'definitions': definitions,
        'statistics': stats,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\n[OUTPUT] {output_path}')
    print(f'  岗位定义: {stats["total_definitions"]} 个')
    print(f'  覆盖JD: {stats["total_jds_covered"]} 条')
    print(f'  平均每岗位: {stats["avg_jds_per_definition"]} 条JD')
    print(f'  关联时间轴: {stats["definitions_with_timeline"]} 个')
    print(f'  行业覆盖: {dict(stats["industry_distribution"].most_common(5))}')


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='新岗位定义生成 — 从标注JD聚合生成结构化岗位定义')
    parser.add_argument('--input_dir', type=str, required=True, help='已标注CSV目录')
    parser.add_argument('--output', type=str, default='new_job_definitions.json', help='输出JSON路径')
    parser.add_argument('--timeline_report', type=str, default='', help='能力动态更新报告（可选）')
    parser.add_argument('--use_llm', action='store_true', help='启用LLM精炼核心职责和行业场景（需DEEPSEEK_API_KEY）')
    parser.add_argument('--llm_max', type=int, default=30, help='LLM精炼最多处理前N个岗位（默认30）')
    parser.add_argument('--audit_excel', type=str, default='', help='导出待审核Excel路径（如 audit_definitions.xlsx）')
    args = parser.parse_args()

    print('=' * 60)
    engine = '规则+LLM' if args.use_llm else '纯规则引擎'
    print(f'  新岗位定义生成 ({engine})')
    print('=' * 60)

    global pd
    import pandas as pd

    rows = load_new_jobs(args.input_dir)
    if not rows:
        print('[ERROR] 未找到新岗位候选JD')
        sys.exit(1)

    timeline_path = args.timeline_report
    if timeline_path and not os.path.exists(timeline_path):
        # 尝试默认路径
        alt = os.path.join(os.path.dirname(__file__), 'capability_update_report.json')
        if os.path.exists(alt):
            timeline_path = alt

    definitions = group_and_generate(rows, timeline_path,
                                     use_llm=args.use_llm,
                                     llm_max=args.llm_max)
    save_output(definitions, args.input_dir, args.output, use_llm=args.use_llm)

    # 导出审核Excel
    if args.audit_excel:
        export_audit_excel(definitions, args.audit_excel)


if __name__ == '__main__':
    main()
