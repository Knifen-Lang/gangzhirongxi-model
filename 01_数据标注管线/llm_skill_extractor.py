#!/usr/bin/env python3
"""
llm_skill_extractor.py — 三级抽取管线：字典 → LLM → RAG

流程：
  JD原文
    │
    ▼
  第1级：字典抽取（annotate_v3）
    ├─ 抽到技能 → 直接输出（7600条，零成本）
    └─ 未抽到任何技能 → 进入第2级
          │
          ▼
  第2级：LLM语义理解
    │  把口语化JD（"协助团队使用AI工具完成文案生成"）
    │  转为结构化技能（Prompt Engineering, AIGC应用）
    │  Prompt强制要求：每项技能必须引述JD原句
    │
    ▼
  第3级：RAG验证
    │  每项LLM输出 → 检索技能库 + JD原句匹配
    ├─ 两个都有 → verified → 入库
    ├─ 仅技能库有 → needs_review
    └─ 都没有 → hallucinated → 丢弃

Usage:
    python llm_skill_extractor.py --input ./【已标注】 4/【已标注】 --output debug_llm_skills.csv
"""

import os, sys, csv, re, json, glob
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '【已标注v3】'))
from annotate_v3 import (
    extract_skills_from_text, classify_skill, get_mastery_level,
    is_bonus_signal, is_soft_skill, AI_SKILLS, TRADITIONAL_SKILLS,
)

# ── RAG知识库（与 hallucination_guard 同源） ──
VERIFIED_SKILLS = AI_SKILLS | TRADITIONAL_SKILLS


class LLMSkillExtractor:
    """三级抽取管线"""

    def __init__(self, llm_mode='local'):
        """
        llm_mode:
          'local'  — 本地规则模拟LLM（测试用）
          'coze'   — Coze API
          'custom' — 自定义回调
        """
        self.llm_mode = llm_mode
        self.stats = {'dict_hit': 0, 'llm_called': 0, 'rag_verified': 0,
                      'rag_hallucinated': 0, 'rag_needs_review': 0}

    # ═══════════════════════════════════════
    #  第1级：字典抽取
    # ═══════════════════════════════════════

    def dict_extract(self, jd_text):
        """字典抽取，返回 (skills_dict, is_empty)"""
        found = extract_skills_from_text(jd_text)
        if not found:
            return {}, True  # 字典空，需要LLM
        self.stats['dict_hit'] += 1
        return found, False

    # ═══════════════════════════════════════
    #  第2级：LLM语义理解
    # ═══════════════════════════════════════

    def llm_extract(self, jd_text, job_name=''):
        """
        LLM从口语化JD中提取技能。
        本地模式用规则模拟；Coze模式调API。
        """
        if self.llm_mode == 'local':
            return self._local_llm_extract(jd_text)
        elif self.llm_mode == 'coze':
            return self._coze_llm_extract(jd_text)
        else:
            raise ValueError(f'Unknown llm_mode: {self.llm_mode}')

    def _build_llm_prompt(self, jd_text, job_name=''):
        """构建LLM prompt — 强制约束输出格式"""
        return f"""你是一个JD技能提取器。请从以下JD文本中提取所有硬性技术技能。

规则：
1. 只提取JD中明确提到的技术名词（编程语言、框架、工具、技术领域等）
2. 不提取软技能（沟通、团队合作、责任心等）
3. 不编造JD中未出现的技能
4. 每项技能必须附带JD原句作为证据

输出JSON格式：
{{
  "skills": [
    {{"skill": "技能名", "category": "AI新兴技能 或 传统技术", "evidence": "JD原句摘录"}}
  ]
}}

JD文本：
{jd_text[:2000]}

请输出JSON："""

    def _local_llm_extract(self, jd_text):
        """
        本地规则模拟LLM — 处理字典失效的口语化表述。
        字典是精确匹配，这里补充模糊语义匹配。
        """
        skills = []

        # ── AI绘画拆解 + 模型开发推断 ──
        fuzzy_patterns = [
            # AI绘画 → 拆为具体工具
            (r'AI(绘图|生成|绘画|出图|画图)', 'AIGC', 'AI新兴技能'),
            (r'(Stable Diffusion|Midjourney|ComfyUI|FLUX|DALL.?E)', 'AIGC', 'AI新兴技能'),
            # 模型开发 → 推断默认技术栈（业内默认Python+PyTorch）
            (r'模型(开发|研发|设计|实现)', 'Python', '传统技术'),
            (r'模型(开发|研发|设计|实现)', 'PyTorch', 'AI新兴技能'),
            (r'模型(开发|研发|设计|实现)', '模型训练', 'AI新兴技能'),
            (r'模型(开发|研发|设计|实现)', '模型部署', 'AI新兴技能'),
            (r'(深度学习|机器学习|Deep Learning)', 'Python', '传统技术'),
            (r'(深度学习|机器学习|Deep Learning)', 'PyTorch', 'AI新兴技能'),
            (r'(自然语言处理|NLP)', 'Python', '传统技术'),
            (r'(计算机视觉|CV|图像识别|目标检测)', 'Python', '传统技术'),
            (r'(计算机视觉|CV|图像识别|目标检测)', 'OpenCV', '传统技术'),
            (r'(数据分析|数据挖掘|数据建模)', 'SQL', '传统技术'),
            (r'(数据分析|数据挖掘|数据建模)', 'Python', '传统技术'),
            # 原有口语化AI规则
            (r'(使用|利用|借助|通过).{0,5}AI.{0,5}(生成|创作|制作|写|画)', 'AIGC', 'AI新兴技能'),
            (r'(文案|文章|内容).{0,5}(生成|创作|写作)', 'AIGC', 'AI新兴技能'),
            (r'(AI|智能).{0,3}(视频|短片|动画)', 'AI视频生成', 'AI新兴技能'),
            (r'(模型|算法).{0,3}(训练|微调|调优)', '模型训练', 'AI新兴技能'),
            (r'(提示词|prompt|Prompt).{0,3}(编写|设计|优化|工程)', '提示工程', 'AI新兴技能'),
            (r'(大模型|大语言模型|LLM|GPT|ChatGPT|Claude|文心|通义|DeepSeek)', '大模型', 'AI新兴技能'),
            (r'(Agent|智能体|AI Agent)', 'Agent', 'AI新兴技能'),
            (r'(知识库|知识图谱|RAG|检索).{0,5}(搭建|构建|开发|应用)', 'RAG', 'AI新兴技能'),
            (r'(ComfyUI|SD|Stable Diffusion|FLUX)', 'ComfyUI', 'AI新兴技能'),
            (r'LoRA|Lora.{0,3}(训练|微调)', 'LoRA', 'AI新兴技能'),
            # ── 产品/运营/管理岗 → 从"做什么"推断"需要会什么" ──
            (r'(产品|功能|需求).{0,5}(规划|设计|迭代|路线|定义)', '产品设计', '传统技术'),
            (r'(需求|用户).{0,5}(分析|调研|挖掘|洞察)', '需求分析', '传统技术'),
            (r'(产品|项目).{0,5}(落地|交付|上线|推进|管理)', '项目管理', '传统技术'),
            (r'(数据|指标|效果).{0,5}(分析|追踪|监控|评估|复盘)', '数据分析', '传统技术'),
            (r'(竞品|竞争|市场|行业).{0,5}(分析|调研|跟踪|研究)', '竞品分析', '传统技术'),
            (r'(用户|客户).{0,5}(调研|访谈|体验|反馈|研究)', '用户研究', '传统技术'),
            (r'(PRD|MRD|BRD|产品文档|需求文档|产品规格)', 'PRD', '传统技术'),
            (r'(原型|线框图|交互|低保真|高保真)', '原型设计', '传统技术'),
            (r'(A/B|AB|灰度|实验).{0,5}(测试|实验)', 'A/B测试', '传统技术'),
            (r'(运营|增长|留存|转化|拉新|促活).{0,5}(策略|方案|计划|优化)', '用户运营', '传统技术'),
            (r'协调.{0,10}(团队|部门|资源|项目)', '项目管理', '传统技术'),
            (r'(技术文档|接口文档|开发文档|API文档|方案).{0,3}(编写|撰写|输出|维护)', '技术文档', '传统技术'),
            (r'(Axure|Figma|Sketch|墨刀|摹客)', 'Axure', '传统技术'),
            # 传统技能口语化
            (r'(Python|python).{0,5}(编程|开发|脚本|代码)', 'Python', '传统技术'),
            (r'(Java|java).{0,5}(编程|开发|后端)', 'Java', '传统技术'),
            (r'(SQL|sql|数据库).{0,5}(查询|操作|编写|管理)', 'SQL', '传统技术'),
            (r'(前端|网页|页面|H5|HTML).{0,5}(开发|制作|搭建)', '前端开发', '传统技术'),
            (r'(Excel|excel|表格|数据).{0,5}(处理|分析|统计)', '数据分析', '传统技术'),
            (r'(Linux|linux|服务器).{0,5}(操作|管理|部署|运维)', 'Linux', '传统技术'),
            (r'(公众号|小红书|抖音|新媒体).{0,5}(运营|编辑)', '新媒体运营', '传统技术'),
            (r'(视频|短视频).{0,5}(剪辑|制作|后期)', '视频剪辑', '传统技术'),
            # 通信/传感器/芯片推断
            (r'(PTN|SPN|OTN|光传输|传输网)', 'PTN', '传统技术'),
            (r'(激光雷达|毫米波雷达|超声波雷达)', '激光雷达', '传统技术'),
            (r'(多传感器|感知融合|传感器融合)', '多传感器融合', '传统技术'),
            (r'(MEMS|版图设计|数字IC|模拟IC)', 'MEMS', '传统技术'),
            (r'(自动驾驶|无人驾驶|智能驾驶).{0,5}(感知|融合|规划|控制)', '自动驾驶系统', '传统技术'),
        ]

        matched_positions = set()
        for pattern, skill_name, category in fuzzy_patterns:
            for m in re.finditer(pattern, jd_text, re.IGNORECASE):
                start, end = m.start(), m.end()
                # 检查是否与已匹配区域重叠
                if any(ps <= start < pe or ps < end <= pe for ps, pe in matched_positions):
                    continue
                matched_positions.add((start, end))
                evidence = jd_text[max(0, start-20):min(len(jd_text), end+30)]
                skills.append({
                    'skill': skill_name,
                    'category': category,
                    'evidence': evidence.strip(),
                })
                break  # 每种模式只匹配一次

        return skills

    def _coze_llm_extract(self, jd_text):
        """Coze API调用 — 与现有server.js同样的模式"""
        import urllib.request, urllib.error
        prompt = self._build_llm_prompt(jd_text)

        # Coze API需要实际token，这里保持接口一致
        # 实际部署时替换为真实API调用
        try:
            # TODO: 接入实际的Coze/其他LLM API
            # 当前fallback到本地模式
            return self._local_llm_extract(jd_text)
        except Exception:
            return self._local_llm_extract(jd_text)

    # ═══════════════════════════════════════
    #  第3级：RAG验证
    # ═══════════════════════════════════════

    def rag_verify(self, llm_skills, jd_text):
        """
        RAG验证LLM输出的每项技能：
        1. 技能库检索 → 是否在已验证技能库中
        2. JD原句匹配 → LLM声称的证据是否真的在JD中
        """
        verified = []

        for item in llm_skills:
            skill = item['skill']
            evidence = item.get('evidence', '')

            # 检查1：技能库
            in_taxonomy = skill in VERIFIED_SKILLS

            # 检查2：JD原句是否真的包含这个技能的关键词
            evidence_in_jd = evidence and len(evidence) > 3 and evidence[:30] in jd_text

            # 检查3：技能名或近义词是否出现在JD中
            skill_in_jd = skill.lower() in jd_text.lower()

            if in_taxonomy and (evidence_in_jd or skill_in_jd):
                # 双锚定 → 高置信度
                item['rag_status'] = 'verified'
                item['rag_confidence'] = 0.9
                item['rag_reason'] = '技能库命中 + JD原文匹配'
                self.stats['rag_verified'] += 1
            elif in_taxonomy:
                # 仅技能库 → 中置信度
                item['rag_status'] = 'needs_review'
                item['rag_confidence'] = 0.6
                item['rag_reason'] = '技能库命中但JD原文匹配弱'
                self.stats['rag_needs_review'] += 1
            elif skill_in_jd:
                # 仅JD匹配 → 可能是新技能
                item['rag_status'] = 'needs_review'
                item['rag_confidence'] = 0.5
                item['rag_reason'] = 'JD提及但不在一已验证技能库中，可能是新技术'
                self.stats['rag_needs_review'] += 1
            else:
                # 双缺失 → 幻觉
                item['rag_status'] = 'hallucinated'
                item['rag_confidence'] = 0.1
                item['rag_reason'] = '技能库无 + JD原文无匹配 → 疑似LLM幻觉'
                self.stats['rag_hallucinated'] += 1

            verified.append(item)

        return verified

    # ═══════════════════════════════════════
    #  第4级：职责→技能隐式推断
    # ═══════════════════════════════════════

    DUTY_SKILL_MAP = [
        # (职责正则, 技能, 类别)
        (r'(制定|策划|执行|优化).{0,5}(推广|营销|市场).{0,5}(策略|方案|计划)', '市场营销', '传统技术'),
        (r'(带领|管理|协调|组织).{0,5}(团队|部门|项目组)', '团队管理', '传统技术'),
        (r'(分析|追踪|评估|监控|复盘).{0,5}(数据|指标|KPI|效果)', '数据分析', '传统技术'),
        (r'(推动|负责|主导).{0,5}(产品|项目).{0,5}(落地|交付|上线|迭代|规划)', '项目管理', '传统技术'),
        (r'(收集|调研|挖掘|洞察).{0,5}(需求|用户|市场|客户)', '需求分析', '传统技术'),
        (r'(协调|对接|推动).{0,5}(跨部门|多方|资源|外部)', '跨部门协作', '传统技术'),
        (r'(撰写|编写|输出|维护).{0,5}(文档|报告|方案|SOP)', '技术文档', '传统技术'),
        (r'(培训|指导|带教|培养).{0,5}(新人|团队|下属|成员)', '培训指导', '传统技术'),
        (r'(规划|设计|搭建).{0,5}(流程|体系|制度|架构)', '流程设计', '传统技术'),
        (r'(商务|客户).{0,5}(谈判|对接|沟通|维护)', '商务沟通', '传统技术'),
        (r'(AI|算法|模型|智能).{0,5}(落地|应用|部署|集成)', 'AI应用', 'AI新兴技能'),
        (r'(大模型|LLM|GPT|Agent).{0,5}(应用|调用|接入)', '大模型应用', 'AI新兴技能'),
        # 通信/传感器
        (r'(5G|LTE|NR|基站|天线).{0,5}(规划|优化|测试|设计)', '5G', '传统技术'),
        (r'(雷达|感知|检测|识别).{0,5}(算法|系统|模型)', '感知算法', '传统技术'),
        (r'(芯片|IC|SoC).{0,5}(设计|验证|流程|开发)', '芯片设计', '传统技术'),
    ]

    def infer_from_duties(self, jd_text):
        """从岗位职责描述推断隐含技能"""
        skills = []
        matched_positions = set()
        for pattern, skill_name, category in self.DUTY_SKILL_MAP:
            for m in re.finditer(pattern, jd_text):
                start, end = m.start(), m.end()
                if any(ps <= start < pe or ps < end <= pe for ps, pe in matched_positions):
                    continue
                matched_positions.add((start, end))
                evidence = jd_text[max(0, start-20):min(len(jd_text), end+30)]
                skills.append({
                    'skill': skill_name,
                    'category': category,
                    'evidence': evidence.strip(),
                    'source': 'inferred_duty',
                    'confidence': 0.5,
                })
                break
        return skills

    # ═══════════════════════════════════════
    #  第5级：同类岗位协同过滤
    # ═══════════════════════════════════════

    _job_stats_cache = None  # 懒加载

    @classmethod
    def _load_job_stats(cls):
        """加载全量标注数据中岗位→高频技能的映射"""
        if cls._job_stats_cache is not None:
            return cls._job_stats_cache
        import csv, glob, os
        from collections import defaultdict, Counter
        anno_dir = os.path.join(os.path.dirname(__file__), 'skill_ner_release', 'data', 'annotated')
        if not os.path.exists(anno_dir):
            cls._job_stats_cache = {}
            return cls._job_stats_cache

        job_skills = defaultdict(Counter)
        for fpath in glob.glob(os.path.join(anno_dir, '*.csv')):
            with open(fpath, 'r', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    job = str(row.get('job_name', '')).strip()
                    essential = str(row.get('必备技能', ''))
                    for m in re.finditer(r'【(.+?)｜', essential):
                        job_skills[job][m.group(1)] += 1

        cls._job_stats_cache = job_skills
        return job_skills

    def collaborative_filter(self, job_name, top_k=5):
        """找同类岗位的高频必备技能"""
        job_stats = self._load_job_stats()
        if not job_stats:
            return []

        # 精确匹配
        if job_name in job_stats:
            return [{'skill': s, 'count': c, 'source': 'collaborative_filter', 'confidence': 0.35}
                    for s, c in job_stats[job_name].most_common(top_k)]

        # 模糊匹配：岗位名包含关键词
        candidates = Counter()
        for job, skills in job_stats.items():
            # 岗位名相似度
            if any(part in job for part in job_name.split('_')) or any(part in job for part in job_name.split('-')):
                for skill, count in skills.items():
                    candidates[skill] += count
        if candidates:
            return [{'skill': s, 'count': c, 'source': 'collaborative_filter', 'confidence': 0.3}
                    for s, c in candidates.most_common(top_k)]
        return []

    # ═══════════════════════════════════════
    #  主管线（五级）
    # ═══════════════════════════════════════

    def extract(self, jd_text, job_name=''):
        # 第1级：字典
        dict_skills, dict_empty = self.dict_extract(jd_text)
        if not dict_empty:
            return self._format_output(dict_skills, tier=1)

        # 第2级：LLM语义推断
        self.stats['llm_called'] += 1
        llm_skills = self.llm_extract(jd_text, job_name)
        if llm_skills:
            verified_skills = self.rag_verify(llm_skills, jd_text)
            valid = [s for s in verified_skills if s['rag_status'] != 'hallucinated']
            result = {}
            for s in valid:
                result[s['skill']] = s.get('evidence', '')
            if result:
                return self._format_output(result, tier=2, rag_details=verified_skills)

        # 第3级：第三类——极简JD直接跳过
        if len(jd_text.strip()) < 60 and job_name in ('', '未知'):
            return self._format_output({}, tier=3, empty=True, source='invalid_skip')

        # 第4级：职责→技能隐式推断
        duty_skills = self.infer_from_duties(jd_text)
        if duty_skills:
            verified = self.rag_verify(duty_skills, jd_text)
            valid = [s for s in verified if s['rag_status'] != 'hallucinated']
            if valid:
                result = {}
                for s in valid:
                    result[s['skill']] = s.get('evidence', '')
                # 打上source标记，区分于直接抽取
                return self._format_output(result, tier=4, source='inferred_duty',
                                           confidence=0.5)

        # 第5级：协同过滤
        cf_skills = self.collaborative_filter(job_name, top_k=5)
        if cf_skills:
            cf_skill_dict = {}
            for s in cf_skills:
                count = s['count']
                cf_skill_dict[s['skill']] = f'同类岗位高频技能(出现{count}次)'
            verified = self.rag_verify(cf_skills, jd_text)
            valid = [s for s in verified if s['rag_status'] != 'hallucinated']
            if valid:
                result = {}
                for s in valid:
                    cnt = s.get('count', '?')
                    result[s['skill']] = f'同类岗位协同过滤推荐(出现{cnt}次)'
                return self._format_output(result, tier=5, source='collaborative_filter',
                                           confidence=0.35)

        # 全部失败
        return self._format_output({}, tier=5, empty=True)

    def _format_output(self, skills_dict, tier=1, empty=False, rag_details=None, source='extracted', confidence=0.85):
        """格式化输出，兼容annotate_v3格式"""
        if empty or not skills_dict:
            return {
                'tier': tier,
                'skills': [],
                'essential_str': '（JD中未识别到明确的必备硬性技能）',
                'bonus_str': '（未识别到明确的加分技能）',
                'essential_count': 0,
                'bonus_count': 0,
                'source': source,
                'confidence': confidence,
                'rag_details': rag_details or [],
            }

        required = []
        bonus = []
        for skill, ctx in skills_dict.items():
            if is_soft_skill(skill):
                continue
            cat = classify_skill(skill, ctx)
            level = get_mastery_level(ctx)
            if level == '了解' and is_bonus_signal(ctx):
                bonus.append((skill, cat, level))
            elif is_bonus_signal(ctx):
                bonus.append((skill, cat, level))
            else:
                required.append((skill, cat, level))

        # 排序：掌握级别优先
        level_order = {'精通': 0, '熟练': 1, '熟悉': 2, '了解': 3}
        required.sort(key=lambda x: (level_order.get(x[2], 2), 0 if x[1] == 'AI新兴技能' else 1))
        bonus.sort(key=lambda x: (level_order.get(x[2], 2), 0 if x[1] == 'AI新兴技能' else 1))

        # 必备为空时从加分补
        if not required and bonus:
            required.append(bonus.pop(0))

        def fmt(skills_list):
            return '；'.join([f'【{s[0]}｜{s[1]}｜{s[2]}】' for s in skills_list])

        return {
            'tier': tier,
            'source': source,
            'confidence': confidence,
            'skills': [{'skill': s[0], 'category': s[1], 'mastery': s[2]} for s in required + bonus],
            'essential_str': fmt(required) if required else '（JD中未识别到明确的必备硬性技能）',
            'bonus_str': fmt(bonus) if bonus else '（未识别到明确的加分技能）',
            'essential_count': len(required),
            'bonus_count': len(bonus),
            'rag_details': rag_details or [],
        }


# ═══════════════════════════════════════
#  演示
# ═══════════════════════════════════════

if __name__ == '__main__':
    extractor = LLMSkillExtractor(llm_mode='local')

    test_cases = [
        # 字典能抽到的（第1级直接过）
        ('Python开发工程师', '岗位职责：1. 熟练使用Python进行后端开发。2. 掌握Django框架。3. 熟悉MySQL数据库。'),
        # 字典抽不到的（第2级+第3级）
        ('AI运营专员', '岗位要求：1. 热爱AI行业。2. 协助团队使用AI工具完成文案生成和视频剪辑。3. 大专学历。4. 有无经验均可。'),
        ('AIGC讲师', '岗位职责：1. 讲授AI短剧、AI动态漫课程，从剧本到成片全流程教学。2. 制定课程大纲，指导学生实操AI绘画、AI视频工具。3. 批改作业，跟踪学员作品产出。'),
        ('AI算法工程师', '岗位职责：1. 配合上级部门要求，完成模型开发相关工作。2. 了解新技术和行业环境。3. 完成专利材料的编写。'),
    ]

    print('=' * 70)
    print('  三级抽取管线演示：字典 → LLM → RAG')
    print('=' * 70)

    for job, text in test_cases:
        result = extractor.extract(text, job)
        tier_tag = {1: '[Dict]', 2: '[LLM+RAG]'}.get(result['tier'], '[?]')
        print(f'\n{tier_tag} {job}')
        print(f'  必备({result["essential_count"]}): {result["essential_str"][:120]}')
        if result['bonus_count']:
            print(f'  加分({result["bonus_count"]}): {result["bonus_str"][:120]}')

        # 显示RAG验证细节
        rag = result.get('rag_details', [])
        for r in rag:
            status_tag = {'verified': '[OK]', 'needs_review': '[REVIEW]', 'hallucinated': '[HAL]'}
            tag = status_tag.get(r.get('rag_status', ''), '[?]')
            reason = r.get('rag_reason', '')
            print(f'    {tag} {r["skill"]}: {reason}')

    print('\n' + '=' * 70)
    print('  统计:')
    print(f'    字典命中: {extractor.stats["dict_hit"]}')
    print(f'    LLM调用: {extractor.stats["llm_called"]}')
    print(f'    RAG验证通过: {extractor.stats["rag_verified"]}')
    print(f'    RAG需审核: {extractor.stats["rag_needs_review"]}')
    print(f'    RAG确认幻觉: {extractor.stats["rag_hallucinated"]}')
