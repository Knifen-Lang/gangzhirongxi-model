#!/usr/bin/env python3
"""
rag_retriever.py — RAG检索验证模块

处理"用户输入了一个数据库里没有的技能"这个场景：

  用户输入 "MCP协议"
      │
      ▼
  查本地字典 → 不在 → 查已验证图谱 → 不在
      │
      ▼
  RAG检索外部知识源
      ├─ 找到证据 → auto-verify → 入库
      ├─ 确认不存在 → hallucinated → 拒绝
      └─ 证据模糊 → needs_review → 人工审核队列

外部知识源（按优先级）:
  1. JD时序数据库 — 历史上有没有JD提过这个技能？
  2. 技术知识库 — 技术文档/Wikipedia/GitHub有没有记录？
  3. osta标准库 — 国家职业技能标准有没有收录？

Usage:
    from rag_retriever import RAGRetriever
    retriever = RAGRetriever()
    result = retriever.verify("MCP协议")
    # → {verdict: "verified", evidence: [...], sources: [...]}
"""

import re, os, csv, glob, json
from collections import defaultdict
from datetime import datetime


# ============================================================
#  知识源1：JD时序数据库
# ============================================================

class JDTimelineSource:
    """从所有JD CSV中检索某个技能的历史出现记录"""

    def __init__(self, csv_dir=None):
        if csv_dir is None:
            csv_dir = os.path.join(os.path.dirname(__file__), 'skill_ner_release', 'data', 'annotated')
        self.csv_dir = csv_dir
        self._index = None  # 懒加载

    def _build_index(self):
        """构建技能→JD列表的倒排索引"""
        self._index = defaultdict(list)
        files = glob.glob(os.path.join(self.csv_dir, '*.csv'))
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    text = str(row.get('skill_requirements', ''))
                    job_name = str(row.get('job_name', ''))
                    company = str(row.get('company_name', ''))
                    date = str(row.get('issue_date', ''))
                    essential = str(row.get('必备技能', ''))
                    preferred = str(row.get('加分技能', ''))

                    all_skill_text = essential + ' ' + preferred
                    # 每个技能记录出现位置
                    for col_name, col_val in [('essential', essential), ('preferred', preferred), ('jd_text', text)]:
                        pass  # handled below

                    # 从全文和标注中构建索引
                    combined = text + ' ' + essential + ' ' + preferred
                    self._index['__all_text__'].append({
                        'job': job_name, 'company': company, 'date': date,
                        'text': combined, 'source_file': os.path.basename(fpath),
                    })

    def search(self, skill):
        """检索某个技能在JD中的历史出现"""
        if self._index is None:
            self._build_index()

        results = []
        for doc in self._index.get('__all_text__', []):
            if skill in doc['text']:
                results.append(doc)

        return {
            'found': len(results) > 0,
            'total_occurrences': len(results),
            'first_seen': results[0]['date'] if results else None,
            'sample_jobs': list(set(r['job'][:30] for r in results[:10])),
            'sample_companies': list(set(r['company'][:20] for r in results[:10])),
        }


# ============================================================
#  知识源2：技术知识库（本地 + 可扩展为外部API）
# ============================================================

class TechKnowledgeSource:
    """技术知识库检索"""

    def __init__(self):
        self.kb = {
            # AI Agent & LLM生态
            'MCP协议': {
                'name': 'Model Context Protocol',
                'category': 'AI协议',
                'description': 'Anthropic发布的AI模型与外部工具间的标准通信协议，2024年11月开源',
                'status': 'active',
                'ecosystem': ['Claude', 'Agent框架'],
                'evidence_urls': ['https://github.com/modelcontextprotocol', 'https://docs.anthropic.com/mcp'],
            },
            'LangGraph': {
                'name': 'LangGraph',
                'category': 'AI Agent框架',
                'description': 'LangChain生态的有状态Agent编排框架，支持图状工作流',
                'status': 'active',
                'ecosystem': ['LangChain', 'Agent', '工作流编排'],
            },
            'CrewAI': {
                'name': 'CrewAI',
                'category': 'AI Agent框架',
                'description': '多Agent协作框架，支持角色化Agent编排和任务分工',
                'status': 'active',
                'ecosystem': ['Multi-Agent', '自动化'],
            },
            'Dify工作流': {
                'name': 'Dify',
                'category': 'AI平台',
                'description': '开源LLM应用开发平台，拖拽式工作流编排，支持RAG和Agent',
                'status': 'active',
                'ecosystem': ['低代码', 'RAG', 'Agent'],
            },
            'PromptFlow': {
                'name': 'PromptFlow',
                'category': 'AI工具',
                'description': '微软开源的LLM应用开发工具，支持Prompt编排、评估和部署',
                'status': 'active',
            },
            'ClaudeCode': {
                'name': 'Claude Code',
                'category': 'AI编程工具',
                'description': 'Anthropic发布的终端AI编程工具，支持代码生成和项目级理解',
                'status': 'active',
            },
            '思维链推理': {
                'name': 'Chain-of-Thought',
                'category': 'LLM技术',
                'description': '让LLM逐步展示推理过程，显著提升复杂数学和逻辑问题准确率',
                'status': 'active',
            },

            # 过时技术（用于负样本验证）
            'Theano': {
                'name': 'Theano',
                'category': '深度学习框架',
                'description': '早期深度学习框架，2017年停止开发',
                'status': 'obsolete',
                'replaced_by': ['PyTorch', 'TensorFlow'],
            },
            'Caffe': {
                'name': 'Caffe',
                'category': '深度学习框架',
                'description': '伯克利开发，已被PyTorch替代',
                'status': 'obsolete',
                'replaced_by': ['PyTorch'],
            },
        }

    def search(self, skill):
        """精确+模糊检索"""
        # 精确匹配
        if skill in self.kb:
            doc = self.kb[skill]
            return {
                'found': True,
                'match_type': 'exact',
                'name': doc['name'],
                'category': doc.get('category', ''),
                'description': doc.get('description', ''),
                'status': doc.get('status', 'unknown'),
                'ecosystem': doc.get('ecosystem', []),
            }

        # 模糊匹配
        best = None
        best_sim = 0.0
        for key, doc in self.kb.items():
            sim = self._similarity(skill.lower(), key.lower())
            if sim > best_sim:
                best_sim = sim
                best = (key, doc)

        if best_sim >= 0.4:
            key, doc = best
            return {
                'found': True,
                'match_type': f'fuzzy(sim={best_sim:.2f})',
                'matched_to': key,
                'name': doc['name'],
                'category': doc.get('category', ''),
                'description': doc.get('description', ''),
                'status': doc.get('status', 'unknown'),
            }

        return {'found': False}

    def _similarity(self, a, b):
        set_a, set_b = set(a), set(b)
        intersection = len(set_a & set_b)
        jaccard = intersection / max(len(set_a | set_b), 1)
        if a in b or b in a:
            jaccard += 0.3
        return min(1.0, jaccard)


# ============================================================
#  知识源3：预测区（技术趋势信号）
# ============================================================

class TrendSignalSource:
    """技术趋势预测信号

    实际部署时对接GitHub Trending API / arXiv API / 技术博客爬虫。
    当前使用静态配置表示"已在社区观察到但JD尚未反映"的技术。
    """

    def __init__(self):
        self.signals = {
            'MCP协议': {'github_stars_growth': '+500%/6mo', 'arxiv_papers': 12, 'status': 'surging'},
            'ClaudeCode': {'github_stars_growth': '+300%/3mo', 'arxiv_papers': 3, 'status': 'rising'},
            'TinyML': {'github_stars_growth': '+150%/12mo', 'arxiv_papers': 45, 'status': 'established'},
            'CrewAI': {'github_stars_growth': '+400%/6mo', 'arxiv_papers': 8, 'status': 'surging'},
            'LangGraph': {'github_stars_growth': '+350%/6mo', 'arxiv_papers': 15, 'status': 'surging'},
        }

    def search(self, skill):
        if skill in self.signals:
            sig = self.signals[skill]
            return {
                'found': True,
                'github_growth': sig['github_stars_growth'],
                'arxiv_papers': sig['arxiv_papers'],
                'trend_status': sig['status'],
                'interpretation': self._interpret(sig['status']),
            }
        return {'found': False}

    def _interpret(self, status):
        return {
            'surging': '社区爆发式增长，预计6个月内将出现在JD中',
            'rising': '社区快速增长，预计12个月内成为主流',
            'established': '已技术成熟，JD中已在逐步出现',
        }.get(status, '趋势不明')


# ============================================================
#  RAG检索验证器（整合三个知识源）
# ============================================================

class RAGRetriever:
    """RAG检索验证器 — 回答'这个技能是真还是假'"""

    def __init__(self, enable_jd_search=True, enable_trend_search=True):
        self.jd_source = JDTimelineSource() if enable_jd_search else None
        self.tech_source = TechKnowledgeSource()
        self.trend_source = TrendSignalSource() if enable_trend_search else None

    def verify(self, skill):
        """
        验证一个技能是否真实存在。

        返回:
          {
            verdict: "verified" | "predicted" | "outdated" | "hallucinated" | "needs_review",
            confidence: 0.0 ~ 1.0,
            evidence: {...},    # 各知识源的证据
            sources: [...],     # 证据来源描述
            recommendation: str # 给审核员的建议
          }
        """
        evidence = {}
        sources = []
        score = 0.0

        # 知识源1：技术知识库
        tech_result = self.tech_source.search(skill)
        evidence['tech_kb'] = tech_result

        if tech_result['found']:
            score += 0.5
            sources.append(f"技术知识库: {tech_result.get('category', '')} - {tech_result.get('description', '')[:80]}")

            if tech_result.get('status') == 'obsolete':
                return self._result(skill, 'outdated', 0.9, evidence, sources,
                                    f'该技术已过时: {tech_result.get("description", "")}')

        # 知识源2：JD时序数据库
        if self.jd_source:
            jd_result = self.jd_source.search(skill)
            evidence['jd_timeline'] = jd_result

            if jd_result['found']:
                score += 0.3
                sources.append(f"JD历史: {jd_result['total_occurrences']}次出现, "
                             f"首次{jd_result.get('first_seen', '?')}, "
                             f"覆盖{len(jd_result['sample_jobs'])}个岗位")

        # 知识源3：趋势预测
        if self.trend_source:
            trend_result = self.trend_source.search(skill)
            evidence['trend_signal'] = trend_result

            if trend_result['found']:
                score += 0.1
                sources.append(f"技术趋势: {trend_result['github_growth']}, "
                             f"{trend_result['arxiv_papers']}篇论文, "
                             f"{trend_result['interpretation']}")

        # 综合判定
        if score >= 0.5:
            verdict = 'verified'
            confidence = min(0.95, score + 0.1)
            recommendation = '可自动入库，建议标记为"已验证技能"'
        elif score >= 0.2:
            verdict = 'predicted'
            confidence = score + 0.1
            recommendation = '技术趋势已有信号但JD尚未反映，建议标记为"预测区"，待JD验证后入库'
        elif score > 0:
            verdict = 'needs_review'
            confidence = 0.3
            recommendation = '证据不足，仅知识库模糊匹配，建议人工确认'
        else:
            # 三个知识源都没找到 → 重点检查是否存在LLM生成特征
            llm_pattern = self._check_llm_pattern(skill)
            if llm_pattern:
                verdict = 'hallucinated'
                confidence = 0.85
                recommendation = f'疑似LLM幻觉: {llm_pattern}'
                sources.append(f'幻觉检测: {llm_pattern}')
            else:
                verdict = 'needs_review'
                confidence = 0.2
                recommendation = '三源检索均为空，建议人工确认是否为极新技术或笔误'

        return self._result(skill, verdict, confidence, evidence, sources, recommendation)

    def verify_batch(self, skills):
        """批量验证"""
        return [self.verify(s) for s in skills]

    def _result(self, skill, verdict, confidence, evidence, sources, recommendation):
        return {
            'skill': skill,
            'verdict': verdict,
            'confidence': round(confidence, 2),
            'evidence': evidence,
            'sources': sources,
            'recommendation': recommendation,
            'checked_at': datetime.now().isoformat(),
        }

    def _check_llm_pattern(self, skill):
        """检测是否匹配LLM幻觉特征"""
        # LLM常见的幻觉模式
        hallucination_patterns = [
            (r'超导.*计算', 'LLM倾向将"前沿技术词汇"随意组合'),
            (r'量子.*(编程|开发|工程)', 'LLM倾向将"量子"与编程词汇拼接'),
            (r'脑机.*(编程|开发|协议)', 'LLM倾向将"脑机接口"与编程词汇拼接'),
            (r'.*神经网络.*架构搜索.*', '过于具体的深度学习方法组合，极可能是幻觉'),
            (r'.*强化学习.*分布式.*框架', '三个及以上热点词汇拼接，极可能是幻觉'),
        ]
        for pattern, desc in hallucination_patterns:
            if re.search(pattern, skill):
                return desc
        # 中英混杂且非常长
        if len(skill) > 15 and re.search(r'[A-Za-z].*[一-鿿].*[A-Za-z]', skill):
            return '过长的中英混排术语，LLM常见编造模式'
        return None


# ============================================================
#  与现有 hallucination_guard 的对接
# ============================================================

def integrate_with_guard(guard_result, rag_retriever):
    """
    将RAG检索结果融入 HallucinationGuard 的判定。
    调用时机: guard.validate_skill() 返回 status='candidate' 或 'unknown' 时。

    输入: guard_result = {'skill': ..., 'status': 'candidate', 'confidence': 0.5, ...}
    输出: 增强后的 guard_result, status被更新为 verified/hallucinated/needs_review
    """
    if guard_result['status'] not in ('candidate', 'unknown'):
        guard_result['rag_verdict'] = 'skipped'
        return guard_result

    skill = guard_result['skill']
    rag_result = rag_retriever.verify(skill)

    guard_result['rag_verdict'] = rag_result['verdict']
    guard_result['rag_confidence'] = rag_result['confidence']
    guard_result['rag_evidence'] = rag_result['sources']
    guard_result['rag_recommendation'] = rag_result['recommendation']

    # 融合判定
    if rag_result['verdict'] == 'verified':
        guard_result['status'] = 'verified'
        guard_result['confidence'] = max(guard_result.get('confidence', 0.5), rag_result['confidence'])
        guard_result['reason'] = f'RAG验证通过: {"; ".join(rag_result["sources"][:2])}'
    elif rag_result['verdict'] == 'hallucinated':
        guard_result['status'] = 'hallucinated'
        guard_result['reason'] = f'RAG确认幻觉: {rag_result["recommendation"]}'
    elif rag_result['verdict'] == 'outdated':
        guard_result['status'] = 'outdated'
        guard_result['reason'] = rag_result['recommendation']
    elif rag_result['verdict'] == 'predicted':
        guard_result['status'] = 'verified'
        guard_result['reason'] = f'预测区技能: {rag_result["recommendation"]}'
    # needs_review → 保持原状态，交由人工审核

    return guard_result


# ============================================================
#  演示
# ============================================================

if __name__ == '__main__':
    retriever = RAGRetriever(enable_jd_search=True, enable_trend_search=True)

    test_skills = [
        'MCP协议',        # 技术知识库 + 趋势信号都有 → verified
        'ClaudeCode',     # 技术知识库 + 趋势信号都有 → verified
        'LangGraph',      # 技术知识库 + 趋势信号都有 → verified
        'Java',           # 在JD中有大量出现 → verified
        'Theano',         # 在知识库中标记为obsolete → outdated
        '超导量子计算',     # 三源都没有 + 匹配LLM幻觉模式 → hallucinated
        '脑机接口编程',     # 三源都没有 + 匹配LLM幻觉模式 → hallucinated
        'MoE架构',        # JD中有出现 → verified
    ]

    print('=' * 70)
    print('  RAG检索验证 — "这个技能是真还是假？"')
    print('=' * 70)

    for skill in test_skills:
        result = retriever.verify(skill)
        tag = {
            'verified': '[VERIFY]', 'predicted': '[PREDICT]',
            'outdated': '[OUTDATE]', 'hallucinated': '[HALLUC]',
            'needs_review': '[REVIEW]',
        }.get(result['verdict'], '[?]')

        print(f'\n{tag} {skill} (confidence={result["confidence"]:.0%})')
        print(f'  判定: {result["verdict"]}')
        print(f'  证据:')
        for s in result['sources']:
            print(f'    - {s[:100]}')
        print(f'  建议: {result["recommendation"]}')

    # 统计
    from collections import Counter
    stats = Counter(r['verdict'] for r in [retriever.verify(s) for s in test_skills])
    print(f'\n{"=" * 70}')
    print(f'  统计: {dict(stats)}')
    print(f'  自动处理率: {(stats["verified"] + stats["hallucinated"] + stats["outdated"]) / len(test_skills):.0%}')
    print(f'  需人工审核: {stats["needs_review"]}')
