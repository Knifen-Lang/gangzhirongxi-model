#!/usr/bin/env python3
"""
hallucination_guard.py — 能力图谱幻觉防控模块

三阶段防控：
  1. NER白名单校验：检查抽取技能是否在已验证技能库中
  2. LLM约束输出：确保生成的技能定义不编造新技能
  3. 本体论验证：图谱边的合理性检查

复用 annotate_v3.py 的 AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS, OUTDATED_SKILLS

Usage:
    from hallucination_guard import HallucinationGuard
    guard = HallucinationGuard(skill_taxonomy)
    result = guard.validate_skills(extracted_skills, confidence_scores)
"""

import re, sys, os, json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '【已标注v3】'))
from annotate_v3 import (
    AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS, OUTDATED_SKILLS,
)


# ── RAG知识库 ──
TECH_KNOWLEDGE_BASE = {
    'Context Engine': {'description': 'AI Agent架构核心组件，管理对话上下文和记忆状态', 'category': 'AI Agent', 'sources': ['LangChain', 'Anthropic Agent Guide']},
    'Harness Engineering': {'description': 'AI系统工程化部署与运维，MLOps/LLMOps常见概念', 'category': 'MLOps', 'sources': ['Google MLOps', 'HF Inference']},
    'Memory Engine': {'description': 'Agent记忆管理模块，短期记忆+长期记忆', 'category': 'AI Agent', 'sources': ['LangChain Memory', 'MemGPT']},
    'Plan-and-Execute': {'description': 'Agent架构模式，先规划再执行', 'category': 'AI Agent', 'sources': ['LangGraph']},
    'Agent Loop': {'description': 'Agent感知-思考-行动循环机制', 'category': 'AI Agent', 'sources': ['OpenAI Agents SDK', 'CrewAI']},
    'vLLM': {'description': '高性能LLM推理引擎，PagedAttention', 'category': 'LLM推理', 'sources': ['vLLM GitHub']},
    'SGLang': {'description': 'LLM推理编程框架，结构化生成', 'category': 'LLM推理', 'sources': ['SGLang GitHub']},
    'Graph RAG': {'description': '结合知识图谱的RAG方法，多跳推理', 'category': 'RAG', 'sources': ['Microsoft GraphRAG']},
    'RAG': {'description': '检索增强生成，结合外部知识库提升LLM准确性', 'category': 'RAG', 'sources': ['Meta FAIR', 'LangChain']},
    'MCP': {'description': 'Model Context Protocol，AI模型上下文协议', 'category': 'AI协议', 'sources': ['Anthropic MCP']},
    'AI Agent': {'description': '基于LLM的自主智能体，能感知环境并执行行动', 'category': 'AI Agent', 'sources': ['OpenAI', 'LangChain', 'CrewAI']},
    'Agentic RAG': {'description': '结合Agent能力的RAG，自主决定检索策略', 'category': 'RAG', 'sources': ['LlamaIndex']},
    'Multi-Agent': {'description': '多智能体协作系统', 'category': 'AI Agent', 'sources': ['CrewAI', 'AutoGen']},
    'Deep Research': {'description': 'AI深度研究能力，自主搜索+综合分析', 'category': 'AI应用', 'sources': ['OpenAI', 'Gemini']},
    'MCP协议': {'description': 'Model Context Protocol，AI模型与工具间的标准通信协议', 'category': 'AI协议', 'sources': ['Anthropic MCP Spec']},
    'Dify工作流': {'description': 'Dify是开源LLM应用开发平台，支持可视化工作流编排', 'category': 'AI平台', 'sources': ['Dify GitHub', 'Dify Docs']},
    'PromptFlow': {'description': '微软开源的LLM应用开发工具包，支持Prompt编排和评估', 'category': 'AI工具', 'sources': ['Microsoft PromptFlow GitHub']},
    'ClaudeCode': {'description': 'Anthropic发布的AI编程CLI工具，支持终端内代码生成', 'category': 'AI工具', 'sources': ['Anthropic Blog', 'Claude Code Docs']},
    '思维链推理': {'description': 'Chain-of-Thought推理技术，让LLM逐步思考提升复杂问题准确率', 'category': 'LLM技术', 'sources': ['Google CoT论文', 'OpenAI o1']},
    'TinyML': {'description': '微型机器学习，在嵌入式设备上运行ML模型', 'category': '边缘AI', 'sources': ['TinyML Foundation', 'TensorFlow Lite']},
    'CrewAI': {'description': '多Agent协作框架，支持角色化Agent编排', 'category': 'AI Agent', 'sources': ['CrewAI GitHub']},
    'LangGraph': {'description': 'LangChain生态的Agent状态图编排框架', 'category': 'AI Agent', 'sources': ['LangGraph Docs']},
}


class RAGChecker:
    """轻量RAG检索器 - 嵌入在HallucinationGuard内"""

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base or TECH_KNOWLEDGE_BASE

    def search(self, skill):
        """检索知识库"""
        if skill in self.kb:
            doc = self.kb[skill]
            return [{'skill': skill, 'similarity': 1.0, 'description': doc['description'],
                     'category': doc['category'], 'sources': doc['sources']}]
        # 模糊匹配
        results = []
        for key, doc in self.kb.items():
            sim = self._similarity(skill.lower(), key.lower())
            if sim >= 0.3:
                results.append({'skill': key, 'similarity': round(sim, 3),
                               'description': doc['description'][:120],
                               'category': doc['category'], 'sources': doc['sources']})
        results.sort(key=lambda x: -x['similarity'])
        return results

    def verify(self, skill):
        """RAG验证：返回(verdict, reasoning, evidence)"""
        docs = self.search(skill)
        if not docs:
            return 'hallucinated', '知识库无匹配', []
        sims = [d['similarity'] for d in docs]
        avg = sum(sims) / len(sims)
        if avg >= 0.7 and len(docs) >= 1:
            return 'verified_by_rag', f'检索到{len(docs)}条证据，avg_sim={avg:.2f}', docs[:3]
        elif avg >= 0.3:
            return 'needs_human_review', f'检索到{len(docs)}条但置信度不足(avg_sim={avg:.2f})', docs[:3]
        else:
            return 'hallucinated', f'相似度过低(avg_sim={avg:.2f})', []

    def _similarity(self, a, b):
        set_a, set_b = set(a), set(b)
        intersection = len(set_a & set_b)
        jaccard = intersection / max(len(set_a | set_b), 1)
        bonus = 0.3 if (a in b or b in a) else (0.15 if intersection >= 3 else 0)
        return min(1.0, jaccard + bonus)


class HallucinationGuard:
    """能力图谱幻觉防控器（含RAG增强）"""

    def __init__(self, enable_rag=True, rag_retriever=None):
        # 已验证技能库
        self.verified_skills = AI_SKILLS | TRADITIONAL_SKILLS

        # 软技能黑名单
        self.soft_skills = SOFT_SKILLS

        # 过时技能表
        self.outdated_skills = {skill: reason for skill, reason in OUTDATED_SKILLS}

        # RAG检索器（可注入外部实例）
        self.enable_rag = enable_rag
        if rag_retriever is not None:
            self.rag = rag_retriever  # 使用外部全功能RAG
        elif enable_rag:
            self.rag = RAGChecker()   # 使用内置轻量RAG
        else:
            self.rag = None

        # 本体论规则
        self.ontology_rules = {
            'skill_requires_job': {
                'min_evidence': 2,  # 至少2条JD支撑
                'description': '技能→岗位关系需至少2条JD证据',
            },
            'skill_is_subtype_of': {
                'description': '技能从属关系必须是技术范畴包含',
                'validator': self._validate_subtype,
            },
            'job_is_similar_to': {
                'min_similarity': 0.3,
                'max_similarity': 0.95,
                'description': '岗位相似度需在[0.3, 0.95]之间',
            },
        }

    # ── 第1层：NER白名单校验 ──
    def validate_skill(self, skill, confidence=0.5):
        """
        验证单个技能是否可信
        返回: {skill, status, confidence, reason}
        """
        # 空技能
        if not skill or len(skill) < 2 or len(skill) > 30:
            return {'skill': skill, 'status': 'rejected', 'confidence': 0.0,
                    'reason': '技能名长度异常'}

        # 软技能
        if skill in self.soft_skills:
            return {'skill': skill, 'status': 'rejected', 'confidence': 0.0,
                    'reason': '软技能，不能作为技术技能'}

        # 识别常见的NER误抽模式（描述性短语而非具体技能）
        hallucination_patterns = [
            r'^(具有|具备|拥有|熟悉|掌握|了解|精通).{0,20}$',  # 能力描述而非技能
            r'^(良好|较强|优秀|一定|基本)的',  # 程度修饰而非技能
            r'^(相关|工作|项目|开发|以上)经验$',  # 经验描述而非技能
            r'^(计算机|电子|通信|自动化)相关专业$',  # 专业描述
        ]
        for pattern in hallucination_patterns:
            if re.match(pattern, skill):
                return {'skill': skill, 'status': 'rejected', 'confidence': 0.0,
                        'reason': f'匹配幻觉模式: {pattern}'}

        # 已验证技能库
        if skill in self.verified_skills:
            if skill in AI_SKILLS:
                return {'skill': skill, 'status': 'verified', 'confidence': confidence,
                        'reason': 'AI新兴技能（已验证）'}
            return {'skill': skill, 'status': 'verified', 'confidence': confidence,
                    'reason': '传统技术技能（已验证）'}

        # 过时技能
        if skill in self.outdated_skills:
            reason = self.outdated_skills[skill]
            return {'skill': skill, 'status': 'outdated', 'confidence': confidence,
                    'reason': f'过时技能: {reason}'}

        # 疑似技术术语但未在技能库中 → 候选技能
        if self._is_likely_tech_skill(skill) and confidence > 0.6:
            return {'skill': skill, 'status': 'candidate', 'confidence': confidence,
                    'reason': '疑似新技术术语，待人工确认'}

        # 不在技能库中
        if confidence < 0.4:
            return {'skill': skill, 'status': 'hallucinated', 'confidence': confidence,
                    'reason': '低置信度 + 不在技能库中 → 疑似幻觉'}
        return {'skill': skill, 'status': 'unknown', 'confidence': confidence,
                'reason': '不在技能库中，建议人工审核'}

    def validate_skills(self, skills, confidence_scores=None):
        """批量验证技能列表"""
        if confidence_scores is None:
            confidence_scores = [0.5] * len(skills)
        return [self.validate_skill(s, c) for s, c in zip(skills, confidence_scores)]

    # ── RAG增强验证 ──
    def validate_skill_with_rag(self, skill, confidence=0.5):
        """
        带RAG增强的技能验证：
          白名单验证 → 如果是candidate/unknown → RAG检索 → 最终判定
        """
        # 先用白名单验证
        result = self.validate_skill(skill, confidence)

        # 只有candidate和unknown状态需要RAG
        if result['status'] not in ('candidate', 'unknown'):
            result['rag_verdict'] = 'skipped'
            return result
        if not self.enable_rag or not self.rag:
            result['rag_verdict'] = 'rag_disabled'
            return result

        # RAG检索
        if hasattr(self.rag, 'verify'):
            # 使用全功能RAGRetriever（rag_retriever.py）
            rag_result = self.rag.verify(skill)
            verdict = rag_result['verdict']
            reasoning = rag_result['recommendation']
            evidence = rag_result.get('sources', [])
        else:
            # 使用内置轻量RAGChecker
            verdict, reasoning, evidence = self.rag.verify(skill)
        result['rag_verdict'] = verdict
        result['rag_reasoning'] = reasoning
        result['rag_evidence'] = evidence

        # 融合判定
        if verdict == 'verified_by_rag':
            result['status'] = 'verified'
            result['confidence'] = max(confidence, 0.75)
            result['reason'] = f'RAG验证通过: {reasoning}'
        elif verdict == 'hallucinated':
            result['status'] = 'hallucinated'
            result['reason'] = f'RAG确认幻觉: {reasoning}'
        # needs_human_review → 保持原状态

        return result

    def validate_skills_with_rag(self, skills, confidence_scores=None):
        """批量RAG增强验证"""
        if confidence_scores is None:
            confidence_scores = [0.5] * len(skills)
        return [self.validate_skill_with_rag(s, c) for s, c in zip(skills, confidence_scores)]

    # ── 第2层：LLM输出约束 ──
    def constrain_llm_output(self, generated_skills, allowed_skills):
        """
        约束LLM输出：只保留 allowed_skills 中已有的技能
        generated_skills: LLM生成的技能列表
        allowed_skills: 允许的技能集合（通常来自NER抽取结果）
        """
        constrained = []
        hallucinations = []

        for skill in generated_skills:
            if skill in allowed_skills:
                constrained.append(skill)
            else:
                # 检查是否在已验证技能库中
                if skill in self.verified_skills:
                    constrained.append(skill)  # 虽然不在NER输出中，但在技能库中，允许
                else:
                    hallucinations.append({
                        'skill': skill,
                        'type': 'llm_hallucination',
                        'suggestion': '该技能不在JD抽取结果中，可能是LLM编造',
                    })

        return constrained, hallucinations

    # ── 第3层：本体论验证 ──
    def validate_graph_edge(self, source, target, relation_type, evidence_count=0):
        """
        验证图谱边的合理性
        """
        if relation_type not in self.ontology_rules:
            return {'valid': False, 'reason': f'未知关系类型: {relation_type}'}

        rules = self.ontology_rules[relation_type]

        # 技能→岗位：证据检查
        if relation_type == 'skill_requires_job':
            if evidence_count < rules['min_evidence']:
                return {
                    'valid': False,
                    'reason': f'证据不足：仅{evidence_count}条JD支持（需≥{rules["min_evidence"]}）',
                    'suggestion': '建议标记为低置信度边，等待更多JD验证',
                }
            return {'valid': True, 'reason': '通过'}

        # 技能从属关系
        if relation_type == 'skill_is_subtype_of':
            return self._validate_subtype(source, target)

        # 岗位相似度
        if relation_type == 'job_is_similar_to':
            if hasattr(source, '__len__') and isinstance(source, (int, float)):
                similarity = source
                if similarity < rules['min_similarity']:
                    return {'valid': False, 'reason': f'相似度过低({similarity})，可能不是同类岗位'}
                if similarity > rules['max_similarity']:
                    return {'valid': False, 'reason': f'相似度过高({similarity})，可能是重复岗位'}
                return {'valid': True, 'reason': '通过'}
            return {'valid': True, 'reason': '无法量化验证'}

        return {'valid': True, 'reason': '通过'}

    def _validate_subtype(self, parent, child):
        """验证技能从属关系"""
        # 规则：子技能必须是父技能的技术子集
        # 例如: 'PyTorch' ⊂ '深度学习框架' ✓
        #       '沟通能力' ⊂ 'Python' ✗
        if parent in self.soft_skills or child in self.soft_skills:
            return {'valid': False, 'reason': '软技能不能参与技术从属关系'}

        if parent in self.outdated_skills:
            return {'valid': False, 'reason': f'父技能"{parent}"已过时，不应建立新关系'}

        return {'valid': True, 'reason': '通过（建议人工复审技术从属关系）'}

    def _is_likely_tech_skill(self, skill):
        """启发式判断是否为技术术语"""
        tech_indicators = [
            r'^[A-Za-z0-9+#.]+$',  # 纯英文/数字（如 Docker, K8s, C++）
            r'^[A-Za-z]+[一-鿿]*[A-Za-z]*$',  # 中英混合
            r'.*(框架|系统|平台|工具|引擎|模型|算法|网络|协议|架构|服务|库|语言|数据库).*',
            r'.*(开发|设计|编程|测试|运维|部署|优化|分析|挖掘|训练|推理).*',
        ]
        return any(re.match(p, skill) for p in tech_indicators)

    # ── 报告生成 ──
    def generate_quality_report(self, validated_skills):
        """生成技能质量报告"""
        stats = defaultdict(int)
        for v in validated_skills:
            stats[v['status']] += 1

        hallucination_rate = stats.get('hallucinated', 0) / max(len(validated_skills), 1)

        return {
            'total_skills': len(validated_skills),
            'verified': stats.get('verified', 0),
            'candidate': stats.get('candidate', 0),
            'outdated': stats.get('outdated', 0),
            'unknown': stats.get('unknown', 0),
            'hallucinated': stats.get('hallucinated', 0),
            'rejected': stats.get('rejected', 0),
            'hallucination_rate': round(hallucination_rate, 3),
            'quality_score': round(1.0 - hallucination_rate, 3),
        }


# ── 命令行接口 ──
if __name__ == '__main__':
    guard = HallucinationGuard(enable_rag=True)

    test_skills = [
        # 在字典中 → 直接verified
        ('Python', 0.9),
        ('大模型', 0.95),
        ('沟通能力', 0.3),               # → rejected（软技能）
        ('Theano', 0.7),                # → outdated
        ('ComfyUI', 0.85),              # → candidate（不在白名单）→ RAG验证
        # 不在字典但在RAG知识库 → RAG验证通过
        ('MCP协议', 0.7),               # candidate → RAG → verified
        ('ClaudeCode', 0.65),           # candidate → RAG → verified
        ('思维链推理', 0.6),            # candidate → RAG → verified
        ('Dify工作流', 0.7),            # candidate → RAG → verified
        ('PromptFlow', 0.65),           # candidate → RAG → verified
        # 不在字典也不在RAG知识库 → RAG确认幻觉
        ('超导量子计算', 0.2),          # candidate → RAG → hallucinated
        ('脑机接口编程', 0.15),          # candidate → RAG → hallucinated
    ]

    print('技能验证测试 (含RAG增强):')
    print('=' * 60)
    print(f'{"技能":<30} {"白名单":<12} {"RAG":<18} {"最终":<12}')
    print('-' * 60)

    for skill, conf in test_skills:
        # 无RAG
        r1 = guard.validate_skill(skill, conf)
        # 有RAG
        r2 = guard.validate_skill_with_rag(skill, conf)

        print(f'{skill:<28} {r1["status"]:<12} {r2.get("rag_verdict","?"):<18} {r2["status"]:<12}')

    # 统计
    results_no_rag = guard.validate_skills([s for s, _ in test_skills], [c for _, c in test_skills])
    results_with_rag = guard.validate_skills_with_rag([s for s, _ in test_skills], [c for _, c in test_skills])

    report_no = guard.generate_quality_report(results_no_rag)
    report_rag = guard.generate_quality_report(results_with_rag)

    print(f'\n对比:')
    print(f'  无RAG:  verified={report_no["verified"]}, hallucinated={report_no["hallucinated"]}, candidate={report_no["candidate"]}')
    print(f'  有RAG:  verified={report_rag["verified"]}, hallucinated={report_rag["hallucinated"]}, candidate={report_rag["candidate"]}')
    print(f'  提升:   verified +{report_rag["verified"] - report_no["verified"]}, hallucinated检出 +{report_rag["hallucinated"] - report_no["hallucinated"]}')
