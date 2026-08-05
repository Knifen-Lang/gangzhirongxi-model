"""
全新标注脚本 v3
修复6类问题：
1. 软技能混入硬技能 → 软技能黑名单
2. 硬性技术漏提取 → 改进必备判定
3. ReAct 误标为AI → React≠ReAct 上下文检测
4. 能力更新 → 传统岗+AI技能标记"新增"
5. 加分技能漏提取 → 改进优先/加分信号
6. 过时技能缺失 → Theano/Caffe/CNTK/MXNet/Keras
7. 技能通胀阈值调整
"""
import pandas as pd
import re
import os
import glob
import argparse
from collections import Counter

# ═══════════════════════════════════════
# 技能词典
# ═══════════════════════════════════════

AI_SKILLS = {
    '大模型', 'LLM', 'GPT', 'ChatGPT', 'DeepSeek', '通义千问', '文心一言',
    '大语言模型', '大模型微调', 'SFT', 'RLHF', 'DPO', '指令微调',
    '预训练', '后训练', '模型训练', '模型推理', '模型部署',
    '模型量化', '模型压缩', '模型蒸馏', 'vLLM',
    'Agent', 'AI Agent', '智能体', 'Multi-Agent', '多智能体', 'Agentic AI',
    'Agentic RAG', 'Agent Loop', 'Plan-and-Execute',
    'Function Calling', 'LangChain', 'LangGraph', 'LlamaIndex', 'CrewAI', 'Dify', 'Coze',
    'RAG', '检索增强生成', 'Graph RAG', '向量检索', '语义检索', 'Rerank',
    '多模态', 'VLM', 'Stable Diffusion', 'SDXL', 'Midjourney', 'Sora',
    '视频生成', '图像生成', '文生图', '文生视频', '视觉大模型', '多模态大模型',
    'Prompt Engineering', '提示工程', '向量数据库', 'Milvus', 'Pinecone',
    'LoRA', 'QLoRA', 'PEFT', '微调',
    'AIGC', '生成式AI', 'GenAI', 'AI绘画', 'AI写作', 'AI编程',
    'AI Coding', 'Copilot', 'CodeAgent', 'Deep Research',
    '具身智能', '自动驾驶', '数字孪生', '数字人',
    '联邦学习', '隐私计算', 'AI安全', '模型安全',
    'Context Engine', 'Harness Engineering', 'Memory Engine',
}

TRADITIONAL_SKILLS = {
    # 编程语言
    'Python', 'Java', 'C++', 'C语言', 'C#', 'Go', 'Golang', 'Rust',
    'JavaScript', 'TypeScript', 'Node.js', 'PHP', 'Ruby', 'Scala', 'Kotlin',
    'Swift', 'Objective-C', 'Shell', 'Bash', 'Perl', 'Lua', 'Dart',
    # 前端
    'HTML', 'CSS', 'HTML5', 'CSS3', 'React', 'Vue.js', 'Vue', 'Angular',
    'jQuery', 'Bootstrap', 'Webpack', 'Vite', 'Flutter', 'React Native',
    'UniApp', '微信小程序', 'Electron', '鸿蒙', 'HarmonyOS',
    # 后端
    'Spring Boot', 'Spring Cloud', 'Spring MVC', 'Spring', 'MyBatis',
    'Hibernate', 'Struts', 'Django', 'Flask', 'FastAPI', 'Tornado',
    'Express', 'Koa', 'NestJS', 'Gin', 'Beego', 'Laravel',
    '.NET', 'ASP.NET', 'ASP.NET Core',
    # 数据库
    'MySQL', 'PostgreSQL', 'Oracle', 'SQL Server', 'MongoDB', 'Redis',
    'Elasticsearch', 'Cassandra', 'HBase', 'Neo4j', 'TiDB', 'ClickHouse',
    'Doris', 'SQLite', 'DynamoDB', 'Hive',
    # DevOps/云
    'Docker', 'Kubernetes', 'K8s', 'Jenkins', 'GitLab CI', 'GitHub Actions',
    'Terraform', 'Ansible', 'Prometheus', 'Grafana', 'ELK', 'Nginx',
    'AWS', 'Azure', '阿里云', '腾讯云', '华为云', 'GCP', '云原生',
    '微服务', 'Serverless',
    # 大数据
    'Spark', 'Flink', 'Hadoop', 'Kafka', 'RabbitMQ', 'RocketMQ', 'Airflow',
    'ETL', '数据仓库', '数据湖', '数据治理', '数据建模',
    'SQL', 'Hive SQL', 'Spark SQL',
    # AI/ML（传统）
    'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'XGBoost', 'LightGBM',
    'Pandas', 'NumPy', 'Matplotlib', 'OpenCV', 'YOLO',
    'CNN', 'RNN', 'LSTM', 'Transformer', 'GAN',
    '自然语言处理', 'NLP', '计算机视觉', 'CV', '语音识别',
    '机器学习', '深度学习', '强化学习', '迁移学习', '图神经网络', 'GNN',
    '知识图谱', '数据挖掘', '数据分析', '特征工程', '模型评估',
    'A/B测试', '因果推断', 'AB实验',
    # 操作系统/硬件
    'Linux', 'Unix', 'RTOS', '嵌入式', 'ARM', 'FPGA',
    'Verilog', 'VHDL', 'PCB', '驱动开发', 'BSP', '单片机', 'STM32', 'DSP',
    # 通信/网络
    'TCP/IP', 'HTTP', 'WebSocket', 'gRPC', 'MQTT', '5G', 'LTE',
    '射频', '天线', '光通信', '核心网', 'SDN', 'NFV',
    # 软件工程
    'Git', 'SVN', 'JIRA', 'Confluence', 'Scrum', '敏捷开发',
    'RESTful', 'API', 'GraphQL', '分布式', '高并发', '高可用',
    '多线程', '设计模式', '数据结构', '系统设计',
    '性能优化', '网络安全', '加密', 'OAuth', 'JWT', '区块链', 'Web3',
    # 测试
    '自动化测试', 'Selenium', 'JMeter', 'LoadRunner', '性能测试',
    '单元测试', '集成测试', '回归测试', '测试用例',
    # 设计/工具
    'Axure', 'Figma', 'Sketch', 'Adobe XD', 'Photoshop', 'Illustrator',
    '产品设计', '原型设计', 'PRD', 'MRD',
    '项目管理', '需求分析', '技术文档',
    # 硬件
    'PCB设计', '模拟电路', '数字电路', '电路设计',
    'SLAM', 'ROS', 'ROS2', '自动驾驶系统',
    # 爬虫
    '爬虫', 'Scrapy', 'Selenium',
}

# ═══════════════════════════════════════
# 软技能黑名单（不能作为必备技能）
# ═══════════════════════════════════════

SOFT_SKILLS = {
    '沟通能力', '沟通协调', '沟通', '表达',
    '团队合作', '团队协作', '团队管理', '团队建设', '跨部门协作', '跨部门',
    '领导力', '管理能力', '组织协调', '执行能力',
    '学习能力', '自学能力', '自驱力', '责任心', '抗压能力',
    '运营', '用户运营', '活动运营', '内容运营', '产品运营', '用户增长',
    '用户体验', 'UX', '用户研究', '用户访谈', '可用性测试',
    '商务谈判', '客户关系', 'CRM', '销售', '市场营销',
    '产品策略', '产品规划', '竞品分析', '市场分析', '行业分析',
    '创新思维', '逻辑思维', '分析能力', '解决问题', '问题解决',
    '英语', '英语能力', '英语四级', '英语六级', '外语',
    '文字功底', '文档能力', '方案撰写', 'PPT',
    '细心', '耐心', '严谨', '细致',
}

# ═══════════════════════════════════════
# 过时技能（扩充版）
# ═══════════════════════════════════════

OUTDATED_SKILLS = [
    ('JSP', '被前后端分离替代'),
    ('Struts', 'SSH框架已淘汰'),
    ('Hibernate', '被MyBatis/Spring Data替代'),
    ('Flash', 'HTML5已全面替代'),
    ('Flex', '富客户端技术已淘汰'),
    ('Silverlight', '微软已停止支持'),
    ('VB6', '微软已停止支持'),
    ('Visual Basic', '已停止主流开发'),
    ('Delphi', '被C#/Java替代'),
    ('J2EE', '更名为Jakarta EE'),
    ('ActiveX', '浏览器已不支持'),
    ('Applet', '浏览器已不支持'),
    ('WebLogic', '被轻量级容器替代'),
    ('WebSphere', '被轻量级容器替代'),
    ('EJB', '被Spring替代'),
    ('JSF', '被前后端分离替代'),
    ('CVS', '被Git替代'),
    ('ClearCase', '被Git替代'),
    # 新增过时AI框架
    ('Theano', '2017年已停止开发维护'),
    ('Caffe', '被PyTorch/TensorFlow替代，已停止更新'),
    ('CNTK', '微软2019年已停止维护'),
    ('MXNet', 'Apache已停止维护，社区不活跃'),
    ('Keras', '作为独立框架已被TensorFlow/PyTorch生态替代'),
]

# ═══════════════════════════════════════
# 新兴岗位判定
# ═══════════════════════════════════════

CORE_AI_SIGNALS = {
    '大模型': 3, 'LLM': 3, '大语言模型': 3,
    'AIGC': 3, '生成式AI': 3, 'GenAI': 3,
    'Agent': 2, '智能体': 2, 'Agentic AI': 3, 'Multi-Agent': 3,
    'RAG': 3, '检索增强生成': 3,
    '大模型微调': 3, 'SFT': 2, 'RLHF': 2, '模型训练': 2,
    '多模态大模型': 3, '视觉大模型': 3,
    'Prompt Engineering': 2, '提示工程': 2,
    'AI产品经理': 3, 'AI训练师': 3,
    'LangChain': 2, 'LangGraph': 2, 'LlamaIndex': 2,
    'Stable Diffusion': 2, 'Sora': 2, 'Midjourney': 2,
    '数字孪生': 2, '数字人': 2,
}

TRADITIONAL_TECH_BASE = {
    '鸿蒙': 3, 'HarmonyOS': 3,
    'Android': 3, 'iOS': 3,
    'Spring Boot': 2, 'Spring Cloud': 2, 'Django': 2, 'Flask': 2,
    'SLAM': 3, 'ROS': 3, 'ROS2': 3,
    '单片机': 3, 'STM32': 3, '嵌入式': 3, 'FPGA': 3,
    '驱动开发': 3, 'BSP': 3,
    '射频': 3, '天线': 3, '光通信': 3, '核心网': 3,
    '爬虫': 2, 'Scrapy': 2,
}

# ═══════════════════════════════════════
# 技能提取
# ═══════════════════════════════════════

def extract_skills_from_text(text):
    if pd.isna(text) or not text:
        return {}

    found = {}
    text_lower = text.lower()
    all_skills = sorted(AI_SKILLS | TRADITIONAL_SKILLS, key=len, reverse=True)
    matched_positions = set()

    for skill in all_skills:
        search = skill.lower() if skill.isascii() else skill
        text_search = text_lower if skill.isascii() else text

        for m in re.finditer(re.escape(search), text_search):
            pos = m.start()
            if any(ps <= pos < pe for ps, pe in matched_positions):
                continue
            end_pos = m.end()
            ctx_start = max(0, pos - 150)
            ctx_end = min(len(text), end_pos + 150)
            context = text[ctx_start:ctx_end]
            found[skill] = context
            matched_positions.add((pos, end_pos))
            break
    # ReAct 不再单独检测 — 与 React(前端框架) 在中文JD中无法可靠区分
    # 涉及 ReAct 的AI岗位会通过 Agent/智能体/RAG 等信号被正确识别

    return found


def classify_skill(skill, context=''):
    """分类技能"""
    if skill in AI_SKILLS:
        return 'AI新兴技能'
    return '传统技术'


def get_mastery_level(context):
    """从上下文中判断掌握级别"""
    if re.search(r'精通|深入理解|专家|资深|深度|多年|扎实|丰富.{0,5}经验', context):
        return '精通'
    elif re.search(r'熟练|掌握|独立.{0,10}(完成|开发|负责|设计)', context):
        return '熟练'
    elif re.search(r'熟悉|具备|有.{0,10}经验|相关经验|理解.{0,10}原理', context):
        return '熟悉'
    elif re.search(r'了解|知晓|知道|接触过|愿意学习|感兴趣|关注', context):
        return '了解'
    return '熟悉'


def is_bonus_signal(context):
    """判断上下文是否为加分信号"""
    # 先排除非技能的"优先"描述
    non_skill_bonus = [
        r'博士优先', r'硕士优先', r'985优先', r'211优先',
        r'双一流优先', r'学历优先', r'年龄.*优先',
        r'有户口', r'本地.*优先',
    ]
    for pat in non_skill_bonus:
        if re.search(pat, context):
            return False  # 不是技能加分，是学历/身份加分

    bonus_patterns = [
        r'优先', r'加分', r'者优先', r'更佳', r'尤佳',
        r'优先考虑', r'加分项',
        r'了解.{0,10}更佳', r'熟悉.{0,10}更佳',
        r'有.{0,10}经验者优先', r'具有.{0,10}经验优先',
    ]
    for pat in bonus_patterns:
        if re.search(pat, context):
            return True
    return False


def is_soft_skill(skill):
    """判断是否为软技能"""
    return skill in SOFT_SKILLS


# ═══════════════════════════════════════
# 新兴岗位判定
# ═══════════════════════════════════════

def is_new_job(text, job_name=''):
    if pd.isna(text) or not text or len(text) < 20:
        return '否'

    # 排除信号
    exclusion_score = 0
    for tech, weight in TRADITIONAL_TECH_BASE.items():
        search = tech.lower() if tech.isascii() else tech
        target = text.lower() if tech.isascii() else text
        if search in target:
            exclusion_score += weight

    # 核心AI信号
    core_score = 0
    for sig, weight in CORE_AI_SIGNALS.items():
        search = sig.lower() if sig.isascii() else sig
        target = text.lower() if sig.isascii() else text
        if search in target:
            core_score += weight

    # 硬排除
    hard_exclude = ['鸿蒙', 'HarmonyOS', 'Android', 'iOS']
    excluded = False
    for h in hard_exclude:
        search = h.lower() if h.isascii() else h
        target = text.lower() if h.isascii() else text
        if search in target:
            excluded = True
            break
    if excluded:
        return '否'

    # SLAM/嵌入式/FPGA + AI弱 → 否
    if exclusion_score >= 3 and core_score < 4:
        return '否'

    # 传统后端 + AI弱 → 否
    if exclusion_score >= 2 and core_score < 3:
        return '否'

    # 核心AI ≥ 6 → 是（阈值从4提高到6，避免单个AIGC术语误判）
    if core_score >= 6:
        return '是'

    # 核心AI ≥ 4 且 排除分<3 → 需进一步验证AI职责占比
    if core_score >= 4 and exclusion_score < 3:
        # 额外检查：AI相关句子占比必须>50%
        sentences = [s.strip() for s in re.split(r'[。；;.\n]', text) if len(s.strip()) > 10]
        if sentences:
            ai_sentences = 0
            for sent in sentences:
                for sig in CORE_AI_SIGNALS:
                    search = sig.lower() if sig.isascii() else sig
                    target = sent.lower() if sig.isascii() else sent
                    if search in target:
                        ai_sentences += 1
                        break
            if ai_sentences / len(sentences) >= 0.5:
                return '是'
        return '否'

    return '否'


# ═══════════════════════════════════════
# 能力更新
# ═══════════════════════════════════════

def check_ability_update(text, job_name=''):
    """检测传统岗位是否新增AI技能要求"""
    if pd.isna(text) or not text:
        return '无'

    update_parts = []

    # 检测AI新兴技能要求
    ai_found = []
    for sig in CORE_AI_SIGNALS:
        search = sig.lower() if sig.isascii() else sig
        target = text.lower() if sig.isascii() else text
        if search in target:
            ai_found.append(sig)

    # 传统岗位关键词（扩大范围）
    traditional_jobs = [
        '产品经理', '前端', '后端', '测试', '运维', '爬虫',
        'Java', 'Python', 'C++', 'Android', 'iOS', '嵌入式',
        '项目经理', '数据分析', '硬件', '网络', '通信',
        '运营', '设计', '销售', '客服', '技术支持', '实施',
        '架构师', 'DBA', '系统管理员', '安全', '质量',
        '电子', '电气', '自动化', '机械', '结构',
    ]

    # 判定：岗位名命中传统关键词，或非纯AI岗位（不含大模型/LLM/Agent/AIGC在岗位名中）
    is_traditional = (
        any(t in job_name for t in traditional_jobs) or
        not any(sig in job_name for sig in ['大模型', 'LLM', 'Agent', 'AIGC', '算法', '深度学习', 'NLP', '自然语言', '计算机视觉'])
    )

    # 过时技能
    outdated_found = []
    for skill, reason in OUTDATED_SKILLS:
        search = skill.lower() if skill.isascii() else skill
        target = text.lower() if skill.isascii() else text
        if search in target:
            outdated_found.append(skill)

    # 降低阈值：≥1个AI信号即标记（原为≥2，导致单AI信号的传统岗位漏标）
    if is_traditional and len(ai_found) >= 1:
        update_parts.append(f'新增AI技能要求（{";".join(ai_found[:5])}）')

    if outdated_found:
        update_parts.append(f'删除过时技能（{";".join(outdated_found)}）')

    if update_parts:
        return '；'.join(update_parts)
    return '无'


# ═══════════════════════════════════════
# 主标注函数
# ═══════════════════════════════════════

def _llm_promote(jd_text, job_name, candidates):
    """
    LLM判断：JD全是"优先/加分"语气时，哪些其实应该是必备。

    candidates: [(技能名, 级别), ...]
    返回: [应提升的技能名, ...]  或  []（全部保留加分）
    """
    # 只有1个候选 → 直接提升，不用调LLM
    if len(candidates) <= 1:
        return [c[0] for c in candidates]

    # 构建prompt
    cand_str = '、'.join([f'{name}({level})' for name, level in candidates])
    prompt = (
        f'岗位名称：{job_name}\n'
        f'JD原文：{jd_text[:500]}\n\n'
        f'以下技能在JD中都是以"优先""加分""更佳"等可选语气提到的：{cand_str}\n\n'
        f'请根据岗位名称和JD上下文，判断其中哪些技能实际上是该岗位的硬性要求（必备），哪些只是锦上添花（加分）。\n'
        f'注意：JD写的是可选语气不代表不需要——比如后端岗写"Python优先"，Python大概率是必备。\n\n'
        f'只输出必备技能名，用逗号分隔。如果全部都是加分，输出"无"。'
    )

    # 先试LLM（如果可用），不可用则用启发式规则
    llm_output = _try_llm(prompt)
    if llm_output:
        skills = [s.strip() for s in llm_output.split(',') if s.strip() and s.strip() != '无']
        # RAG验证：每项必须在JD原文或技能库中找到证据
        verified = []
        for skill in skills:
            skill_in_jd = skill.lower() in jd_text.lower()
            skill_in_taxonomy = skill in (AI_SKILLS | TRADITIONAL_SKILLS)
            if skill_in_jd or skill_in_taxonomy:
                verified.append(skill)
        return verified if verified else None  # 返回None表示LLM不可靠

    return None  # LLM不可用


def _try_llm(prompt):
    """尝试调用LLM，不可用时返回None"""
    try:
        # 检查是否有可用的LLM API配置
        coze_token = os.environ.get('COZE_TOKEN', '')
        if coze_token:
            # TODO: 接入Coze API
            pass
        # 当前返回None，使用降级策略（按级别取）
        return None
    except Exception:
        return None


def annotate_row(row):
    text = str(row.get('skill_requirements', ''))
    job_name = str(row.get('job_name', ''))

    if len(text) < 20:
        return pd.Series({
            '必备技能': '（JD文本过短）', '加分技能': '（JD文本过短）',
            '特征技能': '（未识别）', '技能通胀': '无', '过时技能': '无',
            '是否新岗位候选': '否', '能力更新': '无',
        })

    skills_found = extract_skills_from_text(text)

    # 分类：必备 vs 加分（软技能排除）
    required = []
    bonus = []

    for skill, ctx in skills_found.items():
        # 软技能不能作为必备
        if is_soft_skill(skill):
            bonus.append((skill, '传统技术', '熟悉'))
            continue

        cat = classify_skill(skill, ctx)
        level = get_mastery_level(ctx)

        # "了解"级别有加分信号 → 加分；裸"了解"没有加分信号 → 仍是必备（只是要求低）
        if level == '了解' and is_bonus_signal(ctx):
            bonus.append((skill, cat, level))
        elif is_bonus_signal(ctx):
            bonus.append((skill, cat, level))
        else:
            required.append((skill, cat, level))

    # 排序：按掌握级别优先，同级别内AI新兴技能排前
    # 修复：不再将传统技术整体排在AI技能后面，避免Python/Java等被挤出
    level_order = {'精通': 0, '熟练': 1, '熟悉': 2, '了解': 3}
    required.sort(key=lambda x: (level_order.get(x[2], 2), 0 if x[1] == 'AI新兴技能' else 1))
    bonus.sort(key=lambda x: (level_order.get(x[2], 2), 0 if x[1] == 'AI新兴技能' else 1))

    # 限制数量：必备1-6, 加分0-4
    required = required[:6]
    bonus = bonus[:4]

    # 必备为空时，尝试LLM+RAG推断（优于纯按级别猜测）
    promoted_from_bonus = False
    if len(required) < 1 and bonus:
        # 构建LLM输入：JD原文 + 候选项（加分列表）
        candidates = [(s[0], s[2]) for s in bonus]
        llm_result = _llm_promote(text, job_name, candidates)

        if llm_result:
            promoted_names = set(llm_result)
            promoted = [s for s in bonus if s[0] in promoted_names]
            rest = [s for s in bonus if s[0] not in promoted_names]
            required = promoted[:3]
            bonus = rest
        else:
            # LLM不可用 → 降级为按级别取
            best_level = bonus[0][2]
            promoted = [s for s in bonus if s[2] == best_level][:3]
            rest = [s for s in bonus if s not in promoted]
            required = promoted
            bonus = rest
        promoted_from_bonus = True

    def fmt_skills(skills):
        return '；'.join([f'【{s[0]}｜{s[1]}｜{s[2]}】' for s in skills])

    required_str = fmt_skills(required) if required else '（JD中未识别到明确的必备硬性技能）'
    if promoted_from_bonus:
        required_str += '（注：以下技能从加分项推断，JD未明确列为硬性要求）'
    bonus_str = fmt_skills(bonus) if bonus else '（未识别到明确的加分技能）'

    # 特征技能
    features = [s for s in required if s[1] == 'AI新兴技能'][:2]
    if len(features) < 2:
        features += [s for s in required if s[1] == '传统技术'][:2 - len(features)]
    if not features and bonus:
        features = [bonus[0]]
    feature_str = '、'.join([f'【{s[0]}】' for s in features[:3]]) if features else '（未识别）'

    # ── 技能通胀（调整阈值）──
    # 统计AI名词密度（去软技能）
    ai_count = sum(1 for s in skills_found if classify_skill(s, skills_found.get(s, '')) == 'AI新兴技能')
    total = len(skills_found)

    if ai_count <= 2:
        inflation = '无'
    elif ai_count <= 4:
        inflation = '轻度'
    elif ai_count <= 8:
        inflation = '中度'
    else:
        inflation = '重度'

    # ── 过时技能（扩充版）──
    outdated_found = []
    for skill, reason in OUTDATED_SKILLS:
        search = skill.lower() if skill.isascii() else skill
        target = text.lower() if skill.isascii() else text
        if search in target:
            outdated_found.append(skill)
    outdated_str = '无' if not outdated_found else f'有（{"、".join(outdated_found)}）'

    # ── 新岗位候选 ──
    new_job = is_new_job(text, job_name)

    # ── 能力更新 ──
    ability_update = check_ability_update(text, job_name)

    return pd.Series({
        '必备技能': required_str,
        '加分技能': bonus_str,
        '特征技能': feature_str,
        '技能通胀': inflation,
        '过时技能': outdated_str,
        '是否新岗位候选': new_job,
        '能力更新': ability_update,
    })


# ═══════════════════════════════════════
# 批量处理
# ═══════════════════════════════════════

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    for col in ['必备技能', '加分技能', '特征技能', '技能通胀', '过时技能', '是否新岗位候选', '能力更新']:
        df[col] = ''

    for idx in range(len(df)):
        annotations = annotate_row(df.iloc[idx])
        for col in annotations.index:
            df.at[idx, col] = annotations[col]

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--original_dir', type=str, default=None)
    args = parser.parse_args()

    # 如果指定了原始目录，从那里读 + 写入输出目录
    if args.original_dir:
        input_dir = args.original_dir
        output_dir = args.output_dir
        csv_files = sorted(glob.glob(os.path.join(input_dir, 'zhilian_direct_*.csv')))
    else:
        input_dir = args.input_dir
        output_dir = args.output_dir
        csv_files = sorted(glob.glob(os.path.join(input_dir, 'zhilian_direct_*.csv')))

    print(f'找到 {len(csv_files)} 个文件')
    total = 0

    for i, f in enumerate(csv_files):
        fname = os.path.basename(f)
        out_name = f'【已标注】{fname}'
        out_path = os.path.join(output_dir, out_name)

        rows = process_file(f, out_path)
        total += rows

        if (i + 1) % 15 == 0 or i == len(csv_files) - 1:
            print(f'  [{i+1}/{len(csv_files)}] {out_name} ({rows}条) — 累计 {total} 条')

    # 统计
    print(f'\n=== 标注统计 ===')
    yes_count = 0
    no_count = 0
    outdated_count = 0
    ability_update_count = 0

    for f in sorted(glob.glob(os.path.join(output_dir, '【已标注】zhilian_direct_*.csv'))):
        df = pd.read_csv(f)
        yes_count += (df['是否新岗位候选'] == '是').sum()
        no_count += (df['是否新岗位候选'] == '否').sum()
        outdated_count += (df['过时技能'] != '无').sum()
        ability_update_count += (df['能力更新'] != '无').sum()

    print(f'新兴岗位: 是={yes_count}, 否={no_count}')
    print(f'有过时技能标记: {outdated_count} 条')
    print(f'有能力更新标记: {ability_update_count} 条')
    print(f'\n全部完成: {len(csv_files)} 文件, {total} 条')


if __name__ == '__main__':
    main()
