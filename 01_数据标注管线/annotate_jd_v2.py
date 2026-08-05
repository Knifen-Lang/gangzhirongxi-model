"""
标注脚本 v2 — 适配 jd 目录
- 第1步：数据清洗（剔除非新一代信息技术行业）
- 第2步：按6项标注规范标注（修复阈值、新岗位判定、加分提取等）
- 第3步：五级抽取管线兜底（字典→语义推断→极简跳过→职责推断→协同过滤）
"""
import pandas as pd
import re
import os
import sys
import glob
import argparse
from collections import Counter

# 引入五级抽取管线（字典失效时的兜底）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
try:
    from llm_skill_extractor import LLMSkillExtractor
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    LLMSkillExtractor = None

# 全局管线实例（懒加载）
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None and _PIPELINE_AVAILABLE:
        _pipeline = LLMSkillExtractor(llm_mode='local')
    return _pipeline

# ═══════════════════════════════════════
# 第0步：数据清洗 — 新一代信息技术行业判定
# ═══════════════════════════════════════

# 新一代信息技术强信号（命中≥1个即认为是IT行业）
IT_STRONG_SIGNALS = [
    # 编程/开发
    '编程语言', '软件开发', '软件工程', '代码', '开发工程师', '程序员',
    'Java', 'Python', 'C++', 'Golang', 'Rust', 'JavaScript', 'TypeScript',
    'React', 'Vue', 'Spring', 'Django', 'Flask', 'Node.js',
    # AI
    '人工智能', '机器学习', '深度学习', '自然语言处理', '计算机视觉', 'NLP', 'CV',
    '大模型', 'LLM', 'AIGC', 'Agent', 'RAG', '大语言模型', '神经网络',
    'PyTorch', 'TensorFlow', 'Paddle', 'MindSpore',
    # 数据
    '数据分析', '数据挖掘', '数据工程', '数据科学', '大数据', 'SQL', '数据库',
    '数据仓库', 'ETL', 'Hadoop', 'Spark', 'Flink', 'Kafka',
    # 云/基础设施
    '云计算', '云原生', 'Kubernetes', 'Docker', 'DevOps', 'Serverless',
    '阿里云', '腾讯云', '华为云', 'AWS', 'Azure',
    # 安全
    '网络安全', '信息安全', '渗透测试', '安全审计', '加密算法',
    # 硬件/嵌入式（新一代信息技术包含智能硬件）
    '嵌入式', '单片机', 'STM32', 'FPGA', '芯片设计', '驱动开发', 'BSP',
    'ARM', 'RTOS', 'Verilog', 'VHDL',
    # 通信
    '5G', 'LTE', '射频', '天线', '光通信', '核心网', 'SDN', 'NFV',
    # IoT
    '物联网', 'IoT', '传感器', '智能硬件', '智能传感',
    # 数字孪生/区块链
    '区块链', '数字孪生', 'Web3', '智能合约',
    # 鸿蒙/移动
    '鸿蒙', 'HarmonyOS', 'Android', 'iOS', 'Flutter', 'React Native',
    # 产品/设计/项目（IT领域）
    '产品经理', '产品设计', 'PRD', 'Axure', 'Figma', 'Sketch',
    'UI设计', 'UX设计', '交互设计', '原型设计',
    '需求分析', '技术文档', '项目经理',
    # 测试/运维
    '自动化测试', '性能测试', '测试用例', 'Selenium', 'JMeter',
    '运维', 'SRE', 'CI/CD', 'Jenkins', 'GitLab',
    # 架构
    '架构设计', '系统设计', '分布式', '微服务', '高并发',
    # 其他IT
    '爬虫', '自动化', 'ROS', 'SLAM', '规控', '自动驾驶',
    '语音识别', '图像识别', '知识图谱', '强化学习',
    '量化', '数据标注', '数据治理', '数据建模',
    '通信协议', 'TCP/IP', 'HTTP', 'WebSocket', 'gRPC',
    'Linux', 'Unix', 'Git', 'API', 'RESTful',
]

# 明确非新一代信息技术（排除）
NON_IT_KEYWORDS = [
    # 传统制造业
    '流水线', '装配工', '操作工', '普工', '焊工', '电工', '车工', '铣工',
    '数控机床', 'CNC', '注塑', '模具', '钣金', '钳工',
    # 物流/运输
    '司机', '货运', '配送', '快递员', '物流搬运', '叉车', '装卸',
    # 建筑/土木
    '工程造价', '土木工程', '施工员', '监理', '测绘', '建筑设计',
    '装修', '装潢', '给排水', '暖通',
    # 金融/保险（非科技岗）
    '保险销售', '保险顾问', '理财顾问', '贷款专员', '信用卡销售',
    # 传统零售/餐饮
    '店员', '导购', '收银', '服务员', '厨师', '洗碗',
    # 医疗（非IT）
    '护士', '临床医学', '口腔医生', '中医', '药剂师',
    # 教育（非IT培训）
    '学科教师', '幼师', '保育员',
    # 美容/健身
    '美容师', '美发师', '健身教练', '按摩师',
    # 物业/安保
    '物业管理员', '保安', '保洁', '绿化',
    # 农业/能源（非IT）
    '种植', '养殖', '畜牧', '石油', '煤炭', '矿产',
    # 风电/光伏运维（非IT）
    '风电', '光伏电站', '风机', '风电场', '风力发电',
    # 硬件安装/维修（非IT）
    '安装工', '维修员', '设备维修', '电源维修', '焊接',
    'LED照明', '电子维修',
    # 锂电/医疗设备/登机桥等非IT设备
    '锂电设备', '医疗设备售后', '登机桥',
    # 工业传感器销售/工艺（非AI传感器）
    '传感器销售', '传感器工艺', '温度传感器', '压力传感器',
]

# 特定排除的岗位名模式（非IT技术支持/销售/行政）
NON_IT_JOB_PATTERNS = [
    r'^光伏', r'^储能', r'^风电', r'^新能源(?!.*(?:软件|算法|数据|AI))',
    r'^太阳能', r'^电池',
    r'^(?:电话|网络|在线)客服$', r'^呼叫中心',
    r'^文员$', r'^前台$', r'^行政(?!.*(?:IT|技术|信息))',
    r'^会计$', r'^出纳$', r'^财务(?!.*(?:IT|系统|软件|数据))',
    r'^HR$', r'^人事(?!.*(?:IT|系统|信息))', r'^招聘(?!.*(?:IT|技术|科技))',
    r'^法务', r'^律师',
    r'^市场(?:推广|营销|拓展)(?!.*(?:技术|数据|数字|算法|AI))',
    r'^品牌(?:策划|推广)(?!.*(?:数字|技术))',
    # 非IT操作/标注/采集岗（宽泛匹配 + JD有IT信号时救回）
    r'装调', r'硬件装调', r'安装调试', r'装配测试',
    r'物联网安装调试', r'智能硬件装调',
    r'普工', r'操作工', r'装配工',
]

# ═══════════════════════════════════════
# 文件级类别排除 — 整类不属于新一代信息技术
# ═══════════════════════════════════════

NON_IT_FILE_CATEGORIES = {
    # 技工/操作工（非IT）
    '智能硬件装调员',
    '物联网安装调试员',
    # 文件名含"云网运维"实为风电/光伏/地铁设备维护，非云计算IT
    '云网智能运维员',
}


def is_it_industry(job_name, jd_text, file_category=''):
    """
    判断JD是否属于新一代信息技术行业。
    返回: (is_it: bool, reason: str)
    """
    full_text = job_name + ' ' + jd_text

    # Step 0: 文件级类别排除 — 整类非IT，不提供救回路径
    if file_category in NON_IT_FILE_CATEGORIES:
        return False, f'非IT文件类别: {file_category}'

    # Step 1: 明确排除 — 非IT关键词
    for kw in NON_IT_KEYWORDS:
        if kw in full_text:
            return False, f'非IT关键词: {kw}'

    # Step 2: 明确排除 — 非IT岗位模式
    for pat in NON_IT_JOB_PATTERNS:
        if re.search(pat, job_name):
            # 二次校验：如果JD文本有强IT信号则保留
            it_signal_count = sum(1 for s in IT_STRONG_SIGNALS if s.lower() in full_text.lower())
            if it_signal_count == 0:
                return False, f'非IT岗位模式: {pat}'

    # Step 3: 必须有新一代信息技术强信号
    it_signal_count = sum(1 for s in IT_STRONG_SIGNALS if s.lower() in full_text.lower())

    if it_signal_count == 0:
        # 检查岗位名本身是否IT相关
        it_job_title_kw = ['工程师', '架构师', '程序员', '开发', '算法', 'AI', '数据',
                           '产品经理', '测试', '运维', '前端', '后端', '安全', '网络',
                           '嵌入式', '芯片', '通信', '电子', '硬件']
        if not any(kw in job_name for kw in it_job_title_kw):
            return False, '无IT信号且岗位名无IT关键词'

    return True, '通过'


# ═══════════════════════════════════════
# 技能词典（同 v3）
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
    'Python', 'Java', 'C++', 'C语言', 'C#', 'Go', 'Golang', 'Rust',
    'JavaScript', 'TypeScript', 'Node.js', 'PHP', 'Ruby', 'Scala', 'Kotlin',
    'Swift', 'Objective-C', 'Shell', 'Bash', 'Perl', 'Lua', 'Dart',
    'HTML', 'CSS', 'HTML5', 'CSS3', 'React', 'Vue.js', 'Vue', 'Angular',
    'jQuery', 'Bootstrap', 'Webpack', 'Vite', 'Flutter', 'React Native',
    'UniApp', '微信小程序', 'Electron', '鸿蒙', 'HarmonyOS',
    'Spring Boot', 'Spring Cloud', 'Spring MVC', 'Spring', 'MyBatis',
    'Hibernate', 'Struts', 'Django', 'Flask', 'FastAPI', 'Tornado',
    'Express', 'Koa', 'NestJS', 'Gin', 'Beego', 'Laravel',
    '.NET', 'ASP.NET', 'ASP.NET Core',
    'MySQL', 'PostgreSQL', 'Oracle', 'SQL Server', 'MongoDB', 'Redis',
    'Elasticsearch', 'Cassandra', 'HBase', 'Neo4j', 'TiDB', 'ClickHouse',
    'Doris', 'SQLite', 'DynamoDB', 'Hive',
    'Docker', 'Kubernetes', 'K8s', 'Jenkins', 'GitLab CI', 'GitHub Actions',
    'Terraform', 'Ansible', 'Prometheus', 'Grafana', 'ELK', 'Nginx',
    'AWS', 'Azure', '阿里云', '腾讯云', '华为云', 'GCP', '云原生',
    '微服务', 'Serverless',
    'Spark', 'Flink', 'Hadoop', 'Kafka', 'RabbitMQ', 'RocketMQ', 'Airflow',
    'ETL', '数据仓库', '数据湖', '数据治理', '数据建模',
    'SQL', 'Hive SQL', 'Spark SQL',
    'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'XGBoost', 'LightGBM',
    'Pandas', 'NumPy', 'Matplotlib', 'OpenCV', 'YOLO',
    'CNN', 'RNN', 'LSTM', 'Transformer', 'GAN',
    '自然语言处理', 'NLP', '计算机视觉', 'CV', '语音识别',
    '机器学习', '深度学习', '强化学习', '迁移学习', '图神经网络', 'GNN',
    '知识图谱', '数据挖掘', '数据分析', '特征工程', '模型评估',
    'A/B测试', '因果推断', 'AB实验',
    'Linux', 'Unix', 'RTOS', '嵌入式', 'ARM', 'FPGA',
    'Verilog', 'VHDL', 'PCB', '驱动开发', 'BSP', '单片机', 'STM32', 'DSP',
    'TCP/IP', 'HTTP', 'WebSocket', 'gRPC', 'MQTT', '5G', 'LTE',
    '射频', '天线', '光通信', '核心网', 'SDN', 'NFV',
    'Git', 'SVN', 'JIRA', 'Confluence', 'Scrum', '敏捷开发',
    'RESTful', 'API', 'GraphQL', '分布式', '高并发', '高可用',
    '多线程', '设计模式', '数据结构', '系统设计',
    '性能优化', '网络安全', '加密', 'OAuth', 'JWT', '区块链', 'Web3',
    '自动化测试', 'Selenium', 'JMeter', 'LoadRunner', '性能测试',
    '单元测试', '集成测试', '回归测试', '测试用例',
    'Axure', 'Figma', 'Sketch', 'Adobe XD', 'Photoshop', 'Illustrator',
    '产品设计', '原型设计', 'PRD', 'MRD',
    '项目管理', '需求分析', '技术文档',
    'PCB设计', '模拟电路', '数字电路', '电路设计',
    'SLAM', 'ROS', 'ROS2', '自动驾驶系统',
    '爬虫', 'Scrapy',
    # 通信（传输/无线）
    'PTN', 'SPN', 'OTN', 'MIMO', '波束赋形', '光传输', 'ODN', 'PON',
    # 传感器/自动驾驶
    '激光雷达', '毫米波雷达', '超声波雷达', 'IMU', 'GNSS', '惯性导航',
    '多传感器融合', '感知融合', '点云', 'SLAM', 'VIO',
    # 芯片/半导体
    'MEMS', '版图设计', '数字IC', '模拟IC', '数字前端', '数字后端',
    '射频芯片', 'SoC', 'ASIC', 'FPGA原型验证',
}

# ═══════════════════════════════════════
# 软技能黑名单
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
# 过时技能
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
    ('Theano', '2017年已停止开发维护'),
    ('Caffe', '被PyTorch/TensorFlow替代'),
    ('CNTK', '微软2019年已停止维护'),
    ('MXNet', 'Apache已停止维护'),
    ('Keras', '已被TensorFlow/PyTorch生态替代'),
]

# ═══════════════════════════════════════
# 新岗位候选 — 核心AI信号
# ═══════════════════════════════════════

# 只有这些信号能触发「新岗位候选」判定
NEW_JOB_AI_SIGNALS = [
    '大模型', 'LLM', '大语言模型', 'GPT', 'ChatGPT', 'DeepSeek', '文心一言', '通义千问',
    'AIGC', '生成式AI', 'GenAI',
    'Agent', 'AI Agent', '智能体', 'Multi-Agent', '多智能体',
    'RAG', '检索增强生成', 'Graph RAG',
    '大模型微调', 'SFT', 'RLHF', 'DPO',
    '多模态大模型', '视觉大模型', 'VLM',
    'LangChain', 'LangGraph', 'LlamaIndex', 'Dify', 'Coze',
    'Stable Diffusion', 'SDXL', 'Midjourney', 'Sora',
    'LoRA', 'QLoRA', 'PEFT',
    'Prompt Engineering', '提示工程',
    'AI Agent', 'Agentic AI', 'Agentic RAG',
    'AI Coding', 'Copilot', 'CodeAgent',
    '数字孪生', '数字人', '具身智能',
    '联邦学习', '隐私计算',
]

# 传统硬技术岗位名关键词（用于判断能力更新）
TRADITIONAL_JOB_KEYWORDS = [
    '产品经理', '前端', '后端', '测试', '运维', '爬虫',
    'Java', 'Python', 'C++', 'Android', 'iOS', '嵌入式',
    '项目经理', '数据分析', '硬件', '网络', '通信',
    '运营', '设计', '销售', '客服', '技术支持', '实施',
    '架构师', 'DBA', '系统管理员', '安全', '质量',
    '电子', '电气', '自动化', '机械', '结构',
    '数据开发', '数据仓库', '数据治理', '数据采集', '数据标注',
    '系统集成', '需求分析', '技术文档',
]


# ═══════════════════════════════════════
# 辅助函数
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
    return found


def classify_skill(skill, context=''):
    if skill in AI_SKILLS:
        return 'AI新兴技能'
    return '传统技术'


def get_mastery_level(context):
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
    non_skill_bonus = [
        r'博士优先', r'硕士优先', r'985优先', r'211优先',
        r'双一流优先', r'学历优先', r'年龄.*优先',
        r'有户口', r'本地.*优先',
    ]
    for pat in non_skill_bonus:
        if re.search(pat, context):
            return False
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
    return skill in SOFT_SKILLS


# 新增：硬性专业技能（方法论/工具/管理）— 可纳入必备，不算软技能
HARD_PROFESSIONAL_SKILLS = {
    '需求分析', '项目管理', 'PRD', 'MRD', '技术文档',
    'Axure', 'Figma', 'Sketch', 'Adobe XD', 'Photoshop', 'Illustrator',
    '产品设计', '原型设计',
    'JIRA', 'Confluence', 'Scrum', '敏捷开发',
    'Git', 'SVN',
}


def get_job_level(job_name, jd_text):
    """判断岗位层级：初级/中级/高级"""
    full = job_name + ' ' + jd_text[:500]
    if re.search(r'初级|助理|实习|应届|校招|管培生|培训生|无经验|1年以下|入门', full):
        return '初级'
    if re.search(r'高级|资深|专家|首席|总监|VP|CTO|负责人|主管|经理(?!.*产品经理)', full):
        return '高级'
    return '中级'


# ═══════════════════════════════════════
# 主标注函数
# ═══════════════════════════════════════

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
    job_level = get_job_level(job_name, text)
    extraction_tier = 1  # 默认字典命中

    # ── 五级管线兜底：字典未命中时逐级尝试 ──
    if not skills_found:
        pipeline = get_pipeline()
        if pipeline:
            pip_result = pipeline.extract(text, job_name)
            if pip_result and pip_result.get('skills'):
                # 将管线结果转换为 skills_found 格式 {skill: context}
                for s in pip_result['skills']:
                    skill_name = s['skill']
                    # 避免与字典重复（管线可能输出已在字典中的技能）
                    if skill_name not in skills_found:
                        skills_found[skill_name] = f"[Tier{pip_result.get('tier', '?')}] {s.get('category', '')}"
                extraction_tier = pip_result.get('tier', 2)
            elif pip_result and pip_result.get('essential_str') and '未识别' not in str(pip_result.get('essential_str', '')):
                # 管线有格式化输出但 skills 为空——尝试从字符串解析
                pass

    # ── 必备 vs 加分分类 ──
    required = []
    bonus = []

    for skill, ctx in skills_found.items():
        # 软技能不入必备
        if is_soft_skill(skill):
            bonus.append((skill, '传统技术', '熟悉'))
            continue

        cat = classify_skill(skill, ctx)
        level = get_mastery_level(ctx)

        if level == '了解' and is_bonus_signal(ctx):
            bonus.append((skill, cat, level))
        elif is_bonus_signal(ctx):
            bonus.append((skill, cat, level))
        else:
            required.append((skill, cat, level))

    # 排序：级别优先，同级别AI新兴排前
    level_order = {'精通': 0, '熟练': 1, '熟悉': 2, '了解': 3}
    required.sort(key=lambda x: (level_order.get(x[2], 2), 0 if x[1] == 'AI新兴技能' else 1))
    bonus.sort(key=lambda x: (level_order.get(x[2], 2), 0 if x[1] == 'AI新兴技能' else 1))

    required = required[:6]
    bonus = bonus[:4]

    # 必备为空时，从加分提升技能到必备
    # JD全篇用"优先"语气时，所有技能归加分。此时间量放宽：尽可能填必备（最多6），剩余留加分
    promoted_from_bonus = False
    if len(required) < 1 and bonus:
        # 优先提AI新兴技能，再提传统技术；同类型按级别排序
        bonus_ai = [s for s in bonus if s[1] == 'AI新兴技能']
        bonus_trad = [s for s in bonus if s[1] == '传统技术']
        promoted = (bonus_ai + bonus_trad)[:6]  # 最多提6个到必备
        rest = [s for s in bonus if s not in promoted][:4]  # 剩余最多4个留加分
        required = promoted
        bonus = rest
        promoted_from_bonus = True

    def fmt_skills(skills):
        return '；'.join([f'【{s[0]}｜{s[1]}｜{s[2]}】' for s in skills])

    required_str = fmt_skills(required) if required else '（JD中未识别到明确的必备硬性技能）'
    if promoted_from_bonus:
        required_str += '（注：以下技能从加分项推断，JD未明确列为硬性要求）'
    bonus_str = fmt_skills(bonus) if bonus else '（未识别到明确的加分技能）'

    # ── 特征技能 ──
    features = [s for s in required if s[1] == 'AI新兴技能'][:2]
    if len(features) < 2:
        features += [s for s in required if s[1] == '传统技术'][:2 - len(features)]
    if not features and bonus:
        features = [bonus[0]]
    feature_str = '、'.join([f'【{s[0]}】' for s in features[:3]]) if features else '（未识别）'

    # ── 技能通胀（岗位层级感知阈值）──
    ai_count = sum(1 for s in skills_found if classify_skill(s, skills_found.get(s, '')) == 'AI新兴技能')
    # 检测同义词重复堆砌
    ai_names = [s for s in skills_found if classify_skill(s, skills_found.get(s, '')) == 'AI新兴技能']
    synonym_groups = [
        {'大模型', 'LLM', '大语言模型', 'GPT', 'ChatGPT', 'DeepSeek'},
        {'AIGC', '生成式AI', 'GenAI', 'AI绘画', 'AI写作'},
        {'Agent', 'AI Agent', '智能体', 'Multi-Agent', '多智能体'},
        {'Stable Diffusion', 'SDXL', 'Midjourney', 'Sora', '文生图', '文生视频', '图像生成', '视频生成'},
        {'RAG', '检索增强生成', 'Graph RAG'},
        {'LoRA', 'QLoRA', 'PEFT', '微调', '大模型微调', 'SFT', 'RLHF'},
    ]
    synonym_bonus = 0
    for group in synonym_groups:
        hits = group & set(ai_names)
        if len(hits) >= 3:
            synonym_bonus += len(hits) - 2

    # 层级阈值
    if job_level == '初级':
        light_thresh, medium_thresh, heavy_thresh = 3, 5, 8
    elif job_level == '高级':
        light_thresh, medium_thresh, heavy_thresh = 6, 10, 15
    else:  # 中级
        light_thresh, medium_thresh, heavy_thresh = 4, 7, 12

    adjusted_ai = ai_count + synonym_bonus

    if adjusted_ai <= 2:
        inflation = '无'
    elif adjusted_ai < light_thresh:
        inflation = '无'
    elif adjusted_ai < medium_thresh:
        inflation = '轻度'
    elif adjusted_ai < heavy_thresh:
        inflation = '中度'
    else:
        inflation = '重度'

    # ── 过时技能 ──
    outdated_found = []
    for skill, reason in OUTDATED_SKILLS:
        search = skill.lower() if skill.isascii() else skill
        target = text.lower() if skill.isascii() else text
        if search in target:
            outdated_found.append(skill)
    outdated_str = '无' if not outdated_found else f'有（{"、".join(outdated_found)}）'

    # ── 是否新岗位候选（按规范重写）──
    new_job = is_new_job_v2(text, job_name, skills_found, required, bonus)

    # ── 能力更新 ──
    ability_update = check_ability_update_v2(text, job_name)

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
# 新岗位候选 v2（两条件同时满足，按标注规范）
# ═══════════════════════════════════════

def is_new_job_v2(text, job_name, skills_found, required, bonus):
    """
    按标注规范：
    条件①：岗位核心工作依托LLM/AIGC/RAG/Agent/多模态大模型，AI为主业(≥50%)
    条件②：企业长期稳定招聘AI业务
    两项同时满足 → '是'。仅加分项提及AI → '否'。
    岗位名含AI/大模型/LLM/Agent/AIGC → 优先进入校验流程。
    """
    if pd.isna(text) or not text or len(text) < 20:
        return '否'

    # Step 1: 检测JD中的新兴AI信号（检测全文，不依赖截断后的必备列表）
    ai_signals_in_text = []
    for sig in NEW_JOB_AI_SIGNALS:
        search = sig.lower() if sig.isascii() else sig
        target = text.lower() if sig.isascii() else text
        if search in target:
            ai_signals_in_text.append(sig)

    if not ai_signals_in_text:
        return '否'

    # Step 2: AI信号是否只出现在加分语境中
    # 但如果岗位名含核心AI词（大模型/AIGC/Agent/LLM），跳过此检查
    # ——岗位名已说明核心工作依托AI，"了解+优先"可能是JD措辞习惯
    title_core_ai = any(kw in job_name for kw in ['大模型', 'LLM', 'AIGC', 'Agent', '智能体',
                                                     '多模态大模型', '视觉大模型'])

    if not title_core_ai:
        ai_in_required_context = False
        ai_in_bonus_context = False
        for sig in ai_signals_in_text:
            if sig in skills_found:
                ctx = skills_found[sig]
                if is_bonus_signal(ctx):
                    ai_in_bonus_context = True
                else:
                    ai_in_required_context = True

        # 只出现在加分中 → 否
        if ai_in_bonus_context and not ai_in_required_context:
            return '否'
    else:
        ai_in_required_context = True  # 岗位名含核心AI，视为AI在必备语境
        ai_in_bonus_context = False

    # Step 3: 岗位名校验（含AI关键词 → 降低内容门槛）
    title_ai_keywords = ['AI', 'AIGC', '大模型', 'LLM', 'Agent', '智能体', '多模态',
                         '深度学习', '机器学习', 'NLP', '自然语言', '计算机视觉', '算法']
    title_has_ai = any(kw in job_name for kw in title_ai_keywords)

    # 岗位名直接含核心新兴AI关键词，条件①的置信度最高
    title_has_core_ai = any(kw in job_name for kw in ['大模型', 'LLM', 'AIGC', 'Agent', '智能体'])

    # Step 4: 句子级AI占比
    sentences = [s.strip() for s in re.split(r'[。；;.\n]', text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [text]

    ai_sentences = 0
    for sent in sentences:
        for sig in NEW_JOB_AI_SIGNALS:
            search = sig.lower() if sig.isascii() else sig
            target = sent.lower() if sig.isascii() else sent
            if search in target:
                ai_sentences += 1
                break
    ai_ratio = ai_sentences / len(sentences)

    # Step 5: 综合判定
    # ┌─────────────────────────────────────────────────┐
    # │ 条件①：核心工作依托新兴AI (LLM/AIGC/RAG/Agent/多模态) │
    # │ 条件②：企业长期稳定招聘（通过JD质量和AI信号强度推断）   │
    # └─────────────────────────────────────────────────┘

    # 条件①的四种满足方式（满足任一条即视为条件①通过）
    condition1_met = False

    # 1a: 岗位名含核心AI词（大模型/AIGC/Agent/LLM） — 最强信号
    if title_has_core_ai:
        condition1_met = True

    # 1b: 岗位名含算法/AI + ≥2个不同AI信号
    if title_has_ai and len(ai_signals_in_text) >= 2:
        condition1_met = True

    # 1c: AI信号在必备语境中出现 + 句子占比≥30%
    if ai_in_required_context and ai_ratio >= 0.30:
        condition1_met = True

    # 1d: ≥4个不同AI信号（JD深度涉及AI，即使岗位名不含AI）
    if len(ai_signals_in_text) >= 4:
        condition1_met = True

    if not condition1_met:
        return '否'

    # 条件②：企业长期稳定招聘（JD完整度检验：非短JD，有职责+要求）
    has_responsibility = bool(re.search(r'(岗位)?(职责|内容|描述|工作)|工作(内容|职责)|职位描述|工作(任务|范围)', text))
    has_requirements = bool(re.search(r'(任职)?(要求|资格|条件)|技术(要求|栈)|技能要求|岗位要求|录用条件', text))
    jd_length_ok = len(text) >= 60

    # 三项至少满足两项（岗位名含核心AI时降低条件②门槛）
    condition2_score = sum([has_responsibility, has_requirements, jd_length_ok])
    if title_has_core_ai:
        condition2_met = condition2_score >= 1  # 核心AI岗位放宽
    else:
        condition2_met = condition2_score >= 2

    if condition1_met and condition2_met:
        return '是'
    return '否'


# ═══════════════════════════════════════
# 能力更新 v2（按规范，必须写具体技能名）
# ═══════════════════════════════════════

def check_ability_update_v2(text, job_name=''):
    if pd.isna(text) or not text:
        return '无'

    update_parts = []

    # 检测AI新兴技能
    ai_found = []
    for sig in NEW_JOB_AI_SIGNALS:
        search = sig.lower() if sig.isascii() else sig
        target = text.lower() if sig.isascii() else text
        if search in target:
            ai_found.append(sig)

    # 判定是否传统岗位
    is_traditional = (
        any(t in job_name for t in TRADITIONAL_JOB_KEYWORDS) or
        not any(sig in job_name for sig in ['大模型', 'LLM', 'Agent', 'AIGC', '算法', '深度学习',
                                              'NLP', '自然语言', '计算机视觉', '人工智能'])
    )

    # 过时技能
    outdated_found = []
    for skill, reason in OUTDATED_SKILLS:
        search = skill.lower() if skill.isascii() else skill
        target = text.lower() if skill.isascii() else text
        if search in target:
            outdated_found.append(skill)

    if is_traditional and len(ai_found) >= 1:
        update_parts.append(f'新增AI技能要求（{"、".join(ai_found[:5])}）')

    if outdated_found:
        update_parts.append(f'删除过时技能（{"、".join(outdated_found)}）')

    if update_parts:
        return '；'.join(update_parts)
    return '无'


# ═══════════════════════════════════════
# 批量处理
# ═══════════════════════════════════════

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)

    # 从文件名提取岗位类别
    fname = os.path.basename(input_path)
    # zhilian_direct_AIGC算法.csv → AIGC算法, liepin_Java.csv → Java
    file_category = fname.replace('zhilian_direct_', '').replace('liepin_', '').replace('.csv', '')

    # 添加清洗标记列
    df['_it_industry'] = ''
    df['_clean_reason'] = ''

    for idx in range(len(df)):
        jn = str(df.at[idx, 'job_name'])
        jd_text = str(df.at[idx, 'skill_requirements'])
        is_it, reason = is_it_industry(jn, jd_text, file_category)
        df.at[idx, '_it_industry'] = is_it
        df.at[idx, '_clean_reason'] = reason

    # 统计
    cleaned_out = (df['_it_industry'] == False).sum()
    it_df = df[df['_it_industry']].copy()

    # 标注列初始化
    for col in ['必备技能', '加分技能', '特征技能', '技能通胀', '过时技能', '是否新岗位候选', '能力更新']:
        it_df[col] = ''

    for idx in it_df.index:
        annotations = annotate_row(it_df.loc[idx])
        for col in annotations.index:
            it_df.at[idx, col] = annotations[col]

    # 移除临时列
    it_df = it_df.drop(columns=['_it_industry', '_clean_reason'])

    it_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return len(df), len(it_df), cleaned_out


def main():
    parser = argparse.ArgumentParser(description='标注 jd 目录 v2（含清洗）')
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'zhilian', 'liepin'])
    parser.add_argument('--input_dir', type=str,
                        default=r'C:\Users\33247\Desktop\小挑\jd')
    parser.add_argument('--output_dir', type=str,
                        default=r'C:\Users\33247\Desktop\小挑\【已标注】jd_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_raw = 0
    total_cleaned = 0
    total_removed = 0

    for source_label, dir_path, file_pattern in [
        ('zhilian_direct', r'zhilian_direct (3)\zhilian_direct', 'zhilian_direct_*.csv'),
        ('liepin', r'liepin_data\liepin_data', 'liepin_*.csv'),
    ]:
        if args.source != 'all' and args.source not in source_label:
            continue

        full_dir = os.path.join(args.input_dir, dir_path)
        if not os.path.isdir(full_dir):
            print(f'[WARN] 目录不存在: {full_dir}')
            continue

        csv_files = sorted(glob.glob(os.path.join(full_dir, file_pattern)))
        if not csv_files:
            print(f'[WARN] 未找到 {file_pattern} 文件')
            continue

        print(f'\n=== {source_label} ({len(csv_files)} 个文件) ===')
        for i, f in enumerate(csv_files):
            fname = os.path.basename(f)
            out_name = f'【已标注】{fname}'
            out_path = os.path.join(args.output_dir, out_name)
            raw_n, cleaned_n, removed_n = process_file(f, out_path)
            total_raw += raw_n
            total_cleaned += cleaned_n
            total_removed += removed_n

            if (i + 1) % 15 == 0 or i == len(csv_files) - 1:
                print(f'  [{i+1}/{len(csv_files)}] {out_name}: {raw_n}→{cleaned_n}条 (剔除{removed_n}) — 累计清洗后{total_cleaned}条')

    # ── 统计 ──
    print(f'\n{"="*60}')
    print(f'=== 清洗统计 ===')
    print(f'原始总数: {total_raw}')
    print(f'清洗后: {total_cleaned} (+{total_removed} 非IT已剔除)')
    print(f'剔除率: {total_removed/total_raw*100:.1f}%')

    # 标注统计
    yes_count = 0
    no_count = 0
    outdated_count = 0
    ability_update_count = 0
    inflation_counts = Counter()
    required_empty = 0

    for f in sorted(glob.glob(os.path.join(args.output_dir, '【已标注】*.csv'))):
        df = pd.read_csv(f)
        yes_count += (df['是否新岗位候选'] == '是').sum()
        no_count += (df['是否新岗位候选'] == '否').sum()
        outdated_count += (df['过时技能'] != '无').sum()
        ability_update_count += (df['能力更新'] != '无').sum()
        for v in df['技能通胀'].dropna():
            inflation_counts[v] += 1
        required_empty += df['必备技能'].str.contains('未识别', na=False).sum()

    total_annotated = yes_count + no_count
    print(f'\n=== 标注统计 ({total_annotated}条) ===')
    print(f'新岗位候选: 是={yes_count} ({yes_count/total_annotated*100:.1f}%), 否={no_count} ({no_count/total_annotated*100:.1f}%)')
    print(f'有过时技能标记: {outdated_count} 条 ({outdated_count/total_annotated*100:.1f}%)')
    print(f'有能力更新标记: {ability_update_count} 条 ({ability_update_count/total_annotated*100:.1f}%)')
    print(f'技能通胀分布: {dict(inflation_counts)}')
    print(f'必备技能为空: {required_empty} 条 ({required_empty/total_annotated*100:.1f}%)')
    print(f'\n输出目录: {args.output_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
