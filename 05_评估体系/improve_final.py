#!/usr/bin/env python3
"""
improve_final.py — 最终改进评估

修复:
  NER: 使用 SkillNER 自定义类加载（同 inference_demo.py）
  通胀: 使用AI技能计数 + 岗位层级感知阈值 + Jaccard抄袭簇增强
  过时: 词边界匹配

Usage:
    python improve_final.py
"""

import os, sys, json, csv, glob, re, math, random
from collections import defaultdict, Counter
from datetime import datetime
from difflib import SequenceMatcher

import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, '【已标注v3】'))
from annotate_v3 import (
    AI_SKILLS, TRADITIONAL_SKILLS, SOFT_SKILLS, OUTDATED_SKILLS,
    extract_skills_from_text,
)

ANNOTATED_DIR = os.path.join(BASE_DIR, 'skill_ner_release', 'data', 'annotated')
NER_MODEL_PATH = os.path.join(BASE_DIR, 'skill_ner_release', 'models', 'best_model.pt')


# ============================================================
#  SkillNER 模型（同 inference_demo.py）
# ============================================================
class SkillNER(nn.Module):
    def __init__(self, model_name='bert-base-chinese', dropout=0.1, num_labels=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))


# ============================================================
#  数据加载
# ============================================================
def load_data(holdout=10):
    files = sorted(glob.glob(os.path.join(ANNOTATED_DIR, '*.csv')))
    random.seed(42); random.shuffle(files)
    test_fs = files[:holdout]; train_fs = files[holdout:]
    def load(fl):
        rows = []
        for fp in fl:
            with open(fp, 'r', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f): rows.append(row)
        return rows
    return load(train_fs), load(test_fs)


# ============================================================
#  通胀检测: AI技能计数 + 岗位层级 + 抄袭簇增强
# ============================================================

# 同义词组（来自标注规范）
SYNONYM_GROUPS = [
    {'大模型', 'LLM', '大语言模型', 'GPT', 'ChatGPT', 'DeepSeek'},
    {'AIGC', '生成式AI', 'GenAI', 'AI绘画', 'AI写作'},
    {'Agent', 'AI Agent', '智能体', 'Multi-Agent', '多智能体'},
    {'Stable Diffusion', 'SDXL', 'Midjourney', 'Sora', '文生图', '文生视频'},
    {'RAG', '检索增强生成', 'Graph RAG'},
    {'LoRA', 'QLoRA', 'PEFT', '微调', '大模型微调', 'SFT', 'RLHF'},
]


def detect_job_level(job_name, jd_text):
    """检测岗位层级: junior/mid/senior"""
    text = job_name + jd_text
    if any(kw in text for kw in ['实习', '初级', '助理', '培训生', '管培生', '应届']):
        return 'junior'
    if any(kw in text for kw in ['高级', '资深', '专家', '架构师', 'Senior', 'Staff', 'Principal', '总监']):
        return 'senior'
    if any(kw in text for kw in ['经理', '主管', '组长', 'Leader', '负责人']):
        return 'senior'
    return 'mid'


def count_synonym_groups(skills_found):
    """统计同义词组命中数（用于检测重复堆砌）"""
    count = 0
    for group in SYNONYM_GROUPS:
        if any(s in group for s in skills_found):
            count += 1
    return count


def evaluate_inflation_final(rows):
    """
    AI技能计数 + 岗位层级感知 + 同义词组 + 抄袭簇
    对标 annotate_v3 的 inflation 判定逻辑
    """
    n = len(rows)
    jd_texts = [str(r.get('skill_requirements', '')) for r in rows]

    # 计算同岗位抄袭簇
    job_idx = defaultdict(list)
    for i, r in enumerate(rows):
        job_idx[str(r.get('job_name', ''))].append(i)

    plagiarism_factor = [1.0] * n
    for job, indices in job_idx.items():
        for a in range(len(indices)):
            for b in range(a+1, len(indices)):
                i, j = indices[a], indices[b]
                sim = SequenceMatcher(None, jd_texts[i], jd_texts[j]).ratio()
                if sim > 0.8:
                    plagiarism_factor[i] += 1
                    plagiarism_factor[j] += 1

    y_true, y_pred = [], []
    detailed = []

    for i, row in enumerate(rows):
        gt = str(row.get('技能通胀', '')).strip()
        y_true.append(0 if gt in ('无', '') else 1)

        text = str(row.get('skill_requirements', ''))
        job_name = str(row.get('job_name', ''))
        level = detect_job_level(job_name, text)

        skills = extract_skills_from_text(text)
        ai_skills = [s for s in skills if s in AI_SKILLS]
        ai_count = len(ai_skills)
        total_skills = len(skills)
        syn_groups = count_synonym_groups(skills.keys() if isinstance(skills, dict) else skills)
        pf = plagiarism_factor[i]

        # 层级感知阈值（对标 annotate_v3 标注规范）
        if level == 'junior':
            thresholds = (3, 5, 8)  # 轻度/中度/重度
        elif level == 'senior':
            thresholds = (6, 10, 15)
        else:
            thresholds = (4, 7, 12)

        # 判定逻辑
        inflation_level = 0  # 0=无, 1=轻, 2=中, 3=重
        if ai_count >= thresholds[2] or (ai_count >= thresholds[1] and syn_groups >= 4):
            inflation_level = 3
        elif ai_count >= thresholds[1] or (ai_count >= thresholds[0] and syn_groups >= 3):
            inflation_level = 2
        elif ai_count >= thresholds[0]:
            inflation_level = 1

        # 抄袭簇增强: 如果JD在抄袭簇中且有AI技能，提升一级
        if pf >= 3 and ai_count >= 2 and inflation_level == 0:
            inflation_level = 1
        if pf >= 5 and inflation_level == 1:
            inflation_level = 2

        pred = 1 if inflation_level >= 1 else 0
        y_pred.append(pred)

        detailed.append({
            'idx': i, 'level': level, 'ai_count': ai_count,
            'total_skills': total_skills, 'syn_groups': syn_groups,
            'plagiarism': pf, 'infl_level': inflation_level,
            'gt': gt, 'pred': pred,
        })

    tp = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t==0 and p==1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t==0 and p==0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==0)

    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    f1 = 2*p*r/max(p+r,0.001)

    # 统计各层级
    level_stats = Counter()
    for d in detailed:
        level_stats[f'{d["level"]}_gt{d["gt"]}_pred{d["pred"]}'] += 1

    return {
        'precision': round(p,4), 'recall': round(r,4), 'f1': round(f1,4),
        'confusion': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
        'total_inflated_gt': sum(y_true),
        'level_stats': dict(level_stats.most_common(12)),
        'detailed': detailed,
    }


# ============================================================
#  过时技能
# ============================================================
EXPANDED_OUTDATED = {
    'JSP': '被前后端分离替代', 'Struts': 'SSH框架已淘汰',
    'Hibernate': '被MyBatis/Spring Data替代', 'Flash': 'HTML5已全面替代',
    'Flex': '富客户端技术已淘汰', 'Silverlight': '微软已停止支持',
    'VB6': '微软已停止支持', 'Visual Basic': '已停止主流开发',
    'Delphi': '被C#/Java替代', 'J2EE': '更名为Jakarta EE',
    'ActiveX': '浏览器已不支持', 'Applet': '浏览器已不支持',
    'WebLogic': '被轻量级容器替代', 'WebSphere': '被轻量级容器替代',
    'EJB': '被Spring替代', 'JSF': '被前后端分离替代',
    'CVS': '被Git替代', 'ClearCase': '被Git替代',
    'Theano': '2017年已停止开发维护', 'Caffe': '被PyTorch/TensorFlow替代',
    'CNTK': '微软2019年已停止维护', 'MXNet': 'Apache已停止维护',
    'Keras': '已被TensorFlow/PyTorch生态替代', 'MFC': '微软已停止更新',
    'Swing': 'Java桌面GUI已淘汰', 'iBatis': '被MyBatis替代',
    'PowerBuilder': '已被.NET/Java替代', 'FoxPro': '微软已停止支持',
    'ActionScript': '随Flash一起被淘汰', 'CodeIgniter': 'PHP框架被替代',
    'Smarty': 'PHP模板引擎被替代', 'ASP': '被ASP.NET替代',
    'Zend Framework': 'PHP框架停止维护', 'ADO.NET': '被Entity Framework替代',
    'WCF': '微软已停止主推', 'Borland C++': '已被替代',
    'Chainer': '2019年停止维护', 'Lasagne': '基于Theano已消亡',
    'Torch': '被PyTorch替代（Lua版）',
}


def match_skill_word(text, skill):
    if not text or not skill: return False
    if re.search(r'[一-鿿]', skill): return skill in text
    return bool(re.search(r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])', text))


def evaluate_outdated_final(rows):
    tp = fp = fn = 0; total_gt = 0; fp_skills = Counter()
    for row in rows:
        text = str(row.get('skill_requirements', ''))
        obs_str = str(row.get('过时技能', '')).strip()
        gt_old = set()
        if obs_str and obs_str != '无':
            content = re.sub(r'^有[（(]', '', obs_str)
            content = re.sub(r'[）)]$', '', content)
            gt_old = set(s.strip() for s in re.split(r'[；;，,]', content) if s.strip())
        if not gt_old:
            for skill in EXPANDED_OUTDATED:
                if match_skill_word(text, skill):
                    fp += 1; fp_skills[skill] += 1
            continue
        total_gt += len(gt_old)
        for gt_s in gt_old:
            if gt_s in EXPANDED_OUTDATED: tp += 1
            else: fn += 1
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return {
        'dict_size': len(EXPANDED_OUTDATED),
        'precision': round(p,4), 'recall': round(r,4),
        'f1': round(2*p*r/max(p+r,0.001),4),
        'tp': tp, 'fp': fp, 'fn': fn, 'total_gt': total_gt,
        'fp_sources': fp_skills.most_common(5),
    }


# ============================================================
#  NER 对比
# ============================================================
def evaluate_ner_final(rows):
    if not os.path.exists(NER_MODEL_PATH):
        return {'status': 'Model not found', 'ner_available': False}

    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
        model = SkillNER()
        ckpt = torch.load(NER_MODEL_PATH, map_location='cpu', weights_only=True)
        state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('loss_fn')}
        model.load_state_dict(state_dict)
        model.eval()

        id2label = {0: 'O', 1: 'B-SKILL-E', 2: 'I-SKILL-E', 3: 'B-SKILL-P', 4: 'I-SKILL-P'}
        print(f'[INFO] SkillNER loaded. Epochs trained: {ckpt.get("epoch", "?")}')

        samples = rows[:100]
        dict_tp = dict_fp = dict_fn = 0
        ner_tp = ner_fp = ner_fn = 0
        ner_empty = 0

        for row in samples:
            text = str(row.get('skill_requirements', ''))[:500]
            if not text or len(text) < 20: continue

            gt_skills = set()
            for col in ['必备技能', '加分技能']:
                for part in str(row.get(col, '')).split('；'):
                    m = re.match(r'【(.+?)[｜|]', part.strip())
                    if m: gt_skills.add(m.group(1).strip())

            # 字典法
            ds = set(extract_skills_from_text(text).keys())
            ds = {s for s in ds if s not in SOFT_SKILLS}
            dict_tp += len(gt_skills & ds)
            dict_fp += len(ds - gt_skills)
            dict_fn += len(gt_skills - ds)

            # NER: 使用同 inference_demo.py 的 extract 逻辑
            enc = tokenizer(text, max_length=256, truncation=True, padding='max_length',
                           return_offsets_mapping=True, return_tensors='pt')
            offsets = enc['offset_mapping'][0]

            with torch.no_grad():
                logits = model(enc['input_ids'], enc['attention_mask'])
                preds = torch.argmax(logits, dim=-1)[0].numpy()

            required, bonus = [], []
            cur, cur_type = [], None
            for k, (start, end) in enumerate(offsets):
                if start == 0 and end == 0: continue
                label = id2label[preds[k]]
                token_text = text[start:end]

                if label.startswith('B-'):
                    if cur and cur_type == 'E': required.append(''.join(cur))
                    elif cur and cur_type == 'P': bonus.append(''.join(cur))
                    cur = [token_text]
                    cur_type = 'E' if label.endswith('-E') else 'P'
                elif label.startswith('I-') and cur:
                    cur.append(token_text)
                else:
                    if cur and cur_type == 'E': required.append(''.join(cur))
                    elif cur and cur_type == 'P': bonus.append(''.join(cur))
                    cur, cur_type = [], None
            if cur and cur_type == 'E': required.append(''.join(cur))
            elif cur and cur_type == 'P': bonus.append(''.join(cur))

            ns = set(required) | set(bonus)
            ns = {s for s in ns if len(s) > 1}
            ns = {s for s in ns if s not in SOFT_SKILLS}

            if not ns: ner_empty += 1

            ner_tp += len(gt_skills & ns)
            ner_fp += len(ns - gt_skills)
            ner_fn += len(gt_skills - ns)

        def calc(tp, fp, fn):
            p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
            return round(p,4), round(r,4), round(2*p*r/max(p+r,0.001),4)

        dp, dr, df1 = calc(dict_tp, dict_fp, dict_fn)
        np_, nr, nf1 = calc(ner_tp, ner_fp, ner_fn)

        return {
            'ner_available': True,
            'dict': {'p': dp, 'r': dr, 'f1': df1, 'tp': dict_tp, 'fp': dict_fp, 'fn': dict_fn},
            'ner': {'p': np_, 'r': nr, 'f1': nf1, 'tp': ner_tp, 'fp': ner_fp, 'fn': ner_fn},
            'ner_empty_outputs': ner_empty,
            'summary': f'Dict F1={df1:.1%} vs NER F1={nf1:.1%}',
            'delta': round(nf1 - df1, 4),
            'samples': len(samples),
        }
    except Exception as e:
        import traceback
        return {'status': f'{e}\n{traceback.format_exc()[:200]}', 'ner_available': False}


# ============================================================
#  主入口
# ============================================================
def main():
    print('=' * 70)
    print('  Improvements - Final Evaluation')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    train_rows, test_rows = load_data(holdout=10)

    report = {'generated_at': datetime.now().isoformat(), 'sections': {}}

    # ========================================
    # [P0] Inflation vFinal
    # ========================================
    print('\n' + '=' * 60)
    print('  [P0] Inflation Detection (AI-skill + level-aware + plagiarism)')
    print('=' * 60)

    # Baseline
    yt, yp_b = [], []
    for row in test_rows:
        text = str(row.get('skill_requirements', ''))
        yt.append(0 if str(row.get('技能通胀', '')).strip() in ('无', '') else 1)
        skills = extract_skills_from_text(text)
        sc = len(skills)
        density = sc / max(len(text), 1) * 1000
        yp_b.append(1 if (density > 14 or sc > 10) else 0)

    tpo = sum(1 for t, p in zip(yt, yp_b) if t==1 and p==1)
    fpo = sum(1 for t, p in zip(yt, yp_b) if t==0 and p==1)
    fno = sum(1 for t, p in zip(yt, yp_b) if t==1 and p==0)
    po = tpo/max(tpo+fpo,1); ro = tpo/max(tpo+fno,1)
    f1o = 2*po*ro/max(po+ro,0.001)

    rf = evaluate_inflation_final(test_rows)
    cm = rf['confusion']

    print(f'\n  Baseline (density>14 or skills>10):')
    print(f'    P={po:.1%} R={ro:.1%} F1={f1o:.1%} | TP={tpo} FP={fpo} FN={fno}')
    print(f'\n  Improved (AI-skill + level-aware + plagiarism):')
    print(f'    P={rf["precision"]:.1%} R={rf["recall"]:.1%} F1={rf["f1"]:.1%} | TP={cm["TP"]} FP={cm["FP"]} FN={cm["FN"]}')
    print(f'    GT inflated: {rf["total_inflated_gt"]}/{len(test_rows)}')
    df1_i = rf['f1'] - f1o
    print(f'  dF1={df1_i:+.1%} dP={rf["precision"]-po:+.1%} dR={rf["recall"]-ro:+.1%}')

    # 各层级表现
    print(f'  Level stats: {rf["level_stats"]}')

    report['sections']['inflation'] = {
        'baseline': {'p': round(po,4), 'r': round(ro,4), 'f1': round(f1o,4)},
        'improved': {k:v for k,v in rf.items() if k != 'detailed'},
        'delta_f1': round(df1_i, 4),
    }

    # ========================================
    # [P1] Outdated
    # ========================================
    print('\n' + '=' * 60)
    print('  [P1] Outdated Skills')
    print('=' * 60)

    # Original
    tp_o2 = fp_o2 = fn_o2 = 0
    for row in test_rows:
        text = str(row.get('skill_requirements', ''))
        obs_str = str(row.get('过时技能', '')).strip()
        gt_old = set()
        if obs_str and obs_str != '无':
            content = re.sub(r'^有[（(]', '', obs_str)
            content = re.sub(r'[）)]$', '', content)
            gt_old = set(s.strip() for s in re.split(r'[；;，,]', content) if s.strip())
        if not gt_old:
            for skill, _ in OUTDATED_SKILLS:
                if match_skill_word(text, skill): fp_o2 += 1
            continue
        for gt_s in gt_old:
            if any(gt_s == s for s,_ in OUTDATED_SKILLS): tp_o2 += 1
            else: fn_o2 += 1
    po2 = tp_o2/max(tp_o2+fp_o2,1); ro2 = tp_o2/max(tp_o2+fn_o2,1)
    f1o2 = 2*po2*ro2/max(po2+ro2,0.001)

    rn = evaluate_outdated_final(test_rows)
    print(f'\n  Original ({len(OUTDATED_SKILLS)}): P={po2:.1%} R={ro2:.1%} F1={f1o2:.1%} | TP={tp_o2} FP={fp_o2} FN={fn_o2}')
    print(f'  Expanded ({rn["dict_size"]}): P={rn["precision"]:.1%} R={rn["recall"]:.1%} F1={rn["f1"]:.1%} | TP={rn["tp"]} FP={rn["fp"]} FN={rn["fn"]}')
    print(f'  FP sources: {rn["fp_sources"]}')
    df1_o2 = rn['f1'] - f1o2
    print(f'  dF1={df1_o2:+.1%}')

    report['sections']['outdated'] = {
        'original': {'p': round(po2,4), 'r': round(ro2,4), 'f1': round(f1o2,4), 'size': len(OUTDATED_SKILLS)},
        'expanded': rn, 'delta_f1': round(df1_o2, 4),
    }

    # ========================================
    # [P1] NER
    # ========================================
    print('\n' + '=' * 60)
    print('  [P1] NER vs Dictionary')
    print('=' * 60)

    ner_r = evaluate_ner_final(test_rows)
    if ner_r.get('ner_available'):
        print(f'\n  Dictionary:  P={ner_r["dict"]["p"]:.1%} R={ner_r["dict"]["r"]:.1%} F1={ner_r["dict"]["f1"]:.1%}')
        print(f'  NER (BERT):  P={ner_r["ner"]["p"]:.1%} R={ner_r["ner"]["r"]:.1%} F1={ner_r["ner"]["f1"]:.1%}')
        print(f'  {ner_r["summary"]}')
        print(f'  NER empty outputs: {ner_r["ner_empty_outputs"]}/{ner_r["samples"]}')
    else:
        print(f'\n  {ner_r.get("status", "skipped")}')
    report['sections']['ner'] = ner_r

    # ========================================
    # Summary
    # ========================================
    print('\n' + '=' * 70)
    print('  Final Summary')
    print('=' * 70)
    items = [
        ('Inflation Detection', f1o, rf['f1']),
        ('Outdated Skills', f1o2, rn['f1']),
    ]
    if ner_r.get('ner_available'):
        items.append(('NER vs Dictionary', ner_r['dict']['f1'], ner_r['ner']['f1']))

    for name, orig, impr in items:
        d = impr - orig
        verdict = 'IMPROVED' if d > 0.01 else ('SAME' if abs(d) <= 0.01 else 'DEGRADED')
        print(f'  {name:30s} {orig:8.1%} -> {impr:8.1%}  {d:+7.1%}  {verdict}')

    out = os.path.join(BASE_DIR, 'improve_final_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n[INFO] Report: {out}')
    print('=' * 70)


if __name__ == '__main__':
    main()
