#!/usr/bin/env python3
"""
filter_no_skill_jds.py — 剔除无技能JD
1. 删除字典完全抽不到技术技能的JD
2. 保留原始目录，输出到 【已标注】filtered/
3. 输出剔除清单 filter_report.csv
"""

import csv, glob, sys, os, shutil
from collections import Counter

BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE, '【已标注v3】'))
from annotate_v3 import extract_skills_from_text, SOFT_SKILLS, AI_SKILLS, TRADITIONAL_SKILLS

SOURCE_DIRS = [
    (os.path.join(BASE, 'skill_ner_release', 'data', 'annotated'), 'annotated'),
    (os.path.join(BASE, '【已标注】jd_v2'), 'jd_v2'),
]
OUT_DIR = os.path.join(BASE, '【已标注】filtered')


def filter_directory(src_dir, label):
    files = sorted(glob.glob(os.path.join(src_dir, '*.csv')))
    dst_dir = os.path.join(OUT_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    total = 0
    kept = 0
    removed = 0
    remove_reasons = Counter()

    for fp in files:
        basename = os.path.basename(fp)
        kept_rows = []
        removed_rows = []

        with open(fp, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                total += 1
                text = str(row.get('skill_requirements', ''))
                job = str(row.get('job_name', ''))
                skills = extract_skills_from_text(text)
                tech_skills = [s for s in skills if s not in SOFT_SKILLS]

                if len(tech_skills) >= 1:
                    kept_rows.append(row)
                    kept += 1
                else:
                    # 记录剔除原因
                    if len(text) < 60:
                        reason = 'JD过短(<60字)'
                    elif len(text) < 100:
                        reason = 'JD偏短(<100字)且无技能'
                    elif any(kw in job + text for kw in ['销售', '客服', '讲师', '培训', '运营', '推广', '营销']):
                        reason = '非技术岗位(销售/客服/讲师/运营)'
                    elif any(kw in job + text for kw in ['产品经理', '产品专员', '产品助理']):
                        reason = '产品岗纯职责描述无技能'
                    elif any(kw in job + text for kw in ['实习', 'Intern', '应届', '管培']):
                        reason = '实习/初级岗要求模糊'
                    else:
                        reason = '其他(纯职责描述/口语化)'
                    remove_reasons[reason] += 1
                    removed_rows.append({k: row.get(k, '') for k in (fieldnames or [])})

        # 写保留的数据
        if kept_rows:
            out_path = os.path.join(dst_dir, basename)
            with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames or [])
                writer.writeheader()
                writer.writerows(kept_rows)

    print(f'[{label}] 总{total} -> 保留{kept} / 剔除{removed} ({removed/total:.1%})')
    for reason, count in remove_reasons.most_common():
        print(f'  - {reason}: {count}')

    return total, kept, removed, remove_reasons


def main():
    print('=' * 60)
    print('  剔除无技能JD')
    print('=' * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_stats = {}
    for src_dir, label in SOURCE_DIRS:
        if not os.path.exists(src_dir):
            print(f'[SKIP] {label}: 目录不存在')
            continue
        print(f'\n处理: {label}')
        all_stats[label] = filter_directory(src_dir, label)

    # 汇总
    print(f'\n{"="*60}')
    print(f'汇总')
    print(f'{"="*60}')
    grand_total = sum(s[0] for s in all_stats.values())
    grand_kept = sum(s[1] for s in all_stats.values())
    grand_removed = sum(s[2] for s in all_stats.values())
    print(f'总计: {grand_total} -> 保留 {grand_kept} / 剔除 {grand_removed} ({grand_removed/grand_total:.1%})')
    print(f'输出目录: {OUT_DIR}')

    # 保存剔除清单
    import json
    report = {
        'filtered_at': __import__('datetime').datetime.now().isoformat(),
        'stats': {label: {'total': s[0], 'kept': s[1], 'removed': s[2],
                 'removal_rate': f'{s[2]/s[0]:.1%}'}
                  for label, s in all_stats.items()},
        'grand_total': grand_total,
        'grand_kept': grand_kept,
        'grand_removed': grand_removed,
    }
    with open(os.path.join(OUT_DIR, 'filter_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'报告: {os.path.join(OUT_DIR, "filter_report.json")}')


if __name__ == '__main__':
    main()
