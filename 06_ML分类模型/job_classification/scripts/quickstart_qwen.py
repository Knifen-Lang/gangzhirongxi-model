"""
Qwen2.5 微调 — 快速启动脚本

一键完成: 依赖检查 → 模型下载 → 快速测试 → 完整训练

用法:
  # 1. 快速测试 (1 fold, 3 epochs, 确保一切正常)
  python scripts/quickstart_qwen.py --test

  # 2. 完整训练 (5 fold, 所有 epochs)
  python scripts/quickstart_qwen.py

  # 3. 仅推理
  python scripts/quickstart_qwen.py --infer --resume ./outputs/qwen_finetune_xxx/fold_0/best_model
"""

import argparse
import os
import subprocess
import sys


def check_dependencies():
    """检查必需的包是否已安装"""
    required = {
        'torch': 'torch',
        'transformers': 'transformers>=4.40',
        'peft': 'peft>=0.10',
        'accelerate': 'accelerate',
        'bitsandbytes': 'bitsandbytes>=0.43',
        'scikit-learn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
    }

    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
            print(f'  ✓ {pip_name}')
        except ImportError:
            print(f'  ✗ {pip_name} — 需要安装')
            missing.append(pip_name.split('>=')[0])

    return missing


def install_missing(packages):
    """安装缺失的包"""
    for pkg in packages:
        print(f'安装 {pkg}...')
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', pkg, '-i',
             'https://pypi.tuna.tsinghua.edu.cn/simple'],
            stdout=subprocess.DEVNULL,
        )
    print('安装完成!')


def test_model_loading():
    """快速测试模型加载"""
    print('\n' + '=' * 60)
    print('测试模型加载...')
    print('=' * 60)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_name = 'Qwen/Qwen2.5-7B-Instruct'

    # 测试 tokenizer
    print(f'加载 tokenizer: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'  ✓ tokenizer 加载成功, vocab_size={len(tokenizer)}')

    # 测试 4-bit 模型加载
    print(f'加载模型 (4-bit QLoRA)...')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f'  ✓ 模型加载成功')

    # 测试前向
    test_text = '请判断以下候选人的岗位: Python, Django, 3年经验'
    inputs = tokenizer(
        test_text, return_tensors='pt', max_length=128,
        padding='max_length', truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()},
                       output_hidden_states=True)
    print(f'  ✓ 前向传播成功, hidden_size={outputs.hidden_states[-1].shape}')

    # 测试 LoRA
    from peft import get_peft_model, LoraConfig, TaskType
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  ✓ LoRA 注入成功: 可训练 {trainable/1e6:.1f}M / {total/1e6:.1f}M '
          f'({100*trainable/total:.1f}%)')

    print('\n✓ 所有测试通过! 可以开始训练.\n')
    del model
    torch.cuda.empty_cache()


def run_training(args):
    """启动训练"""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'train_qwen.py'),
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--n_folds', str(args.n_folds),
        '--epochs_stage1', str(args.epochs_stage1),
        '--epochs_stage2', str(args.epochs_stage2),
        '--batch_size', str(args.batch_size),
        '--grad_accum', str(args.grad_accum),
        '--max_length', str(args.max_length),
        '--lora_r', str(args.lora_r),
        '--lora_alpha', str(args.lora_alpha),
        '--encoding_format', args.encoding_format,
        '--random_seed', str(args.random_seed),
    ]

    if args.model_name:
        cmd += ['--model_name', args.model_name]

    print(f'\n执行训练命令:\n  {" ".join(cmd)}\n')
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Qwen2.5 微调 — 快速启动'
    )

    parser.add_argument('--test', action='store_true',
                        help='仅测试模型加载，不训练')
    parser.add_argument('--install', action='store_true',
                        help='安装缺失的依赖')
    parser.add_argument('--skip_check', action='store_true',
                        help='跳过依赖检查')
    parser.add_argument('--infer', action='store_true',
                        help='仅推理模式')
    parser.add_argument('--resume', type=str, default='',
                        help='从已有模型恢复')

    # 训练参数 (覆盖默认值)
    parser.add_argument('--model_name', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_dir', type=str,
                        default='../zhilian_direct')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs_stage1', type=int, default=6)
    parser.add_argument('--epochs_stage2', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--encoding_format', type=str, default='B')
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    # ── 依赖检查 ──
    if not args.skip_check:
        print('=' * 60)
        print('检查依赖...')
        print('=' * 60)
        missing = check_dependencies()
        if missing:
            if args.install:
                install_missing(missing)
            else:
                print(f'\n缺失 {len(missing)} 个包: {missing}')
                print('运行 --install 自动安装, 或手动安装:')
                print(f'  pip install {" ".join(missing)}')
                return

    # ── 模型加载测试 ──
    if args.test:
        test_model_loading()
        return

    # ── 推理模式 ──
    if args.infer:
        if not args.resume:
            print('推理模式需要指定 --resume <模型路径>')
            return
        print(f'从 {args.resume} 加载模型进行推理...')
        # TODO: 单独推理脚本
        return

    # ── 训练模式 ──
    print('\n' + '=' * 60)
    print('Qwen2.5-7B + LoRA 微调')
    print('=' * 60)
    print(f'  模型: {args.model_name}')
    print(f'  数据: {args.data_dir}')
    print(f'  Folds: {args.n_folds}')
    print(f'  Stage1 epochs: {args.epochs_stage1}')
    print(f'  Stage2 epochs: {args.epochs_stage2}')
    print(f'  Batch size: {args.batch_size} × {args.grad_accum} '
          f'(effective: {args.batch_size * args.grad_accum})')
    print(f'  LoRA rank: {args.lora_r}')
    print(f'  Max length: {args.max_length}')
    print(f'  编码格式: {args.encoding_format}')
    print(f'  显存: 8GB (Qwen2.5-7B 4-bit QLoRA)')
    print('=' * 60 + '\n')

    run_training(args)


if __name__ == '__main__':
    main()
