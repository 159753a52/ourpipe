"""
PyTorch Profiler JSON 分析脚本

比较有/无 Orion 调度器的 profiler 结果，分析调度效果。

使用方法:
    python analyze_profiler.py --baseline ./profiler_baseline --orion ./profiler_orion
    python analyze_profiler.py --file ./profiler_baseline/rank0/xxx.pt.trace.json
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import glob


def load_trace_json(filepath: str) -> dict:
    """加载 PyTorch profiler 的 trace JSON 文件"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_events(trace_data: dict) -> List[dict]:
    """从 trace 数据中提取事件列表"""
    if 'traceEvents' in trace_data:
        return trace_data['traceEvents']
    return []


def categorize_events(events: List[dict]) -> Dict[str, List[dict]]:
    """将事件按类别分组"""
    categories = defaultdict(list)
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        cat = event.get('cat', 'unknown')
        name = event.get('name', 'unknown')
        
        # 按类别分组
        categories[cat].append(event)
        
        # 特别标记 micro-batch 相关事件
        if 'mb_' in name or 'micro' in name.lower():
            categories['micro_batch'].append(event)
        
        # 标记前向/反向传播事件
        if 'Fwd' in name or 'fwd' in name:
            categories['forward'].append(event)
        if 'Bwd' in name or 'bwd' in name:
            categories['backward'].append(event)
    
    return categories


def analyze_micro_batch_timing(events: List[dict]) -> Dict[str, dict]:
    """分析每个 micro-batch 的时间"""
    mb_events = defaultdict(list)
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        name = event.get('name', '')
        
        # 查找 micro-batch 相关事件
        for mb_idx in range(4):  # 假设 4 个 micro-batch
            if f'mb_{mb_idx}' in name:
                mb_events[f'mb_{mb_idx}'].append(event)
                break
    
    # 计算每个 micro-batch 的统计信息
    mb_stats = {}
    for mb_name, events_list in mb_events.items():
        durations = []
        for event in events_list:
            dur = event.get('dur', 0)
            if dur > 0:
                durations.append(dur)
        
        if durations:
            mb_stats[mb_name] = {
                'count': len(durations),
                'total_us': sum(durations),
                'avg_us': sum(durations) / len(durations),
                'min_us': min(durations),
                'max_us': max(durations),
            }
    
    return mb_stats


def analyze_kernel_events(events: List[dict]) -> Dict[str, dict]:
    """分析 CUDA kernel 事件"""
    kernel_events = []
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        cat = event.get('cat', '')
        # CUDA kernel 事件通常在 'kernel' 或 'cuda_runtime' 类别
        if 'kernel' in cat.lower() or 'cuda' in cat.lower():
            kernel_events.append(event)
    
    # 按 kernel 名称分组
    kernel_stats = defaultdict(lambda: {'count': 0, 'total_us': 0, 'durations': []})
    
    for event in kernel_events:
        name = event.get('name', 'unknown')
        dur = event.get('dur', 0)
        
        kernel_stats[name]['count'] += 1
        kernel_stats[name]['total_us'] += dur
        if dur > 0:
            kernel_stats[name]['durations'].append(dur)
    
    # 计算平均值
    for name, stats in kernel_stats.items():
        if stats['durations']:
            stats['avg_us'] = stats['total_us'] / len(stats['durations'])
            stats['min_us'] = min(stats['durations'])
            stats['max_us'] = max(stats['durations'])
        del stats['durations']  # 删除原始数据以节省内存
    
    return dict(kernel_stats)


def analyze_stream_events(events: List[dict]) -> Dict[str, dict]:
    """分析 CUDA stream 事件"""
    stream_events = defaultdict(list)
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        # 查找 stream ID
        args = event.get('args', {})
        if isinstance(args, dict):
            stream_id = args.get('stream', None)
            if stream_id is not None:
                stream_events[f'stream_{stream_id}'].append(event)
    
    # 计算每个 stream 的统计信息
    stream_stats = {}
    for stream_name, events_list in stream_events.items():
        durations = []
        for event in events_list:
            dur = event.get('dur', 0)
            if dur > 0:
                durations.append(dur)
        
        if durations:
            stream_stats[stream_name] = {
                'event_count': len(events_list),
                'total_us': sum(durations),
                'avg_us': sum(durations) / len(durations),
            }
    
    return stream_stats


def analyze_execution_order(events: List[dict]) -> List[Tuple[str, float, float]]:
    """分析执行顺序，返回 (事件名, 开始时间, 持续时间) 列表"""
    execution_order = []
    
    for event in events:
        if not isinstance(event, dict):
            continue
        
        name = event.get('name', '')
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        
        # 只关注 micro-batch 相关事件
        if 'mb_' in name and dur > 0:
            execution_order.append((name, ts, dur))
    
    # 按开始时间排序
    execution_order.sort(key=lambda x: x[1])
    
    return execution_order


def compare_profiles(baseline_data: dict, orion_data: dict) -> dict:
    """比较基准和 Orion 的 profiler 数据"""
    baseline_events = extract_events(baseline_data)
    orion_events = extract_events(orion_data)
    
    comparison = {
        'baseline': {
            'total_events': len(baseline_events),
            'micro_batch_stats': analyze_micro_batch_timing(baseline_events),
            'kernel_stats': analyze_kernel_events(baseline_events),
            'stream_stats': analyze_stream_events(baseline_events),
        },
        'orion': {
            'total_events': len(orion_events),
            'micro_batch_stats': analyze_micro_batch_timing(orion_events),
            'kernel_stats': analyze_kernel_events(orion_events),
            'stream_stats': analyze_stream_events(orion_events),
        },
    }
    
    # 计算差异
    baseline_mb = comparison['baseline']['micro_batch_stats']
    orion_mb = comparison['orion']['micro_batch_stats']
    
    mb_comparison = {}
    for mb_name in set(baseline_mb.keys()) | set(orion_mb.keys()):
        baseline_avg = baseline_mb.get(mb_name, {}).get('avg_us', 0)
        orion_avg = orion_mb.get(mb_name, {}).get('avg_us', 0)
        
        if baseline_avg > 0:
            speedup = baseline_avg / orion_avg if orion_avg > 0 else 0
            diff_pct = (baseline_avg - orion_avg) / baseline_avg * 100
        else:
            speedup = 0
            diff_pct = 0
        
        mb_comparison[mb_name] = {
            'baseline_avg_us': baseline_avg,
            'orion_avg_us': orion_avg,
            'speedup': speedup,
            'diff_pct': diff_pct,
        }
    
    comparison['micro_batch_comparison'] = mb_comparison
    
    return comparison


def print_analysis(analysis: dict, title: str = "Profile Analysis"):
    """打印分析结果"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    print(f"\n总事件数: {analysis.get('total_events', 0)}")
    
    # Micro-batch 统计
    mb_stats = analysis.get('micro_batch_stats', {})
    if mb_stats:
        print(f"\n--- Micro-batch 统计 ---")
        for mb_name, stats in sorted(mb_stats.items()):
            print(f"  {mb_name}:")
            print(f"    事件数: {stats['count']}")
            print(f"    总时间: {stats['total_us']/1000:.2f} ms")
            print(f"    平均时间: {stats['avg_us']/1000:.2f} ms")
            print(f"    最小/最大: {stats['min_us']/1000:.2f} / {stats['max_us']/1000:.2f} ms")
    
    # Stream 统计
    stream_stats = analysis.get('stream_stats', {})
    if stream_stats:
        print(f"\n--- Stream 统计 ---")
        for stream_name, stats in sorted(stream_stats.items()):
            print(f"  {stream_name}: {stats['event_count']} 事件, 总时间 {stats['total_us']/1000:.2f} ms")


def print_comparison(comparison: dict):
    """打印比较结果"""
    print(f"\n{'='*60}")
    print(f" 基准 vs Orion 比较")
    print(f"{'='*60}")
    
    print(f"\n基准总事件数: {comparison['baseline']['total_events']}")
    print(f"Orion 总事件数: {comparison['orion']['total_events']}")
    
    # Micro-batch 比较
    mb_comp = comparison.get('micro_batch_comparison', {})
    if mb_comp:
        print(f"\n--- Micro-batch 时间比较 ---")
        print(f"{'MB':<10} {'基准(ms)':<12} {'Orion(ms)':<12} {'加速比':<10} {'差异%':<10}")
        print("-" * 54)
        
        for mb_name, stats in sorted(mb_comp.items()):
            baseline_ms = stats['baseline_avg_us'] / 1000
            orion_ms = stats['orion_avg_us'] / 1000
            speedup = stats['speedup']
            diff_pct = stats['diff_pct']
            
            print(f"{mb_name:<10} {baseline_ms:<12.2f} {orion_ms:<12.2f} {speedup:<10.2f}x {diff_pct:<10.1f}%")
    
    # 总结
    print(f"\n--- 总结 ---")
    total_baseline = sum(s.get('total_us', 0) for s in comparison['baseline']['micro_batch_stats'].values())
    total_orion = sum(s.get('total_us', 0) for s in comparison['orion']['micro_batch_stats'].values())
    
    if total_baseline > 0 and total_orion > 0:
        overall_speedup = total_baseline / total_orion
        overall_diff = (total_baseline - total_orion) / total_baseline * 100
        print(f"总 micro-batch 时间 - 基准: {total_baseline/1000:.2f} ms, Orion: {total_orion/1000:.2f} ms")
        print(f"整体加速比: {overall_speedup:.2f}x ({overall_diff:.1f}% 差异)")


def find_latest_trace(directory: str) -> Optional[str]:
    """在目录中找到最新的 trace 文件"""
    pattern = os.path.join(directory, "**", "*.pt.trace.json")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        return None
    
    # 返回最新的文件
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='分析 PyTorch Profiler JSON 文件')
    parser.add_argument('--baseline', type=str, help='基准 profiler 目录')
    parser.add_argument('--orion', type=str, help='Orion profiler 目录')
    parser.add_argument('--file', type=str, help='单个 trace 文件路径')
    parser.add_argument('--rank', type=int, default=0, help='要分析的 rank (默认: 0)')
    
    args = parser.parse_args()
    
    if args.file:
        # 分析单个文件
        print(f"分析文件: {args.file}")
        trace_data = load_trace_json(args.file)
        events = extract_events(trace_data)
        
        analysis = {
            'total_events': len(events),
            'micro_batch_stats': analyze_micro_batch_timing(events),
            'kernel_stats': analyze_kernel_events(events),
            'stream_stats': analyze_stream_events(events),
        }
        
        print_analysis(analysis, f"文件分析: {os.path.basename(args.file)}")
        
        # 打印执行顺序
        exec_order = analyze_execution_order(events)
        if exec_order:
            print(f"\n--- 执行顺序 (前 20 个 micro-batch 事件) ---")
            for i, (name, ts, dur) in enumerate(exec_order[:20]):
                print(f"  {i+1}. {name}: 开始 {ts/1000:.2f} ms, 持续 {dur/1000:.2f} ms")
    
    elif args.baseline and args.orion:
        # 比较两个目录
        baseline_dir = os.path.join(args.baseline, f'rank{args.rank}')
        orion_dir = os.path.join(args.orion, f'rank{args.rank}')
        
        baseline_file = find_latest_trace(baseline_dir)
        orion_file = find_latest_trace(orion_dir)
        
        if not baseline_file:
            print(f"错误: 在 {baseline_dir} 中找不到 trace 文件")
            return
        
        if not orion_file:
            print(f"错误: 在 {orion_dir} 中找不到 trace 文件")
            return
        
        print(f"基准文件: {baseline_file}")
        print(f"Orion 文件: {orion_file}")
        
        baseline_data = load_trace_json(baseline_file)
        orion_data = load_trace_json(orion_file)
        
        comparison = compare_profiles(baseline_data, orion_data)
        
        print_analysis(comparison['baseline'], "基准分析")
        print_analysis(comparison['orion'], "Orion 分析")
        print_comparison(comparison)
    
    else:
        # 尝试自动查找文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_dir = os.path.join(script_dir, 'profiler_baseline')
        orion_dir = os.path.join(script_dir, 'profiler_orion')
        
        if os.path.exists(baseline_dir) and os.path.exists(orion_dir):
            print("自动检测到 profiler_baseline 和 profiler_orion 目录")
            
            for rank in range(4):
                baseline_file = find_latest_trace(os.path.join(baseline_dir, f'rank{rank}'))
                orion_file = find_latest_trace(os.path.join(orion_dir, f'rank{rank}'))
                
                if baseline_file and orion_file:
                    print(f"\n{'#'*60}")
                    print(f" Rank {rank} 分析")
                    print(f"{'#'*60}")
                    
                    baseline_data = load_trace_json(baseline_file)
                    orion_data = load_trace_json(orion_file)
                    
                    comparison = compare_profiles(baseline_data, orion_data)
                    print_comparison(comparison)
        else:
            print("使用方法:")
            print("  python analyze_profiler.py --baseline ./profiler_baseline --orion ./profiler_orion")
            print("  python analyze_profiler.py --file ./path/to/trace.json")
            print("\n请先运行实验生成 profiler 数据:")
            print("  1. 运行基准实验 (无 Orion):")
            print("     srun --gpus=4 bash run_gpipe_2d.sh")
            print("  2. 运行 Orion 实验:")
            print("     srun --gpus=4 bash run_gpipe_orion.sh")


if __name__ == '__main__':
    main()