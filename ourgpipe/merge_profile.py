import json
import glob
import os


def _find_anchor_ts(events):
    """
    Find an anchor timestamp for alignment.
    Prefer "Iteration Start: PyTorch Profiler" if present; else min ts.
    """
    anchor = None
    for ev in events:
        if ev.get("name") == "Iteration Start: PyTorch Profiler":
            ts = ev.get("ts")
            if isinstance(ts, (int, float)):
                if anchor is None or ts < anchor:
                    anchor = ts
    if anchor is not None:
        return anchor
    min_ts = None
    for ev in events:
        ts = ev.get("ts")
        if isinstance(ts, (int, float)):
            if min_ts is None or ts < min_ts:
                min_ts = ts
    return min_ts


def merge_chrome_traces(input_dir, output_file, align_timestamps=True, reference_index=0):
    """
    合并多个Chrome trace JSON文件
    """
    all_events = []
    all_metadata = {}

    # 获取所有trace文件
    trace_files = glob.glob(os.path.join(input_dir, "rank*", "*.pt.trace.json"))
    trace_files.sort()  # 确保顺序一 ?
    print(f"找到 {len(trace_files)} 个trace文件")

    # Pre-load traces and anchors for alignment
    traces = []
    anchors = []
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        traces.append(data)
        anchors.append(_find_anchor_ts(data.get('traceEvents', [])))

    ref_anchor = None
    if align_timestamps and traces:
        if 0 <= reference_index < len(anchors):
            ref_anchor = anchors[reference_index]
        if ref_anchor is None:
            ref_anchor = min([a for a in anchors if a is not None], default=None)

    for i, trace_file in enumerate(trace_files):
        print(f"处理文件: {trace_file}")

        data = traces[i]

        # 提取events
        if 'traceEvents' in data:
            events = data['traceEvents']

            # Align timestamps across ranks if requested
            if align_timestamps and ref_anchor is not None and anchors[i] is not None:
                delta = ref_anchor - anchors[i]
                if delta != 0:
                    for event in events:
                        ts = event.get('ts')
                        if isinstance(ts, (int, float)):
                            event['ts'] = ts + delta

            # 为每个rank的事件添加唯一的进程ID偏移
            pid_offset = i * 10000  # 每个rank使用不同的PID范围

            for event in events:
                if 'pid' in event:
                    # 如果pid是数字，加上偏移 ?
                    if isinstance(event['pid'], int):
                        event['pid'] += pid_offset
                    # 如果pid是字符串，添加rank前缀
                    elif isinstance(event['pid'], str):
                        event['pid'] = f"rank_{i}_{event['pid']}"

                # 同样处理tid
                if 'tid' in event:
                    if isinstance(event['tid'], int):
                        event['tid'] += pid_offset
                    elif isinstance(event['tid'], str):
                        event['tid'] = f"rank_{i}_{event['tid']}"

            all_events.extend(events)

        # 合并metadata
        if 'metadata' in data:
            all_metadata.update(data['metadata'])

    # 创建合并后的trace
    merged_trace = {
        'traceEvents': all_events,
        'metadata': all_metadata
    }

    # 写入输出文件
    with open(output_file, 'w') as f:
        json.dump(merged_trace, f)

    print(f"合并完成！输出文 ? {output_file}")
    print(f"总事件数: {len(all_events)}")


# 使用方法
if __name__ == "__main__":
    # 合并 Orion 调度器的 profiler 文件
    input_directory = "./profiler_orion"
    output_file = "./profiler_orion/merged_orion_trace.json"

    # # 合并 baseline  ?profiler 文件
    # input_directory = "./profiler_baseline"
    # output_file = "./profiler_baseline/merged_baseline_trace.json"

    # 合并 baseline  ?profiler 文件
    # input_directory = "./profiler_thread-stream_512_2"
    # output_file = "./profiler_thread-stream_512_2/merged_baseline_trace.json"

    merge_chrome_traces(input_directory, output_file)
