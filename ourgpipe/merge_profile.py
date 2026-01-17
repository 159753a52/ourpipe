import json
import glob
import os

def merge_chrome_traces(input_dir, output_file):
    """
    合并多个Chrome trace JSON文件
    """
    all_events = []
    all_metadata = {}
    
    # 获取所有trace文件
    trace_files = glob.glob(os.path.join(input_dir, "rank*", "*.pt.trace.json"))
    trace_files.sort()  # 确保顺序一致
    
    print(f"找到 {len(trace_files)} 个trace文件")
    
    for i, trace_file in enumerate(trace_files):
        print(f"处理文件: {trace_file}")
        
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # 提取events
        if 'traceEvents' in data:
            events = data['traceEvents']
            
            # 为每个rank的事件添加唯一的进程ID偏移
            pid_offset = i * 10000  # 每个rank使用不同的PID范围
            
            for event in events:
                if 'pid' in event:
                    # 如果pid是数字，加上偏移量
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
    
    print(f"合并完成！输出文件: {output_file}")
    print(f"总事件数: {len(all_events)}")

# 使用方法
if __name__ == "__main__":
    input_directory = "./profiler_thread-stream_4096"
    output_file = "./profiler_thread-stream_4096/thread-stream_4096_trace.json"
    
    merge_chrome_traces(input_directory, output_file)