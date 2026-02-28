"""
WeightGradStore — ZeroBubble 专用的延迟权重梯度存储

核心思想：将反向传播拆分为 B (input gradient) 和 W (weight gradient)，
B 立即计算（因为下一层的反向传播依赖它），W 延迟执行以填充 pipeline bubble。

用法：
    WeightGradStore.split_bw = True   # 开启 B/W 分离
    loss.backward()                    # BubbleLinear 会把 W 闭包推入 cache
    WeightGradStore.flush()            # 将 cache 打包推入队列
    ...                                # 在 bubble 时间执行 W
    WeightGradStore.pop()              # 取出并执行一组 W 计算
"""

import queue
from torch.profiler import record_function


class WeightGradStore:
    """存储延迟执行的 W (Weight Gradient) 计算任务"""

    cache = []                          # 当前微批次的 W 任务列表
    weight_grad_queue = queue.Queue()   # 等待执行的 W 任务队列
    split_bw = False                    # 是否开启 B/W 分离

    @classmethod
    def put(cls, compute_func):
        """存入一个计算闭包。

        如果 split_bw=False，直接执行（退化为标准 1F1B）。
        """
        if not cls.split_bw:
            compute_func()
            return
        cls.cache.append(compute_func)

    @classmethod
    def flush(cls):
        """将当前 cache 打包推入队列"""
        if cls.cache:
            cls.weight_grad_queue.put(list(cls.cache))
            cls.cache = []

    @classmethod
    def pop(cls):
        """取出并执行一整组 W 计算任务"""
        if cls.weight_grad_queue.qsize() == 0:
            return
        with record_function("ZeroBubble_W_Step"):
            stored_funcs = cls.weight_grad_queue.get()
            for func in stored_funcs:
                func()

    @classmethod
    def pop_all(cls):
        """执行队列中所有剩余的 W 计算任务"""
        while cls.weight_grad_queue.qsize() > 0:
            cls.pop()

    @classmethod
    def clear(cls):
        """重置所有状态"""
        cls.cache = []
        cls.weight_grad_queue = queue.Queue()
        cls.split_bw = False
