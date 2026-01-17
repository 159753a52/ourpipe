# test_profile_gpt.py
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from GPT import CharGPT, tok, process, datasets, batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CharGPT(len(tok.char2ind)).to(device)
model.train()

tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=device)

loader = torch.utils.data.DataLoader(
    tokenized['train'], batch_size=batch_size, shuffle=True
)
batch = next(iter(loader))
inputs, labels = batch['inputs'], batch['labels']

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=False,
) as prof:
    for step in range(3):
        optimizer.zero_grad()
        logits = model(inputs)
        logits = logits.transpose(-2, -1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("gpt_ffn_test_trace.json")
print("trace 保存为 gpt_ffn_test_trace.json")
