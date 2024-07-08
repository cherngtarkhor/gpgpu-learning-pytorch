import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        print(in_features, out_features, bias)
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        #print("forward input", input)
        #print("forward mask", mask)
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            #threshold = out.sum(axis=1).mean().item()
            #hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            #hi_idx = torch.from_numpy(hi_idx)#.cuda()
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)
        #print("out", out)
        #print("hi_idx", hi_idx)
        return out, hi_idx


model = MyModule(500, 10)#.cuda()
input = torch.rand(128, 500)#.cuda()
# mask = torch.rand((500, 500, 500), dtype=torch.double)#.cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float)#.cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
