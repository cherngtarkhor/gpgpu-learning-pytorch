import torch
import torchvision.models as models

# Load a pre-trained model
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()

# Dummy input for inference
dummy_input = torch.randn(1, 3, 224, 224)

from torch.profiler import profile, record_function, ProfilerActivity

# with profile(
#     schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/inf1'),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:
#     # Run inference in the profiled context
#     with torch.no_grad():
#         for _ in range(3):
#             prof.step()
#             model(dummy_input)


prof = torch.profiler.profile(
#        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/inf2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

with torch.no_grad():
    model(dummy_input)

prof.start()
# Run inference in the profiled context
with torch.no_grad():
    for _ in range(5):
        prof.step()
        model(dummy_input)

prof.stop()