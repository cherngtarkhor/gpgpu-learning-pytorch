#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

  std::cout << "arg count:" << argc << std::endl;
  std::cout << "arg 1:" << argv[0] << std::endl;
  std::cout << "arg 2:" << argv[1] << std::endl;

  torch::jit::script::Module module;

  try {
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  //CODE CHANGE
  module.to(at::kXPU);

  std::vector<torch::jit::IValue> inputs;

  //CODE CHANGE
  torch::Tensor input = torch::rand({1, 3, 224, 224}).to(at::kXPU);
  //torch::Tensor input = torch::rand({1, 3, 224, 224});

  inputs.push_back(input);

  at::Tensor output = module.forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

  std::cout << "Execution finished" << std::endl;

  return 0;
}
