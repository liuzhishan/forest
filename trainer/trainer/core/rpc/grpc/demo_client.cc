#include <unistd.h>

#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/grpc/grpc_client.h"
#include "trainer/core/util/monitor/run_status.h"

DEFINE_string(app_name, "demo_client", "");
DEFINE_string(server_addr, "", "");

using namespace sniper::rpc;
using namespace sniper;

std::atomic<bool> stop_{false};

std::vector<std::string> eps({});

void init() {
}

void push_grad() {
  RPCClient* client = RPCClient::GetInstance<GRPCClient>(0);
  PushGradOption opt;
  opt.set_learning_rate(0.1);
  while (!stop_.load()) {
    auto start = std::chrono::steady_clock::now();
    std::vector<RpcHandlePtr> hs;
    // responses.resize(100);
    for (int i = 0; i < 12; ++i) {
      tensorflow::Tensor grad(tensorflow::DT_FLOAT, {1024, 16 * 10});
      tensorflow::Tensor t1(tensorflow::DT_BFLOAT16, grad.shape());
      tensorflow::FloatToBFloat16(grad.flat<float>().data(),
                                  t1.flat<tensorflow::bfloat16>().data(),
                                  grad.NumElements());

      tensorflow::Tensor t2;
      auto ep = eps[i % eps.size()];
      auto h = client->PushGradAsync(ep, 0, "test_var" + std::to_string(i), opt,
                                     t1, t2);
      hs.push_back(h);
    }

    for (int i = 0; i < 10; ++i) {
      hs[i]->Wait();
    }
    auto end = std::chrono::steady_clock::now();
    monitor::RunStatus::Instance()->PushTime(
        monitor::kOpsPushGrad,
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
  }
}

void lookup() {
  RPCClient* client = RPCClient::GetInstance<GRPCClient>(0);
  PushGradOption opt;
  opt.set_learning_rate(0.1);
  while (!stop_.load()) {
    auto start = std::chrono::steady_clock::now();
    std::vector<RpcHandlePtr> hs;
    std::vector<rpc::TensorResponse> responses;
    responses.resize(12);
    for (int i = 0; i < responses.size(); ++i) {
      auto ep = eps[i % eps.size()];
      EmbeddingLookupOption option;
      option.set_batch_size(1024);
      auto h = client->EmbeddingLookupAsync(
          ep, 0, "test_var" + std::to_string(i), option, tensorflow::Tensor(),
          tensorflow::Tensor(), &(responses[i]));

      hs.push_back(h);
    }

    for (int i = 0; i < responses.size(); ++i) {
      hs[i]->Wait();
    }
    //  for (size_t i = 0; i < responses.size(); ++i) {
    //    auto& lookup_ret = responses[i].tensor1();  // bfloat16

    //    tensorflow::Tensor float_lookup_ret(tensorflow::DT_FLOAT,
    //                                        lookup_ret.shape());
    //    tensorflow::BFloat16ToFloat(
    //        lookup_ret.flat<tensorflow::bfloat16>().data(),
    //        float_lookup_ret.flat<float>().data(), lookup_ret.NumElements());
    //  }
    auto end = std::chrono::steady_clock::now();
    monitor::RunStatus::Instance()->PushTime(
        monitor::kOpsPushGrad,
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
  }
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::vector<std::shared_ptr<std::thread>> threads;
  for (int i = 0; i < 40; ++i) {
    threads.push_back(std::make_shared<std::thread>(push_grad));
  }
  for (int i = 0; i < 40; ++i) {
    threads.push_back(std::make_shared<std::thread>(lookup));
  }

  for (auto t : threads) {
    t->join();
  }

  return 0;
}
