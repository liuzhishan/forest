#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "trainer/core/rpc/grpc/grpc_service.h"
#include "trainer/core/rpc/rpc_server.h"

namespace sniper {
namespace rpc {

class RequestBase;

class AsyncGRPCServer final : public RPCServer {
 public:
  explicit AsyncGRPCServer(const std::string& address, int client_num)
      : RPCServer(address, client_num), ready_(0) {}

  virtual ~AsyncGRPCServer() {}
  void WaitServerReady() override;
  void StartServer() override;

 private:
  // HandleRequest needs to be thread-safe.
  void HandleRequest(
      ::grpc::ServerCompletionQueue* cq, const std::string& rpc_name,
      std::function<void(const std::string&, int)> TryToRegisterNewOne);

  void TryToRegisterNewOne(const std::string& rpc_name, int req_id);
  void ShutdownQueue();
  void ShutDownImpl() override;

 private:
  static const int kRequestBufSize = 200;

  std::mutex cq_mutex_;
  volatile bool is_shut_down_ = false;

  std::unique_ptr<GrpcService::AsyncService> service_;
  std::unique_ptr<::grpc::Server> server_;

  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;
  int ready_;

  std::map<std::string, std::unique_ptr<::grpc::ServerCompletionQueue>> rpc_cq_;
  std::map<std::string, std::vector<std::unique_ptr<std::thread>>> rpc_threads_;
  std::map<std::string, std::vector<RequestBase*>> rpc_reqs_;
};

};  // namespace rpc
};  // namespace sniper
