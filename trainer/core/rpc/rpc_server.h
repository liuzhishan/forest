#pragma once

#include <atomic>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/rpc/request_handler.h"

namespace sniper {
namespace rpc {

class RPCServer {
 public:
  explicit RPCServer(const std::string& address, int client_num)
      : bind_address_(address), exit_flag_(false), client_num_(client_num) {}

  virtual ~RPCServer() {}
  virtual void StartServer() = 0;
  virtual void WaitServerReady() = 0;

  void ShutDown();

  bool IsExit() { return exit_flag_.load(); }

  int GetClientNum();

  // RegisterRPC, register the rpc method name to a handler
  // class, and auto generate a condition id for this call
  // to be used for the barrier.
  void RegisterRPC(const std::string& rpc_name, RequestHandler* handler,
                   int thread_num = 5);

  int GetThreadNum(const std::string& rpc_name) {
    return rpc_thread_num_[rpc_name];
  }

 protected:
  virtual void ShutDownImpl() = 0;

 private:
  std::mutex mutex_;

 protected:
  std::string bind_address_;
  std::atomic<int> exit_flag_;
  int client_num_;

  std::unordered_map<std::string, RequestHandler*> rpc_call_map_;
  std::unordered_map<std::string, int> rpc_thread_num_;
  friend class RequestHandler;
};

std::string GetHostIP();

};  // namespace rpc
};  // namespace sniper
