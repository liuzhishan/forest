#include "trainer/core/rpc/rpc_server.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <immintrin.h>
#include <net/if.h>
#include <netinet/in.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "glog/logging.h"

namespace sniper {
namespace rpc {

void RPCServer::ShutDown() {
  LOG(INFO) << "RPCServer ShutDown";
  ShutDownImpl();

  exit_flag_ = true;
}

int RPCServer::GetClientNum() {
  std::unique_lock<std::mutex> lock(mutex_);
  return client_num_;
}

void RPCServer::RegisterRPC(const std::string& rpc_name,
                            RequestHandler* handler, int thread_num) {
  rpc_call_map_[rpc_name] = handler;
  rpc_thread_num_[rpc_name] = thread_num;

  LOG(INFO) << "RegisterRPC rpc_name: " << rpc_name
            << ", handler: " << (void*)handler;
}

std::string GetHostIP() {
  struct ifaddrs* ifAddrStruct = nullptr;
  struct ifaddrs* ifa = nullptr;
  std::string ip;

  getifaddrs(&ifAddrStruct);
  for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
    if (nullptr == ifa->ifa_addr) continue;

    if (AF_INET == ifa->ifa_addr->sa_family &&
        0 == (ifa->ifa_flags & IFF_LOOPBACK)) {
      char address_buffer[INET_ADDRSTRLEN];
      void* sin_addr_ptr =
          &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);

      ip = address_buffer;
      break;
    }
  }
  if (nullptr != ifAddrStruct) freeifaddrs(ifAddrStruct);
  return ip;
}

}  // namespace rpc
}  // namespace sniper
