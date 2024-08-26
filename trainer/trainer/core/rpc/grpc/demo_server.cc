#include <unistd.h>

#include <limits>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "trainer/core/rpc/grpc/grpc_server.h"
#include "trainer/core/rpc/request_handler_impl.h"

DEFINE_string(app_name, "demo_server", "");
DEFINE_int32(listen_port, 5800, "listen port");

using namespace sniper;
using namespace sniper::rpc;

std::unique_ptr<RPCServer> g_rpc_service;
std::unique_ptr<RequestHandler> embedding_lookup_handler;
std::unique_ptr<RequestHandler> push_gradient_handler;

void StartServer() {
  g_rpc_service->RegisterRPC(kRequestEmbeddingLookup,
                             embedding_lookup_handler.get(), 56);
  g_rpc_service->RegisterRPC(kRequestPushGrad, push_gradient_handler.get(), 56);

  std::thread server_thread(
      std::bind(&RPCServer::StartServer, g_rpc_service.get()));

  server_thread.join();
}

void init() {
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  embedding_lookup_handler.reset(new RequestLookupHandler(kSync));
  push_gradient_handler.reset(new RequestPushGradHandler(kSync));
  g_rpc_service.reset(new AsyncGRPCServer(
      GetHostIP() + ":" + std::to_string(FLAGS_listen_port), 2));
  std::thread server_thread(StartServer);

  g_rpc_service->WaitServerReady();

  server_thread.join();

  g_rpc_service->ShutDown();
  g_rpc_service.reset(nullptr);
  embedding_lookup_handler.reset(nullptr);
  push_gradient_handler.reset(nullptr);

  return 0;
}
