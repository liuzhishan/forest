#include <unistd.h>

#include <limits>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "core/rpc/grpc/grpc_server.h"
#include "core/rpc/request_handler_impl.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

DEFINE_int64(spdlog_max_size, 1024 * 1024 * 1024, "log file max size");
DEFINE_int32(spdlog_max_files, 10, "log files num");
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
  // Set the default logger to file logger
  auto logger =
      spdlog::rotating_logger_mt(FLAGS_app_name, "../log/" + FLAGS_app_name,
                                 FLAGS_spdlog_max_size, FLAGS_spdlog_max_files);
  logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f %t] [%@] [%L] %v");
  spdlog::set_default_logger(logger);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  init();

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
