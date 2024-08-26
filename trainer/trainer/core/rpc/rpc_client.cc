#include "trainer/core/rpc/rpc_client.h"

#include "gflags/gflags.h"

DEFINE_int32(rpc_deadline, 60000, "deadline timeouts for rpc, default 10s");
DEFINE_int32(rpc_retry_times, 0, "retry times for rpc");

namespace sniper {
namespace rpc {

std::once_flag RPCClient::init_flag_;
std::unique_ptr<RPCClient> RPCClient::rpc_client_(nullptr);
int RPCClient::role_id_ = 0;

}  // namespace rpc
}  // namespace sniper
