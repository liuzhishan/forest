#include <chrono>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

REGISTER_OP("SniperRestore")
    .Attr("varname: string")
    .Attr("shard_idx: int")
    .Attr("shard_num: int")
    .Attr("shard_nfs_weight_paths: list(string)")
    .Attr("shard_nfs_adagrad_paths: list(string)")
    .Attr("conf_file: string")
    .Attr("trainer_id: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    });

}  // namespace ops
}  // namespace sniper
