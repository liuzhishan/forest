#include <chrono>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

REGISTER_OP("SniperSave")
    .Input("version: int64")
    .Attr("varname: string")
    .Attr("type: int")
    .Attr("target: int")
    .Attr("nfs_path: string")
    .Attr("queue: string")
    .Attr("conf_file: string")
    .Attr("trainer_id: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    });

}  // namespace ops
}  // namespace sniper