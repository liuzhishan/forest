#include <chrono>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

REGISTER_OP("StartSample")
    .Attr("parallel: int")
    .Attr("src: int")
    .Attr("file_list: list(string)")
    .Attr("work_mode: int")
    .Attr("conf_file: string")
    .Attr("trainer_id: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    });

}  // namespace ops
}  // namespace sniper