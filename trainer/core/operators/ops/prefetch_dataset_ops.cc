#include <chrono>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

REGISTER_OP("InputPrefetchDataset")
    .Input("conf_file: string")
    .Input("trainer_id: int32")
    .Input("work_mode: int32")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Output("feature: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
prefetch from feed_queue
)doc");

}  // namespace ops
}  // namespace sniper
