#include <chrono>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

REGISTER_OP("FeedQueue")
    .Output("handle: resource")
    .Attr("conf_file: string")
    .Attr("trainer_id: int")
    .Attr("work_mode: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A TF resource that comsume dense_label from hub servers
and then fetch sparse embedding from ps shard servers that is ready st ps.
container: required for a resource op kernel.
shared_name: If non-empty, this queue will be shared under the given name across multiple sessions.
)doc");

REGISTER_OP("Prefetch")
    .Input("handle: resource")
    .Attr("conf_file: string")
    .Attr("trainer_id: int")
    .Attr("work_mode: int")
    .Attr("batch_size: int")
    .Attr("label_size: int")
    .Attr("dense_total_size: int")
    .Attr("sparse_emb_size: list(int)")
    .Attr("T: list({float, int64, int32})")
    .Output("feature: T")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::int32 batch_size;
      tensorflow::int32 label_size;
      tensorflow::int32 dense_total_size;
      std::vector<tensorflow::int32> sparse_emb_size;
      TF_RETURN_IF_ERROR(c->GetAttr("batch_size", &batch_size));
      TF_RETURN_IF_ERROR(c->GetAttr("label_size", &label_size));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_total_size", &dense_total_size));
      TF_RETURN_IF_ERROR(c->GetAttr("sparse_emb_size", &sparse_emb_size));

      int output_idx = 0;
      c->set_output(output_idx++, c->Vector(1));
      c->set_output(output_idx++, c->MakeShape({label_size, batch_size}));
      c->set_output(output_idx++, c->MakeShape({batch_size, dense_total_size}));
      for (size_t i = 0; i < sparse_emb_size.size(); ++i) {
        c->set_output(output_idx++,
                      c->MakeShape({batch_size, sparse_emb_size[i]}));
      }

      return Status::OK();
    })
    .Doc(R"doc(
prefetch from feed_queue
)doc");

}  // namespace ops
}  // namespace sniper
