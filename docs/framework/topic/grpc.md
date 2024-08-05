# Grpc

`grpc` 主要用于各模块之间的通信。如何实现 `grpc` 相关的接口?

##  各模块之间需要通信的内容

模型训练的整体流程如下:

1. 各模块初始化。
2. `hub` 读取数据，分别发送给 `trainer` 和 `ps`, 两者对接受到的数据并行进行处理。
3. `trainer` 请求 `ps` 获取 `sparse embedding` 参数，进行模型训练，并更新梯度等参数到 `ps`。
4. `ps` 进行参数更新、保存等操作。
5. `trainer` 周期性的保存模型到 `hdfs`。

结合如上流程，我们首先来看下各模块之间需要通信的内容。


### `trainer` 和 `hub`

`trainer` 要进行训练，需要 `hub` 先读取数据, 之后 `hub` 再将 `batch_id` 发送给 `trainer`。

因此需要以下接口。

- `StartSample`:
  - `client`: `trainer`, 将读取数据需要的参数发送给 `hub`, 通知 `hub` 开始读取数据。
  - `server`: `hub`, 启动读数据任务。
- `ReadSample`:
  - `client`: `trainer`, 发送读取数据请求。
  - `server`: `hub`, 读取特征数据，组装 `batch`, 并返回 `batch_id`、`dense` 特征、`label` 等训练
  所需数据给 `trainer`。


### `trainer` 和 `ps`

整个训练阶段从初始化，再到模型训练、参数更新、模型保存，`trainer` 都需要和 `ps` 进行通信
以交换各种信息。需要以下接口。

- `Create`:
  - `client`: `trainer`, 发送特征列表。
  - `server`: `ps`, 初始化参数。
- `Freeze`:
  - `client`: `trainer`, 通知 `ps` 进行 `Freeze` 操作。
  - `server`: `Freeze`。
- `EmbeddingLookup`:
  - `client`: `trainer`, 发送 `batch_id` 到 `ps`，根据 `batch_id` 获取 `embedding` 参数。
  - `server`: `ps`, 根据 `batch_id` 查找 `embedding` 参数，并进行 `sum` 操作，返回给 `trainer`。
- `Push`:
  - `client`: `trainer`, 根据变量名发送参数到 `ps`。
  - `server`: `ps`, 保存参数。
- `Pull`:
  - `client`: `trainer`, 根据变量名获取参数。
  - `server`: `ps`, 根据变量名发送参数到 `trainer`。
- `PushGrad`:
  - `client`: `trainer`, 发送梯度参数到 `ps`。
  - `server`: `ps`, 保存梯度参数到 `map`。
- `Save`:
  - `client`: `trainer`, 发送需要保存的变量名到 `ps`。
  - `server`: `ps`, 保存参数到 `hdfs`。
- `Restore`:
  - `client`: `trainer`, 发送加载模型需要的参数到 `ps`。
  - `server`: `ps`, 根据 `trainer` 参数加载模型参数。
- `Heartbeat`:
  - `client`: `trainer`, 检查 `ps` 是否正常。
  - `server`: `ps`, 发送是否正常的结果。
  

### `hub` 和 `ps`

`hub` 读取数据后将数据发送给 `ps` 进行并行处理。

- `FeedSample`:
  - `client`: `hub`, 发送读取的 `sparse` 特征数据和 `batch_id` 到 `ps`。
  - `server`: `ps`, 对接受到的 `sparse` 特征数据进行查找 `embedding` 的操作，并将结果放
  到缓存中。


## 接口设计

### 易用性以及性能

易用性以及性能是接口设计需要重点考虑的两个因素。

各模块之间通信的请求处理各不相同，但是请求接口的逻辑基本是一致的，有很多可以复用的逻辑。
而各模块之间的接口主要以返回训练中 `tensor` 使用的各种数据为主，以及 `batch_id`, `req_id`
等信息。因此可以将接口参数列表都统一到同一个参数列表，返回类型用 `TensorMessage` 或者
`VoidMessage`。这样可以将请求 `grpc` 的接口进行统一，不同的 `grpc` 请求只需要根据实际逻辑
构造请求参数即可。

在 `c++` 中采可以采用继承的方式来实现不同的 `grpc` 请求。为了提高性能，可以为每个 `grpc`
都分配一个枚举，从而在分发请求时根据枚举进行静态分发。

返回结果中的 `tensor` 通常包含大量的参数，`trainer` 需要获取这些数据进行训练。在 `tensor`
与 `grpc buffer` 的互相转换过程中，我们可以采取零拷贝的方式进行处理，可以避免拷贝数据的开销，
从而提升性能。

而在 `rust` 中，`tonic` 已经提供了封装非常好的 `grpc client`, 可以直接使用。


## 参考

https://www.reddit.com/r/rust/comments/d7w6n7/is_it_idiomatic_to_write_setters_and_getters/

### Pin
