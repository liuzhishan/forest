# hub

`hub` 是 `IO` 模块，主要负责读取数据，并且与 `ps` 和 `trainer` 进行通信。

`hub` 读取输入数据，如果是非 batch 数据组装成 batch，生成 batch_id, 并将 batch_id
和数据发送给 `ps` 进行 `embedding` 处理，同时将 batch_id 发送给 `trainer` 供 `trainer`
获取样本。

主要包含以下部分:
- `hub_server`: 主程序，负责启动各个 `worker` 线程。
- `request_handler`: 负责处理 `grpc` 请求，调用具体的处理逻辑，并返回结果。
- `node`: 工作节点，复杂处理各种中间逻辑。
- `pipeline`: 处理不同格式数据的 `node pipeline`, 由多个 `node` 顺序组成。
- `stream`: 读取数据的接口，如 `hdfs`, `kafka` 等, 第一版只支持 `hdfs`。
