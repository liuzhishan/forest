# 自动负载均衡

## 方案

###  `trainer`

###  `ps`

### `hub`

`hub` 的 `placement` 是在创建 `pipeline` 时根据 `StartSampleOption` 中的 `ps shard` 参数确定的，
`pipeline` 运行过程中不会改变。并且在每一个 `batch` 数据组装时，都会调用 `placement` 来确定每个 `sparse`
特征对应的 `ps` 节点。因此调用频率也比较高，需要考虑对性能的影响。

但是 `auto shard` 的 `placement` 是在运行时确定的，并且可能随时改变。因此需要一个运行时机制修改 `hub`
中的 `placement`。

有两种思路可以考虑:

1. 在 `hub server` 中增加一个 `Arc<Mutex<Placement>>`，各子任务都读取同一个 `placement`，当 `auto shard` 确定
   新的 `placement` 后，`hub server` 收到请求后只需更新 `Arc<Mutex<Placement>>` 中的 `placement` 即可。但是
   这种方式的缺点是多个子任务会同时访问 `placement`，每次访问都需要加锁，性能损耗较大。
2. 每个子任务保存自己的 `placement`，当 `hub server` 收到 `auto shard` 更新的请求后，通过 `channel` 通知各子任务
   更新 `placement`。每个子任务需要一个线程来接收更新请求, 因此 `placement` 依然需要加锁。不过大部分时间
   都是空闲的，在不更新的时候, 每个子任务都只有一个只读操作，不会有其他线程竞争。因此不会有太多性能开销。

经过分析，第二种方案更优。采用第二种方案实现。

## 实现细节

1. 在 `hub` 更新 `placement` 后，需要将读取的 `batch` 数据清空，否则 `ps` 在查 `embedding` 表时会找不到表。
2. 因为新的 `shard` 都是在旧的基础上增加分片, 因此 `ps` 只需要更新新增的分片, 不需要删除旧的。
3. `ps` 也需要更新 `scheduler` 中的 `ps_shard`，用于判断保存是否结束。

### 