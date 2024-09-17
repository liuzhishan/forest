# training

## ps 添加 lru

使用 `VecDeque` 和 `DashMap` 结合的方式实现 `lru`。

## 训练速度慢

### 怀疑点: 是否和 `lru` 有关

添加 `lru` 后速度变慢。


    [1,0]<stderr>:2024-09-08 11:46:07,421 - INFO [hooks.py:269 - after_run] - 2024-09-08 11:46:07.421444: step 447, auc = 0.6508 (1.5 it/sec; 1571.5 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,421 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.421618: step 447, xentropy_mean:0 = 0.23541591 (1.5 it/sec; 1571.5 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,421 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.421672: step 447, prob_mean:0 = 0.09976812 (1.5 it/sec; 1571.5 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,421 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.421711: step 447, real_mean:0 = 0.07519531 (1.5 it/sec; 1571.5 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,449 - INFO [hooks.py:269 - after_run] - 2024-09-08 11:46:07.449222: step 448, auc = 0.6510 (36.0 it/sec; 36859.4 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,449 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.449362: step 448, xentropy_mean:0 = 0.32492530 (36.0 it/sec; 36907.9 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,449 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.449408: step 448, prob_mean:0 = 0.07809436 (36.1 it/sec; 36919.0 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,449 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.449444: step 448, real_mean:0 = 0.11425781 (36.1 it/sec; 36923.1 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07.494499: E ./trainer/core/rpc/grpc/grpc_client.h:91] /sniper.Sniper/PushGrad meets grpc error, error_code: 4, error_message: Deadline Exceedederror_details:
    [1,0]<stderr>:I0908 11:46:07.666785   651 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 24  P50: 12950000.000000  P95: 20800000.000000  P99: 21760000.000000  Max: 21812452.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 24  P50: 41500.000000  P95: 47139.000000  P99: 47139.000000  Max: 47139.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 24  P50: 250000.000000  P95: 326836.000000  P99: 326836.000000  Max: 326836.000000
    [1,0]<stderr>:2024-09-08 11:46:07,787 - INFO [hooks.py:269 - after_run] - 2024-09-08 11:46:07.787546: step 449, auc = 0.6512 (3.0 it/sec; 3026.7 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,787 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.787689: step 449, xentropy_mean:0 = 0.27475822 (3.0 it/sec; 3026.7 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,787 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.787737: step 449, prob_mean:0 = 0.12379644 (3.0 it/sec; 3026.6 examples/sec)
    [1,0]<stderr>:2024-09-08 11:46:07,787 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:46:07.787773: step 449, real_mean:0 = 0.09765625 (3.0 it/sec; 3026.6 examples/sec)


去掉 `lru`, 依然很慢, 还有些超时的报错。

    [1,0]<stderr>:2024-09-08 11:53:09,247 - INFO [hooks.py:269 - after_run] - 2024-09-08 11:53:09.247207: step 218, auc = 0.5843 (5.0 it/sec; 5110.7 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:09,247 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:53:09.247359: step 218, xentropy_mean:0 = 0.27160841 (5.0 it/sec; 5110.0 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:09,247 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:53:09.247410: step 218, prob_mean:0 = 0.06434439 (5.0 it/sec; 5109.9 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:09,247 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:53:09.247447: step 218, real_mean:0 = 0.08593750 (5.0 it/sec; 5109.9 examples/sec)
    [1,0]<stderr>:I0908 11:53:10.626993   651 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 29  P50: 11768750.000000  P95: 21280000.000000  P99: 25132320.000000  Max: 25132320.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 29  P50: 41564.000000  P95: 47583.000000  P99: 47583.000000  Max: 47583.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 29  P50: 11821.875000  P95: 20066.666667  P99: 20099.000000  Max: 20099.000000
    [1,0]<stderr>:2024-09-08 11:53:10.707045: E ./trainer/core/rpc/grpc/grpc_client.h:91] /sniper.Sniper/PushGrad meets grpc error, error_code: 4, error_message: Deadline Exceedederror_details:
    [1,0]<stderr>:2024-09-08 11:53:11.007986: E ./trainer/core/rpc/grpc/grpc_client.h:91] /sniper.Sniper/PushGrad meets grpc error, error_code: 4, error_message: Deadline Exceedederror_details:
    [1,0]<stderr>:2024-09-08 11:53:11.046993: E ./trainer/core/rpc/grpc/grpc_client.h:91] /sniper.Sniper/PushGrad meets grpc error, error_code: 4, error_message: Deadline Exceedederror_details:
    [1,0]<stderr>:2024-09-08 11:53:11,321 - INFO [hooks.py:269 - after_run] - 2024-09-08 11:53:11.321585: step 219, auc = 0.5846 (0.5 it/sec; 493.6 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:11,321 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:53:11.321766: step 219, xentropy_mean:0 = 0.27486286 (0.5 it/sec; 493.6 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:11,321 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:53:11.321839: step 219, prob_mean:0 = 0.08873546 (0.5 it/sec; 493.6 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:11,321 - INFO [hooks.py:237 - after_run] - 2024-09-08 11:53:11.321877: step 219, real_mean:0 = 0.09472656 (0.5 it/sec; 493.6 examples/sec)
    [1,0]<stderr>:2024-09-08 11:53:11,327 - INFO [hooks.py:269 - after_run] - 2024-09-08 11:53:11.327843: step 220, auc = 0.5848 (159.7 it/sec; 163536.8 examples/sec)


说明和 `lru` 没关系，就是 `EmbeddingLookup` 慢, 比 `PushGrad` 和 `ReadSample` 慢几个数量级。


### 怀疑点: `EmbeddingLookup` 时对整个 `EmbeddingManager` 加锁，锁的粒度太粗。

目前的实现如下, 在 `tokio::spawn` 每个任务获取 `embedding_manager.clone()`, 在进行其他操作。
这样每个线程在运行时锁是加在整个 `embedding_manager` 上的，导致其他 `var` 不能同时访问。


    for i in 0..varnames.len() {
        if i >= lookup_option.field_idx.len() {
            error_bail!(
                "out of range when embedding lookup, i: {}, field_idx.len(): {}",
                i,
                lookup_option.field_idx.len(),
            );
        }

        let new_batch_id = batch_id;
        let new_varname = varnames[i].clone();
        let new_lookup_option = lookup_option.clone();
        let batch_size = lookup_option.batch_size as usize;

        let field = lookup_option.field_idx[i];

        tasks.push(tokio::spawn(async move {
            let embedding_manager_clone = self.embedding_manager.clone();
            let embedding_manager_read = embedding_manager_clone.read();

            let var_option = embedding_manager_read.get(&new_varname);

            match var_option {
                Some(var) => {
                    var.value().embedding_lookup(field, batch_id, batch_size)
                },
                None => {
                    error_bail!(
                        "cannot find embedding by varname: {}",
                        new_varname.clone()
                    );
                }
            }
        }));
    }


### 怎样将锁加在每个 `Embedding` 上 ?

#### 使用 `Vec` 保存 `Embedding` ?

#### `Rayon` ?

#### `Arc<Vec>` ?

`Arc` 表示 `Atomically Reference Counted`。

[Arc and Mutex in Rust](https://itsallaboutthebit.com/arc-mutex/)

修改为 `Arc<Vec>` 后训练速度如下:

    [1,0]<stderr>:2024-09-12 00:17:15,286 - INFO [hooks.py:269 - after_run] - 2024-09-12 00:17:15.286718: step 212, auc = 0.6884 (0.7 it/sec; 751.9 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:15,286 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:15.286890: step 212, xentropy_mean:0 = 0.31322363 (0.7 it/sec; 751.9 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:15,286 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:15.286942: step 212, prob_mean:0 = 0.09636442 (0.7 it/sec; 751.9 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:15,286 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:15.286981: step 212, real_mean:0 = 0.10058594 (0.7 it/sec; 751.9 examples/sec)
    [1,0]<stderr>:I0912 00:17:15.702440   651 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 15  P50: 558125.000000  P95: 1297151.000000  P99: 1297151.000000  Max: 1297151.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 15  P50: 41500.000000  P95: 42711.000000  P99: 42711.000000  Max: 42711.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 15  P50: 4666666.666667  P95: 14063108.000000  P99: 14063108.000000  Max: 14063108.000000
    [1,0]<stderr>:2024-09-12 00:17:19,122 - INFO [hooks.py:269 - after_run] - 2024-09-12 00:17:19.122036: step 213, auc = 0.6888 (0.3 it/sec; 267.0 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,122 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:19.122210: step 213, xentropy_mean:0 = 0.27282825 (0.3 it/sec; 267.0 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,122 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:19.122260: step 213, prob_mean:0 = 0.09703215 (0.3 it/sec; 267.0 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,122 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:19.122299: step 213, real_mean:0 = 0.10058594 (0.3 it/sec; 267.0 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,539 - INFO [hooks.py:269 - after_run] - 2024-09-12 00:17:19.539652: step 214, auc = 0.6891 (2.4 it/sec; 2452.0 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,539 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:19.539801: step 214, xentropy_mean:0 = 0.28859243 (2.4 it/sec; 2452.2 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,539 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:19.539849: step 214, prob_mean:0 = 0.10570036 (2.4 it/sec; 2452.2 examples/sec)
    [1,0]<stderr>:2024-09-12 00:17:19,539 - INFO [hooks.py:237 - after_run] - 2024-09-12 00:17:19.539886: step 214, real_mean:0 = 0.09863281 (2.4 it/sec; 2452.2 examples/sec)


可以看出，`EmbeddingLookup` 速度提高了一个数量级，但是 `ReadSample` 速度变慢了。可能和 `ps FeedSample` 有关。 


### `hash` 函数

多个 `ps` 时经常报错 `EmbeddingLookup failed`, 经排查是因为不同的 `variable` 分配到不同的 `ps` 是根据变量名 `hash` 后对 ps
总数取模决定的，而 `rust`, `c++`, `python` 中使用的 `hash` 函数不同。`rust` 中默认使用的是 `SipHash13`, `c++` 中使用的是
`std::hash`, `python` 中使用的是 `hashlib.sha1`, 导致 `trainer` 发送请求时会发送到错误的 `ps` 上查 `Embedding`。


修复 `hash` 问题。

使用同样的逻辑手动实现 `string` 到 `u64` 的 `hash`。`rust` 版本示例如下

    pub fn simple_string_to_int_hash(s: &str) -> u64 {
        let mut hash = 0u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }

        hash
    }

修复后速度如下, 可以看出，主要瓶颈在 `EmbeddingLookup`, 比其他环节高了两个数量级。

    [1,0]<stderr>:2024-09-17 15:03:39,914 - INFO [hooks.py:237 - after_run] - 2024-09-17 15:03:39.914639: step 346, prob_mean:0 = 0.10711852 (0.8 it/sec; 776.9 examples/sec)
    [1,0]<stderr>:2024-09-17 15:03:39,914 - INFO [hooks.py:237 - after_run] - 2024-09-17 15:03:39.914676: step 346, real_mean:0 = 0.08300781 (0.8 it/sec; 776.9 examples/sec)
    [1,0]<stderr>:I0917 15:03:40.424185   658 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 21  P50: 6650000.000000  P95: 7253659.000000  P99: 7253659.000000  Max: 7253659.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 21  P50: 41500.000000  P95: 42352.000000  P99: 42352.000000  Max: 42352.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 21  P50: 8332.500000  P95: 9891.750000  P99: 10794.000000  Max: 10794.000000
    [1,0]<stderr>:2024-09-17 15:03:41,350 - INFO [hooks.py:269 - after_run] - 2024-09-17 15:03:41.350214: step 347, auc = 0.7320 (0.7 it/sec; 713.2 examples/sec)
    [1,0]<stderr>:2024-09-17 15:03:41,350 - INFO [hooks.py:237 - after_run] - 2024-09-17 15:03:41.350390: step 347, xentropy_mean:0 = 0.26466653 (0.7 it/sec; 713.2 examples/sec)
    [1,0]<stderr>:2024-09-17 15:03:41,350 - INFO [hooks.py:237 - after_run] - 2024-09-17 15:03:41.350441: step 347, prob_mean:0 = 0.10305227 (0.7 it/sec; 713.2 examples/sec)

### 排查 `EmbeddingLookup`

添加统计耗时直方图逻辑 `Histogram`, 对 `EmbeddingLookup` 中的步骤进行计时。

`EmbeddingLookup` 主要逻辑如下:
1. 每次请求查询一个 `batch_id` 中所有 `sparse` 特征对应到 `Embedding` 结果，同一个 `sparse` 特征对
应的 `Embedding` 结果相加，得到 `embedding_size` 个 `float` 参数。
2. 根据请求中的 `varname`, 用 `tokio::spawn` 创建并发查询任务，每个任务查询一个 `sparse` 特征的结果。
3. 等待所有查询结果，并按下标保存到相应的 `Vec` 中。

计时结果如下:

    2024-09-17T15:56:02 [INFO] sniper/util/src/histogram.rs:643 - PsEmbeddingLookup statistics in microseconds, total: 1000, p50: 6483757, p95: 7867860, p99: 16443152, max: 22269938
    2024-09-17T15:56:02 [INFO] sniper/util/src/histogram.rs:643 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 1320300, p50: 16473, p95: 145835, p99: 408183, max: 775997
    2024-09-17T15:56:02 [INFO] sniper/util/src/histogram.rs:643 - PsEmbeddingLookupDispatch statistics in microseconds, total: 1000, p50: 0, p95: 992, p99: 1062, max: 6999
    2024-09-17T15:56:02 [INFO] sniper/util/src/histogram.rs:643 - PsEmbeddingLookupWaiting statistics in microseconds, total: 1000, p50: 6483757, p95: 7867860, p99: 16443152, max: 22268938

各统计量含义如下:

- `PsEmbeddingLookup`: `EmbeddingLookup` 整体耗时，`p50` 约 `6.4s`。
- `PsEmbeddingLookupOneVariable`: 查询一个变量的耗时, `p50` 约 `0.016s`。
- `PsEmbeddingLookupDispatch`: 分发并行任务的时间，`p50` 约 `0.0009s`。
- `PsEmbeddingLookupWaiting`: 等待并行任务结束的时间，`p50` 约 `6.4s`。

可以看出，创建并发任务所占的时间比较少，每个变量查询的也不长。总共 `76` 个 `sparse` 变量，平均分配到 `2` 个 `ps`,
按照每个 `ps` 上 `38` 个 `sparse` 变量分别对分位数进行估计, 即按照单线程顺序执行来估计:

- `p50`: `0.016 * 38 = 0.608s`, 也只需要 `0.608s` 即可，与实际 `6.4s` 相差太大。
- `p95`: `0.145 * 38 = 5.51s`, 与总共耗时 `p95 7.8s` 更接近。
- `p99`: `0.408 * 38 = 15.504s`, 与总耗时 `p99 16.44s` 更接近。

从以上结果分析:
- `ps` 共有 `60` 核，按说有足够多的线程来同时执行并发任务。看起来像是实际单线程在运行。会不会实际就是单线程在运行？
- 看起来是各别比较慢的 `sparse` 变量导致耗时增加，可能是各别 `sparse` 特征 `sign` 明显比其他特征多，导致查询慢。

#### 实际并发数

搜索 `rust tonic thread number`, 发现了以下一些解释。

[Throughput doesn't increase with cores/threads count #1405](https://github.com/hyperium/tonic/issues/1405)
[Scalable server design in Rust with Tokio](https://medium.com/@fujita.tomonori/scalable-server-design-in-rust-with-tokio-4c81a5f350a3)

基本思路是 `tonic` 默认只用一个 `thread` 来监听和接收请求，可能成为瓶颈。解决方案是起多个进程，每个进程一个服务，
并且监听同一个端口。

这一方案也有问题，每个进程都必须维护所有的状态，并且 `hub` 和 `trainer` 发送的 `batch_id` 可能被发往不同的进程，
导致查询 `embedding` 参数时找不到 `batch_id`。

并且，从请求个数和请求的逻辑来看，主要的时间还是在处理逻辑上，接收请求应该不会成为瓶颈。问题可能还是出在处理逻辑慢。


#### 比较慢的 `sparse` 变量

统计同一个 `batch_id` 内个变量的耗时，结果如下:

    2024-09-17T17:00:50 [INFO] ps/src/request_handler.rs:367 - [get_embedding_lookup_result] start waiting, time: 2024-09-17 17:00:50.284341500 +08:00, batch_id: 4863390038324374882
    2024-09-17T17:00:50 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:50.300800182 +08:00, i: 0, batch_id: 4863390038324374882, varname: embedding_0, time spend micros: 16000, total_signs: 2048
    2024-09-17T17:00:50 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:50.449578125 +08:00, i: 1, batch_id: 4863390038324374882, varname: embedding_2, time spend micros: 165001, total_signs: 83035
    2024-09-17T17:00:50 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:50.449678826 +08:00, i: 2, batch_id: 4863390038324374882, varname: embedding_4, time spend micros: 96000, total_signs: 43390
    2024-09-17T17:00:50 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:50.449704750 +08:00, i: 3, batch_id: 4863390038324374882, varname: embedding_6, time spend micros: 21000, total_signs: 3072
    2024-09-17T17:00:50 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:50.449722026 +08:00, i: 4, batch_id: 4863390038324374882, varname: embedding_8, time spend micros: 18000, total_signs: 3066
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517529490 +08:00, i: 5, batch_id: 4863390038324374882, varname: embedding_11, time spend micros: 409003, total_signs: 270444
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517731333 +08:00, i: 6, batch_id: 4863390038324374882, varname: embedding_13, time spend micros: 20000, total_signs: 4552
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517778841 +08:00, i: 7, batch_id: 4863390038324374882, varname: embedding_15, time spend micros: 12000, total_signs: 2047
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517808971 +08:00, i: 8, batch_id: 4863390038324374882, varname: embedding_17, time spend micros: 12000, total_signs: 1124
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517824860 +08:00, i: 9, batch_id: 4863390038324374882, varname: embedding_19, time spend micros: 24000, total_signs: 7425
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517840591 +08:00, i: 10, batch_id: 4863390038324374882, varname: embedding_20, time spend micros: 17000, total_signs: 2751
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517856235 +08:00, i: 11, batch_id: 4863390038324374882, varname: embedding_22, time spend micros: 61000, total_signs: 12156
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517874026 +08:00, i: 12, batch_id: 4863390038324374882, varname: embedding_24, time spend micros: 21000, total_signs: 5116
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517889955 +08:00, i: 13, batch_id: 4863390038324374882, varname: embedding_26, time spend micros: 30000, total_signs: 7425
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517915780 +08:00, i: 14, batch_id: 4863390038324374882, varname: embedding_28, time spend micros: 27000, total_signs: 7425
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517934399 +08:00, i: 15, batch_id: 4863390038324374882, varname: embedding_31, time spend micros: 21000, total_signs: 2241
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517950302 +08:00, i: 16, batch_id: 4863390038324374882, varname: embedding_33, time spend micros: 12000, total_signs: 2085
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517965727 +08:00, i: 17, batch_id: 4863390038324374882, varname: embedding_35, time spend micros: 11000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517981459 +08:00, i: 18, batch_id: 4863390038324374882, varname: embedding_37, time spend micros: 10000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517996983 +08:00, i: 19, batch_id: 4863390038324374882, varname: embedding_39, time spend micros: 15000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518012031 +08:00, i: 20, batch_id: 4863390038324374882, varname: embedding_40, time spend micros: 11000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518027527 +08:00, i: 21, batch_id: 4863390038324374882, varname: embedding_42, time spend micros: 17000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518042633 +08:00, i: 22, batch_id: 4863390038324374882, varname: embedding_44, time spend micros: 65000, total_signs: 29506
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518057317 +08:00, i: 23, batch_id: 4863390038324374882, varname: embedding_46, time spend micros: 17000, total_signs: 3072
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518072034 +08:00, i: 24, batch_id: 4863390038324374882, varname: embedding_48, time spend micros: 12000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518087103 +08:00, i: 25, batch_id: 4863390038324374882, varname: embedding_51, time spend micros: 12000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518102370 +08:00, i: 26, batch_id: 4863390038324374882, varname: embedding_53, time spend micros: 26000, total_signs: 4090
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518118024 +08:00, i: 27, batch_id: 4863390038324374882, varname: embedding_55, time spend micros: 12000, total_signs: 1442
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518133122 +08:00, i: 28, batch_id: 4863390038324374882, varname: embedding_57, time spend micros: 24000, total_signs: 4060
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518148337 +08:00, i: 29, batch_id: 4863390038324374882, varname: embedding_59, time spend micros: 15000, total_signs: 2269
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518163460 +08:00, i: 30, batch_id: 4863390038324374882, varname: embedding_60, time spend micros: 11000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518180106 +08:00, i: 31, batch_id: 4863390038324374882, varname: embedding_62, time spend micros: 19000, total_signs: 3114
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518195344 +08:00, i: 32, batch_id: 4863390038324374882, varname: embedding_64, time spend micros: 12000, total_signs: 1024
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518210190 +08:00, i: 33, batch_id: 4863390038324374882, varname: embedding_66, time spend micros: 22000, total_signs: 3900
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518225156 +08:00, i: 34, batch_id: 4863390038324374882, varname: embedding_68, time spend micros: 29000, total_signs: 8793
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518240055 +08:00, i: 35, batch_id: 4863390038324374882, varname: embedding_71, time spend micros: 13000, total_signs: 1110
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518255571 +08:00, i: 36, batch_id: 4863390038324374882, varname: embedding_73, time spend micros: 17000, total_signs: 1396
    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.518270145 +08:00, i: 37, batch_id: 4863390038324374882, varname: embedding_75, time spend micros: 13000, total_signs: 1160


可以看出，等待耗时最多的是 `embedding_11`, 其 `total_signs` 也比其他变量多了几倍甚至十几倍。

    2024-09-17T17:00:56 [INFO] ps/src/request_handler.rs:377 - [get_embedding_lookup_result] get res, time: 2024-09-17 17:00:56.517529490 +08:00, i: 5, batch_id: 4863390038324374882, varname: embedding_11, time spend micros: 409003, total_signs: 270444

但是也有问题，从 `sparse` 变量内部耗时来看，`embedding_11` 花的时间是 `0.4s`, 但是从 `start waiting 2024-09-17 17:00:50.284341500` 
开始，到拿到 `embedding_11` 的时间 `2024-09-17 17:00:56.517529490`, 总共经过了 `6s`。即 `embedding_11` 的查询任务从 `tokio::spawn`
开始经过了 `6s` 才开始执行。在任务开始执行后会先获取变量的锁，然后再执行查询操作。

怀疑和锁有关，也可能和 `tokio` 调度算法有关。需要加日志看下每个任务开始和结束的时间。

作为对比，`embedding_2` 详细时间如下:

    2024-09-17T20:39:41 [INFO] ps/src/request_handler.rs:1011 - [Ps.feed_sample] feed sample, batch_id: 7451420526765469264, varname: embedding_2, i: 0, field: 2
    2024-09-17T20:39:48 [INFO] ps/src/request_handler.rs:340 - [Ps.get_embedding_lookup_result] in spawn, before get lock, time: 2024-09-17 20:39:48.266016852 +08:00, batch_id: 7451420526765469264, varname: embedding_2, field: 2
    2024-09-17T20:39:48 [INFO] ps/src/request_handler.rs:350 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 20:39:48.266123594 +08:00, batch_id: 7451420526765469264, varname: embedding_2, field: 2
    2024-09-17T20:39:48 [INFO] ps/src/embedding.rs:585 - [Embedding.embedding_lookup] start, time: 2024-09-17 20:39:48.266140228 +08:00, batch_id: 7451420526765469264, varname: embedding_2, field: 2
    2024-09-17T20:39:48 [INFO] ps/src/embedding.rs:663 - [Embedding.embedding_lookup] done, time: 2024-09-17 20:39:48.409103972 +08:00, batch_id: 7451420526765469264, varname: embedding_2, field: 2, total_signs: 85497
    2024-09-17T20:39:48 [INFO] ps/src/request_handler.rs:394 - [get_embedding_lookup_result] get res, time: 2024-09-17 20:39:48.409286937 +08:00, i: 1, batch_id: 7451420526765469264, varname: embedding_2, time spend micros: 143001, total_signs: 85497

`embedding_11` 详细时间如下:

    2024-09-17T20:39:48 [INFO] ps/src/request_handler.rs:1011 - [Ps.feed_sample] feed sample, batch_id: 7451420526765469264, varname: embedding_11, i: 0, field: 11

    2024-09-17T20:39:48 [INFO] ps/src/request_handler.rs:384 - [get_embedding_lookup_result] start waiting, time: 2024-09-17 20:39:48.266171136 +08:00, batch_id: 7451420526765469264

    2024-09-17T20:39:48 [INFO] ps/src/request_handler.rs:340 - [Ps.get_embedding_lookup_result] in spawn, before get lock, time: 2024-09-17 20:39:48.266220991 +08:00, batch_id: 7451420526765469264, varname: embedding_11, field: 11
    2024-09-17T20:39:53 [INFO] ps/src/request_handler.rs:350 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 20:39:53.913851175 +08:00, batch_id: 7451420526765469264, varname: embedding_11, field: 11
    2024-09-17T20:39:53 [INFO] ps/src/embedding.rs:585 - [Embedding.embedding_lookup] start, time: 2024-09-17 20:39:53.913931969 +08:00, batch_id: 7451420526765469264, varname: embedding_11, field: 11
    2024-09-17T20:39:54 [INFO] ps/src/embedding.rs:663 - [Embedding.embedding_lookup] done, time: 2024-09-17 20:39:54.271030970 +08:00, batch_id: 7451420526765469264, varname: embedding_11, field: 11, total_signs: 283731
    2024-09-17T20:39:54 [INFO] ps/src/request_handler.rs:394 - [get_embedding_lookup_result] get res, time: 2024-09-17 20:39:54.271209451 +08:00, i: 5, batch_id: 7451420526765469264, varname: embedding_11, time spend micros: 357003, total_signs: 283731

可以看出，`embedding_11` 仅在 `tokio::spawn` 分发任务后 `47 microseconds` 就开始执行，但是耗费了
`5.7s` 的时间获取锁。可以看出大部分时间都花在了获取锁的过程中。

加入获取锁的日志，`embedding_11` 结果如下, 而其他变量都在 `0ms` 所有。

    2024-09-17T21:06:13 [INFO] ps/src/request_handler.rs:351 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 21:06:13.368528759 +08:00, batch_id: 11558439139618182071, varname: embedding_11, field: 11, spend millis for lock: 5424
    2024-09-17T21:06:14 [INFO] ps/src/request_handler.rs:351 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 21:06:14.797484015 +08:00, batch_id: 14075177451965434773, varname: embedding_11, field: 11, spend millis for lock: 5484
    2024-09-17T21:06:16 [INFO] ps/src/request_handler.rs:351 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 21:06:16.137346500 +08:00, batch_id: 4882904890636045856, varname: embedding_11, field: 11, spend millis for lock: 5341
    2024-09-17T21:06:18 [INFO] ps/src/request_handler.rs:351 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 21:06:18.402225729 +08:00, batch_id: 401060307385547222, varname: embedding_11, field: 11, spend millis for lock: 6276

有没有办法不加锁？因为 `sparse` 参数很稀疏，两个 `batch` 更新的参数可能只有少部分是重合的。
因此如果不加锁，直接读取也写入，就会快很多。

#### 使用 `SyncUnsafeCell` ?

改为 `SyncUnsafeCell` 后速度快了很多, `embedding_11` 一次查询在 `1s` 左右，相比之前提高了 5 倍。

    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookup statistics in microseconds, total: 165300, p50: 874687, p95: 1034185, p99: 1048363, max: 1549997
    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 238383600, p50: 7973, p95: 150609, p99: 916234, max: 1230997
    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookupDispatch statistics in microseconds, total: 165300, p50: 0, p95: 999, p99: 1055, max: 10999
    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookupWaiting statistics in microseconds, total: 165300, p50: 874626, p95: 1034180, p99: 1048362, max: 1549997
    
`lock` 获取时间为 `0`:

    2024-09-17T22:09:44 [INFO] ps/src/request_handler.rs:341 - [Ps.get_embedding_lookup_result] in spawn, before get lock, time: 2024-09-17 22:09:44.968751761 +08:00, batch_id: 8607674304782930109, varname: embedding_11, field: 11
    2024-09-17T22:09:44 [INFO] ps/src/request_handler.rs:351 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 22:09:44.968813013 +08:00, batch_id: 8607674304782930109, varname: embedding_11, field: 11, spend millis for lock: 0

`trainer` 训练速度也有提高，最快能达到 `150758.8 examples/sec`。

    [1,0]<stderr>:2024-09-17 22:17:19,147 - INFO [hooks.py:269 - after_run] - 2024-09-17 22:17:19.147784: step 8097, auc = 0.7665 (147.2 it/sec; 150758.8 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,147 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.147870: step 8097, xentropy_mean:0 = 0.25924635 (148.1 it/sec; 151679.9 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,147 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.147918: step 8097, prob_mean:0 = 0.09372127 (148.2 it/sec; 151738.8 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,147 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.147958: step 8097, real_mean:0 = 0.08007812 (148.2 it/sec; 151771.0 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,161 - INFO [hooks.py:269 - after_run] - 2024-09-17 22:17:19.161423: step 8098, auc = 0.7665 (73.3 it/sec; 75077.7 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,161 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.161513: step 8098, xentropy_mean:0 = 0.23164335 (73.3 it/sec; 75056.7 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,161 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.161560: step 8098, prob_mean:0 = 0.09266146 (73.3 it/sec; 75063.2 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,161 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.161602: step 8098, real_mean:0 = 0.07421875 (73.3 it/sec; 75055.3 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,259 - INFO [hooks.py:269 - after_run] - 2024-09-17 22:17:19.259786: step 8099, auc = 0.7665 (10.2 it/sec; 10410.6 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,259 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.259907: step 8099, xentropy_mean:0 = 0.24669521 (10.2 it/sec; 10407.1 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,259 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.259956: step 8099, prob_mean:0 = 0.08644607 (10.2 it/sec; 10407.0 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,260 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.259999: step 8099, real_mean:0 = 0.08496094 (10.2 it/sec; 10406.7 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,274 - INFO [hooks.py:269 - after_run] - 2024-09-17 22:17:19.274603: step 8100, auc = 0.7665 (67.5 it/sec; 69108.7 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,274 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.274717: step 8100, xentropy_mean:0 = 0.32623392 (67.5 it/sec; 69143.2 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,274 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.274764: step 8100, prob_mean:0 = 0.08997593 (67.5 it/sec; 69148.8 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,274 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:17:19.274804: step 8100, real_mean:0 = 0.11718750 (67.5 it/sec; 69164.3 examples/sec)
    [1,0]<stderr>:2024-09-17 22:17:19,274 - INFO [trainer.py:300 - train] - current lr: 0.050000


`trainer` 耗时统计如下, 虽然 `OpsEmbeddingLookup` 还是需要 `1s` 左右，但是比之前快了很多。

    [1,0]<stderr>:I0917 22:07:37.851832   658 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 307  P50: 1024865.771812  P95: 1081070.000000  P99: 1081070.000000  Max: 1081070.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 306  P50: 41816.949153  P95: 49752.203390  P99: 62416.000000  Max: 62416.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 306  P50: 12345.180723  P95: 21159.493671  P99: 27022.000000  Max: 27022.000000

### 增加资源 

`ps` 由 `2` 个增加到 `8` 个，`hub` 由 `1` 增加到 `2` 个。速度和 `2 ps` 差不多。

    [1,0]<stderr>:2024-09-17 22:56:14,881 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:14.881023: step 13318, xentropy_mean:0 = 0.36226147 (145.3 it/sec; 148820.8 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:14,881 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:14.881066: step 13318, prob_mean:0 = 0.11309537 (145.4 it/sec; 148893.0 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:14,881 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:14.881103: step 13318, real_mean:0 = 0.14648438 (145.4 it/sec; 148924.0 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:14,890 - INFO [hooks.py:269 - after_run] - 2024-09-17 22:56:14.890717: step 13319, auc = 0.7707 (102.0 it/sec; 104449.6 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:14,890 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:14.890827: step 13319, xentropy_mean:0 = 0.29208055 (102.0 it/sec; 104444.5 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:14,890 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:14.890872: step 13319, prob_mean:0 = 0.11844525 (102.0 it/sec; 104431.8 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:14,890 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:14.890908: step 13319, real_mean:0 = 0.10351562 (102.0 it/sec; 104434.4 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:15,042 - INFO [hooks.py:269 - after_run] - 2024-09-17 22:56:15.042905: step 13320, auc = 0.7707 (6.6 it/sec; 6728.6 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:15,043 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:15.043075: step 13320, xentropy_mean:0 = 0.33678800 (6.6 it/sec; 6725.9 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:15,043 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:15.043124: step 13320, prob_mean:0 = 0.11866179 (6.6 it/sec; 6725.7 examples/sec)
    [1,0]<stderr>:2024-09-17 22:56:15,043 - INFO [hooks.py:237 - after_run] - 2024-09-17 22:56:15.043162: step 13320, real_mean:0 = 0.12109375 (6.6 it/sec; 6725.6 examples/sec)

查看资源利用率监控，发现 `hub` cpu 利用率都不高，只有 `20%`, `ps` 不均匀，`ps 0` cpu 利用率 `59%`,
其他 `ps` 只有 `10%` 左右。

### 实验 `auto_shard`


