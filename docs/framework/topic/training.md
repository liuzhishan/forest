# training

## ps 添加 lru

使用 `VecDeque` 和 `DashMap` 结合的方式实现 `lru`。

## `c++ 版本` `batch` 数据速度

相同的资源情况下速度能到 `243848 examples/sec`, 特征一样，不过数据是拼好的 `batch`。

    [1,0]<stderr>:2024-09-21 10:28:35,027 - INFO [hooks.py:465 - after_run] - step 9600, auc = 0.8267, loss = 0.2872257071, prob_mean: 0.109512, real_mean: 0.125947, (238.1 it/sec; 243848.0 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:35,456 - INFO [hooks.py:465 - after_run] - step 9700, auc = 0.8267, loss = 0.2872743337, prob_mean: 0.109527, real_mean: 0.125963, (232.8 it/sec; 238371.1 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:35,889 - INFO [hooks.py:465 - after_run] - step 9800, auc = 0.8267, loss = 0.2872754446, prob_mean: 0.109525, real_mean: 0.125965, (231.1 it/sec; 236693.9 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:36,313 - INFO [hooks.py:465 - after_run] - step 9900, auc = 0.8267, loss = 0.2872792002, prob_mean: 0.109525, real_mean: 0.125968, (235.6 it/sec; 241291.7 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:36,739 - INFO [hooks.py:465 - after_run] - step 10000, auc = 0.8266, loss = 0.2873227067, prob_mean: 0.109530, real_mean: 0.125983, (234.9 it/sec; 240575.3 examples/sec)
    
    
查看耗时监控如下

    [1,0]<stdout>:[2024-09-21 10:27:34.875] [info] [run_status.cc:72] [RunStatus] OpsEmbeddingLookup statistics => count: 757  P50: 27567.438692  P95: 32672.547684  P99: 39838.000000  Max: 39838.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 758  P50: 7128.000000  P95: 9629.400000  P99: 9851.746667  Max: 10258.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 757  P50: 3972.443182  P95: 302995.000000  P99: 302995.000000  Max: 302995.000000

    [1,0]<stdout>:[2024-09-21 10:28:04.876] [info] [run_status.cc:72] [RunStatus] OpsEmbeddingLookup statistics => count: 2612  P50: 28552.437902  P95: 44828.169014  P99: 48997.558685  Max: 50970.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 2616  P50: 4062.656904  P95: 6443.281853  P99: 11396.500000  Max: 298651.000000
    [1,0]<stdout>:[2024-09-21 10:28:34.876] [info] [run_status.cc:72] [RunStatus] OpsEmbeddingLookup statistics => count: 6952  P50: 28703.185841  P95: 45343.420016  P99: 48266.000000  Max: 48266.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 6956  P50: 4365.349076  P95: 6430.344002  P99: 7487.526316  Max: 32105.000000
    
重新跑了一遍，速度有些差异，可能和集群跑的任务有关

速度

    [1,0]<stderr>:2024-09-24 20:51:40,494 - INFO [hooks.py:465 - after_run] - step 18900, auc = 0.8761, loss = 0.1924512684, prob_mean: 0.121220, real_mean: 0.101562, (193.4 it/sec; 198036.4 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:41,002 - INFO [hooks.py:465 - after_run] - step 19000, auc = 0.8761, loss = 0.2136482298, prob_mean: 0.139329, real_mean: 0.103516, (197.2 it/sec; 201916.2 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:41,503 - INFO [hooks.py:465 - after_run] - step 19100, auc = 0.8761, loss = 0.1577771753, prob_mean: 0.084985, real_mean: 0.074219, (199.4 it/sec; 204152.6 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:42,020 - INFO [hooks.py:465 - after_run] - step 19200, auc = 0.8762, loss = 0.1820228547, prob_mean: 0.102443, real_mean: 0.079102, (193.3 it/sec; 197945.0 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:42,534 - INFO [hooks.py:465 - after_run] - step 19300, auc = 0.8762, loss = 0.1943198144, prob_mean: 0.106581, real_mean: 0.090820, (194.6 it/sec; 199262.9 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:43,046 - INFO [hooks.py:465 - after_run] - step 19400, auc = 0.8762, loss = 0.1421866119, prob_mean: 0.096939, real_mean: 0.052734, (195.3 it/sec; 199998.4 examples/sec)


`trainer` 监控

    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 5508  P50: 27394.852941  P95: 44173.026316  P99: 56622.222222  Max: 363990.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 5512  P50: 9640.076336  P95: 13799.932461  P99: 19276.981132  Max: 23315.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 5513  P50: 3959.264480  P95: 6492.352103  P99: 9530.824561  Max: 27105.000000

`ps` 耗时监控如下


    PsFeedSample statistics => count: 85098  P50: 85.895506  P95: 818.502277  P99: 1205.698832  Max: 2756.000000
    PsEmbeddingLookup statistics => count: 5684  P50: 6928.621908  P95: 9640.070671  P99: 9881.088339  Max: 19403.000000
    PsPushGrad statistics => count: 5683  P50: 8465.338310  P95: 12913.279570  P99: 13824.337243  Max: 24890.000000
    PsFeedCached statistics => count: 85111  P50: 332.000000  P95: 340.000000  P99: 340.000000  Max: 340.000000
    PsLookupCached statistics => count: 85061  P50: 27.978623  P95: 31.000000  P99: 31.000000  Max: 31.000000

`hub` 耗时监控如下

    HubReadSample statistics => count: 50  P50: 4.538462  P95: 28.000000  P99: 210000.000000  Max: 223220.000000
    HubNext statistics => count: 214  P50: 64375.000000  P95: 5302804.000000  P99: 5302804.000000  Max: 5302804.000000
    HubFeedSample statistics => count: 132  P50: 6780.000000  P95: 12658.181818  P99: 16035.000000  Max: 16035.000000
    HubCountSamplePos statistics => count: 27416  P50: 1.000000  P95: 1.000000  P99: 1.000000  Max: 1.000000
    HubCountSampleNeg statistics => count: 190724  P50: 1.000000  P95: 1.000000  P99: 1.000000  Max: 1.000000
    HubBatchProcessor statistics => count: 194  P50: 3030.434783  P95: 7634394.000000  P99: 7634394.000000  Max: 7634394.000000


### 训练资源

- `trainer`: 1 T4 GPU, 30 core, 60G Mem, 1 副本。
- `ps`: 0 GPU, 60 core, 200G Mem, 4 副本。
- `hub`: 0 GPU, 20 core, 50G Mem, 4 副本。


## 训练速度慢

### 第一轮排查

#### 现在是读取单条样本，`hub` 里拼 `batch`

之前是离线拼好的 `batch`，会导致一些速度差异，`batch` 数据很快。

#### 是否和 `lru` 有关

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


#### `EmbeddingLookup` 时对整个 `EmbeddingManager` 加锁，锁的粒度太粗。

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


##### 怎样将锁加在每个 `Embedding` 上 ?

###### 使用 `Vec` 保存 `Embedding` ?

###### `Rayon` ?

###### `Arc<Vec>` ?

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


#### `hash` 函数

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

#### 排查 `EmbeddingLookup` 耗时

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

#### 使用 `SyncUnsafeCell` 不加锁

改为 `SyncUnsafeCell` 后速度快了很多, `embedding_11` 一次查询在 `1s` 左右，相比之前提高了 5 倍。

    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookup statistics in microseconds, total: 165300, p50: 874687, p95: 1034185, p99: 1048363, max: 1549997
    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 238383600, p50: 7973, p95: 150609, p99: 916234, max: 1230997
    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookupDispatch statistics in microseconds, total: 165300, p50: 0, p95: 999, p99: 1055, max: 10999
    2024-09-17T22:13:13 [INFO] /share/ad/liuzhishan/klearn_debug/sniper/util/src/histogram.rs:643 - PsEmbeddingLookupWaiting statistics in microseconds, total: 165300, p50: 874626, p95: 1034180, p99: 1048362, max: 1549997
    
`lock` 获取时间为 `0`:

    2024-09-17T22:09:44 [INFO] ps/src/request_handler.rs:341 - [Ps.get_embedding_lookup_result] in spawn, before get lock, time: 2024-09-17 22:09:44.968751761 +08:00, batch_id: 8607674304782930109, varname: embedding_11, field: 11
    2024-09-17T22:09:44 [INFO] ps/src/request_handler.rs:351 - [Ps.get_embedding_lookup_result] in spawn, after get lock, time: 2024-09-17 22:09:44.968813013 +08:00, batch_id: 8607674304782930109, varname: embedding_11, field: 11, spend millis for lock: 0

`trainer` 训练速度也有提高，最快能达到 `150758.8 examples/sec`, 但是不稳定。

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

#### 增加资源 

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

#### 替换最慢的特征

`embedding_11` 比其他特征 `sign` 个数多了一个数量级, 将其替换为其他同样规模的特征。`OpsEmbeddingLookup` 耗时确实变小
了，只有原来的 `1/4`, 但是训练速度能到 `3.7万每秒`, 还是有差距。

奇怪的是还会报 `feed queue not rich`, 说明 `trainer` 的 `queue` 一直是空的。瓶颈还是在取 `embedding`。

    [1,0]<stderr>:2024-09-19 00:19:11,793 - INFO [hooks.py:269 - after_run] - 2024-09-19 00:19:11.793460: step 15900, auc = 0.7428 (36.5 it/sec; 37378.3 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:11,793 - INFO [hooks.py:237 - after_run] - 2024-09-19 00:19:11.793641: step 15900, xentropy_mean:0 = 0.39009726 (36.5 it/sec; 37378.1 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:11,793 - INFO [hooks.py:237 - after_run] - 2024-09-19 00:19:11.793699: step 15900, prob_mean:0 = 0.11534940 (36.5 it/sec; 37378.3 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:11,793 - INFO [hooks.py:237 - after_run] - 2024-09-19 00:19:11.793741: step 15900, real_mean:0 = 0.14453125 (36.5 it/sec; 37378.4 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:11,793 - INFO [trainer.py:300 - train] - current lr: 0.050000
    [1,0]<stderr>:INFO:tensorflow:global_step/sec: 36.6389
    [1,0]<stderr>:2024-09-19 00:19:11,856 - INFO [basic_session_run_hooks.py:692 - _log_and_record] - global_step/sec: 36.6389
    [1,0]<stderr>:I0919 00:19:13.452296   661 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1067  P50: 206292.481977  P95: 245851.699279  P99: 249368.074150  Max: 273263.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1067  P50: 42420.000000  P95: 49681.558567  P99: 59878.000000  Max: 59878.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1067  P50: 12311.558219  P95: 20680.139373  P99: 21869.825784  Max: 56117.000000
    [1,0]<stderr>:2024-09-19 00:19:14,572 - INFO [hooks.py:269 - after_run] - 2024-09-19 00:19:14.572660: step 16000, auc = 0.7427 (36.0 it/sec; 36845.1 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:14,572 - INFO [hooks.py:237 - after_run] - 2024-09-19 00:19:14.572830: step 16000, xentropy_mean:0 = 0.35552624 (36.0 it/sec; 36845.3 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:14,572 - INFO [hooks.py:237 - after_run] - 2024-09-19 00:19:14.572884: step 16000, prob_mean:0 = 0.12149657 (36.0 it/sec; 36845.3 examples/sec)
    [1,0]<stderr>:2024-09-19 00:19:14,572 - INFO [hooks.py:237 - after_run] - 2024-09-19 00:19:14.572929: step 16000, real_mean:0 = 0.13476562 (36.0 it/sec; 36845.3 examples/sec)


#### 实验 `auto_shard`


#### 底层实现

##### tokio scheduler

[Making the Tokio scheduler 10x faster](https://tokio.rs/blog/2019-10-scheduler)
[How Tokio schedule tasks: A hard Lesson learnt](https://rustmagazine.org/issue-4/how-tokio-schedule-tasks/)

This is all to say the obvious: avoid cross thread synchronization as much as possible because it is slow.

Scheduler strategy:
- one queue, many processors.
- concurrency and mechanical sympathy.
- many processor, each with their own queue.
- work-stealing scheduler.


#### lock free

[Low Latency in Rust with Lock-Free Data Structures](https://www.linkedin.com/pulse/low-latency-rust-lock-free-data-structures-luis-soares-m-sc--va5xf)
[Exploring lock-free Rust 1: Locks](https://morestina.net/blog/742/exploring-lock-free-rust-1-locks)
[lockfree](https://crates.io/crates/lockfree)


### 经过第一轮排查与优化后的理论速度

    [1,0]<stderr>:I0919 00:19:13.452296   661 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1067  P50: 206292.481977  P95: 245851.699279  P99: 249368.074150  Max: 273263.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1067  P50: 42420.000000  P95: 49681.558567  P99: 59878.000000  Max: 59878.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1067  P50: 12311.558219  P95: 20680.139373  P99: 21869.825784  Max: 56117.000000
    
训练时获取 `EmbeddingLookup` 是同步操作，即每个 `batch` 都需要获取所有 `sparse` 特征的 `embedding sum`
结果后再继续后面的训练，按以上耗时统计分析，`PushGrad` 和 `ReadSample` 相比 `EmbeddingLookup` 很小，
因此瓶颈是 `EmbeddingLookup` 环节。假如按 `p95` 估计每个 `batch` 需要 `0.24s`, 则每个线程每秒可获取 `4`
个 `batch`, 目前 `trainer` 默认有 `10` 个线程进行预取，预取队列大小为 `8`, 因此理论上美妙可以获取 `40` 个
`batch`, `batch_size` 按 `1024` 计数，则每秒可获取 `40960` 条样本，而训练日志中的速度为 `37000/s`, 可以认为
符合预期。即目前的速度已接近理论极限。

每个 `batch` 的 `EmbeddingLookup` 返回结果大小约为 `batch_size * sparse_feature_count * embedding_size` 个
`float`, 按 `76` 个 `sparse` 估计，结果为

    1024 * 76 * 16 = 1245184 = 1m
    
`1m` 对于带宽来说并不大，因此主要时间还是花在 `ps` 计算上，而不是网络通信上。说明 `trainer` 主要还是在
等待结果。因此如果加大 `trainer` 预取线程，训练速度应该会线性增长。

将预取线程从 `10` 增加到 `20` 后，速度甚至有点降低, 有点奇怪。需要加点日志看下 `feed_queue` 里从 `ReadSample`
到 `push` 到队列中间各个步骤的耗时。


    [1,0]<stderr>:I0919 10:11:30.807346   786 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 906  P50: 200574.712644  P95: 247436.781609  P99: 277085.000000  Max: 277085.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 906  P50: 41939.000000  P95: 49440.337079  P99: 60843.750000  Max: 70957.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 903  P50: 12515.517241  P95: 29311.333333  P99: 64178.571429  Max: 369546.000000
    [1,0]<stderr>:2024-09-19 10:11:32,482 - INFO [hooks.py:269 - after_run] - 2024-09-19 10:11:32.482195: step 4200, auc = 0.7472 (29.0 it/sec; 29695.3 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:32,482 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:32.482437: step 4200, xentropy_mean:0 = 0.25089473 (29.0 it/sec; 29694.6 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:32,482 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:32.482512: step 4200, prob_mean:0 = 0.10106875 (29.0 it/sec; 29694.4 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:32,482 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:32.482572: step 4200, real_mean:0 = 0.07910156 (29.0 it/sec; 29694.2 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:32,482 - INFO [trainer.py:300 - train] - current lr: 0.050000
    [1,0]<stderr>:INFO:tensorflow:global_step/sec: 29.9825
    [1,0]<stderr>:2024-09-19 10:11:32,489 - INFO [basic_session_run_hooks.py:692 - _log_and_record] - global_step/sec: 29.9825
    [1,0]<stderr>:2024-09-19 10:11:35,838 - INFO [hooks.py:269 - after_run] - 2024-09-19 10:11:35.838329: step 4300, auc = 0.7470 (29.8 it/sec; 30511.3 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:35,838 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:35.838490: step 4300, xentropy_mean:0 = 0.27898747 (29.8 it/sec; 30512.0 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:35,838 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:35.838553: step 4300, prob_mean:0 = 0.09167473 (29.8 it/sec; 30512.1 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:35,838 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:35.838593: step 4300, real_mean:0 = 0.09375000 (29.8 it/sec; 30512.3 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:35,838 - INFO [trainer.py:300 - train] - current lr: 0.050000
    [1,0]<stderr>:INFO:tensorflow:global_step/sec: 29.6871
    [1,0]<stderr>:2024-09-19 10:11:35,858 - INFO [basic_session_run_hooks.py:692 - _log_and_record] - global_step/sec: 29.6871
    [1,0]<stderr>:2024-09-19 10:11:38.229491: W trainer/core/operators/kernels/feed_queue.cc:67] Trainer feed queue is not rich. trainer_id: 0, queue.size(): 0, training will bound at prefetch,  you can check trainer network and hub / ps resource.
    [1,0]<stderr>:2024-09-19 10:11:39,212 - INFO [hooks.py:269 - after_run] - 2024-09-19 10:11:39.212078: step 4400, auc = 0.7467 (29.6 it/sec; 30352.1 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:39,212 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:39.212414: step 4400, xentropy_mean:0 = 0.30004069 (29.6 it/sec; 30350.4 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:39,212 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:39.212470: step 4400, prob_mean:0 = 0.10385576 (29.6 it/sec; 30350.5 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:39,213 - INFO [hooks.py:237 - after_run] - 2024-09-19 10:11:39.213134: step 4400, real_mean:0 = 0.10449219 (29.6 it/sec; 30344.9 examples/sec)
    [1,0]<stderr>:2024-09-19 10:11:39,213 - INFO [trainer.py:300 - train] - current lr: 0.050000


### 第二轮排查 

#### `feed_queue` 中各步骤的耗时

将请求 `hub`, `ReadSample`, `EmbeddingLookup`, `push to queue` 等步骤都加上日志, `thread_id 0` 结果如下, 可以
看出，大部分 `ReadSample` 请求都没有获取到数据。而如果从 `hub` 获取到数据，后面的 `EmbeddingLookup` 耗时与监控一致。


说明瓶颈是 `hub` ?


    [1,0]<stderr>:2024-09-20 13:48:34.646883: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.646898: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.650838: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.650848: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.654586: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.654607: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.658092: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.658129: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.661794: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.661806: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.665330: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.665343: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.669174: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.669191: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.672761: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.672774: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.676267: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.676285: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.679938: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.679953: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.683477: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.683485: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.687396: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.687403: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.690899: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.690909: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.694596: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.694602: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.698400: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.698407: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.701864: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.701871: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.705676: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.705685: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.709361: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.709374: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.713283: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.713297: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.716862: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.716869: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.726340: I trainer/core/operators/kernels/feed_queue.cc:258] trainer thread id: 0, after read sample, start check dim
    [1,0]<stderr>:2024-09-20 13:48:34.726352: I trainer/core/operators/kernels/feed_queue.cc:277] trainer thread id: 0, after check dim, start embedding lookup
    [1,0]<stderr>:2024-09-20 13:48:34.887918: I trainer/core/operators/kernels/feed_queue.cc:451] trainer thread id: 0, after embedding lookup, start check embedding result
    [1,0]<stderr>:2024-09-20 13:48:34.887940: I trainer/core/operators/kernels/feed_queue.cc:468] trainer thread id: 0, after check embedding result, start push to queue
    [1,0]<stderr>:2024-09-20 13:48:34.898154: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-20 13:48:34.898168: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.898173: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.907967: I trainer/core/operators/kernels/feed_queue.cc:258] trainer thread id: 0, after read sample, start check dim
    [1,0]<stderr>:2024-09-20 13:48:34.907981: I trainer/core/operators/kernels/feed_queue.cc:277] trainer thread id: 0, after check dim, start embedding lookup
    [1,0]<stderr>:2024-09-20 13:48:35.106380: I trainer/core/operators/kernels/feed_queue.cc:451] trainer thread id: 0, after embedding lookup, start check embedding result
    [1,0]<stderr>:2024-09-20 13:48:35.106403: I trainer/core/operators/kernels/feed_queue.cc:468] trainer thread id: 0, after check embedding result, start push to queue


#### 增加 `hub` 个数

从 `2` 增加到 `4`。

有一点上涨，能到 `38400 exampls/sec`, 但是与预期还是有差距。

    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:269 - after_run] - 2024-09-21 09:21:46.800022: step 1300, auc = 0.6715 (37.5 it/sec; 38399.8 examples/sec)
    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:237 - after_run] - 2024-09-21 09:21:46.800209: step 1300, xentropy_mean:0 = 0.30443680 (37.5 it/sec; 38400.1 examples/sec)
    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:237 - after_run] - 2024-09-21 09:21:46.800266: step 1300, prob_mean:0 = 0.08712210 (37.5 it/sec; 38400.1 examples/sec)
    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:237 - after_run] - 2024-09-21 09:21:46.800310: step 1300, real_mean:0 = 0.09960938 (37.5 it/sec; 38400.1 examples/sec)
    

查看 `after push to queue` 日志结果如下:

    # grep '09:22:27' log/dsp_ctr_lzs_test_v6_2024_09_21_09_19_17.log | grep 'after push to queue'

    [1,0]<stderr>:2024-09-21 09:22:27.053202: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.064763: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.076186: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.080873: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.081193: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.088784: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.102404: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.165555: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.166353: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.200455: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 7, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.305205: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.310202: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.333078: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.339847: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.342972: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.363182: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.373962: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.384125: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.431764: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.485752: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 7, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.565343: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.572960: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.607563: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.608794: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.611064: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.615670: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.633463: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.638674: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.705027: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.741057: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 7, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.855039: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.882023: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.884056: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.888113: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.888182: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.898029: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.898356: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.937386: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.962224: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue


    grep '09:22:27' log/dsp_ctr_lzs_test_v6_2024_09_21_09_19_17.log | grep 'after push to queue' | wc -l
    
    39
    
去掉 `trainer` 日志，能到 `49772 examples/sec`。

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1440  P50: 209488.636364  P95: 246306.818182  P99: 249579.545455  Max: 264189.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1440  P50: 42283.000000  P95: 49962.800875  P99: 70073.529412  Max: 91791.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1438  P50: 13212.349914  P95: 25916.822430  P99: 31830.093458  Max: 39568.000000

    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:264 - after_run] - 2024-09-21 14:08:46.568055: step 3600, auc = 0.6713 (48.6 it/sec; 49772.2 examples/sec)
    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:232 - after_run] - 2024-09-21 14:08:46.568233: step 3600, xentropy_mean:0 = 0.29283547 (48.6 it/sec; 49772.1 examples/sec)
    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:232 - after_run] - 2024-09-21 14:08:46.568289: step 3600, prob_mean:0 = 0.08826549 (48.6 it/sec; 49772.0 examples/sec)
    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:232 - after_run] - 2024-09-21 14:08:46.568333: step 3600, real_mean:0 = 0.08984375 (48.6 it/sec; 49772.0 examples/sec)


#### 增加 `trainer` 预取线程数

将 `trainer` 预取线程数从 `10` 增加到 `20`, 速度和之前差不多, 离预期还有距离。

    [1,0]<stderr>:2024-09-21 09:29:01,576 - INFO [hooks.py:269 - after_run] - 2024-09-21 09:29:01.576279: step 15700, auc = 0.6591 (37.0 it/sec; 37902.6 examples/sec)
    
耗时监控如下, 可以看出, `EmbeddingLookup` 耗时增加约 `1` 倍, `PushGrad` 基本不变, `ReadSample` `p50` 基本不变，但是 `p95` 增加约 `1` 倍。

    [1,0]<stderr>:I0921 09:38:38.752875   786 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1316  P50: 462387.931034  P95: 559386.206897  P99: 568008.275862  Max: 576471.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1318  P50: 41835.173502  P95: 49786.829653  P99: 68785.714286  Max: 75602.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1316  P50: 15726.872247  P95: 44936.956522  P99: 49800.434783  Max: 92753.000000
    [1,0]<stderr>:I0921 09:39:08.753228   786 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1313  P50: 462747.205503  P95: 559274.720550  P99: 567854.944110  Max: 568404.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1308  P50: 41818.000000  P95: 49607.075472  P99: 65916.666667  Max: 68691.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1310  P50: 13814.439655  P95: 31207.865169  P99: 41405.555556  Max: 99806.000000

资源利用率:

- `hub`: cpu 利用率约 `30%`。
- `ps`: `ps0` 利用率约为 `60%`,  不均匀，其他由于监控问题看不到。

### 经过第二轮排查与优化后与 `c++` 版本对比

`c++` 版本速度能到 `240000 examples/sec`, `rust` 版本约为 `40000 examples/sec`, 相差 `6` 倍。

对比 `rust` `38000 examples/sec` 版本统计耗时

    [1,0]<stderr>:OpsReadSample statistics => count: 1108  P50: 9651.190476  P95: 14241.509434  P99: 20931.320755  Max: 46424.000000
    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1106  P50: 261759.868421  P95: 310601.000000  P99: 310601.000000  Max: 310601.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1107  P50: 41825.000000  P95: 49584.461967  P99: 65456.896552  Max: 71762.000000

`c++` 版本耗时统计如下。

    [1,0]<stdout>:OpsReadSample statistics => count: 757  P50: 3972.443182  P95: 302995.000000  P99: 302995.000000  Max: 302995.000000
    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 757  P50: 27567.438692  P95: 32672.547684  P99: 39838.000000  Max: 39838.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 758  P50: 7128.000000  P95: 9629.400000  P99: 9851.746667  Max: 10258.000000
    

`c++` 版本与 `rust` 版本对比如下。
备注: 计时单位为微妙。

| 统计项          | rust p50 | rust p95 | c++ p50 | c++ p95 | p50 speedup | p95 speedup |
|-----------------|----------|----------|---------|---------|-------------|-------------|
| ReadSample      | 9,651    | 14,241   | 3,972   | 302,995 | 2.43x       | 0.05x       |
| EmbeddingLookup | 261,759  | 310,601  | 27,567  | 32,672  | 9.50x       | 9.51x       |
| PushGrad        | 41,825   | 49,584   | 7,128   | 9,626   | 5.87x       | 5.15x       |

从主要的三项耗时统计来看，`ReadSample` 有 `2.43x` 的差距，可能是因为 `c++` 版本读取的是 `batch` 数据,
这种数据格式基本就是二进制的格式，直接按 `byte` 进行读取，速度非常快，而 `rust` 读取的是 `proto` 格式的
单条样本格式，还需要再拼 `batch`，速度会有差异。后面会尝试一种更新的可能比 `batch` 数据更好的格式。

更大的差异在于 `ps` 上的速度差异。从耗时可以看出，不管是 `c++` 版本还是 `rust` 版本，主要瓶颈都在
`EmbeddingLookup` 这一步。`c++` 版本快了 `9` 倍多。`ps` 上涉及的计算逻辑比较多，而且涉及到各种多线程
并发访问的问题，可能还是和具体实现有关系。

理论上 `rust` 的性能应该能和 `c++` 到统一水平, 实际 `rust` 差这么多有点出乎意料。

从实现上看差异比较大的主要是三个: 

- 保存参数的实现：
    - `rust`: 采用 `Vec<DashMap>` 来保存参数，固定分片，`shard` 会有锁，内部保存的数据是 `lock free`,
    并且 `map` 采用 `swiss table` 的思路实现，利用 `simd` 加速。
    - `c++`: 基于 `circular buffer` 自己实现了 `lru`，结合 `folly concurrent hashmap` 手动管理内存。
    这种实现会预分配一大片内存用户保存参数，不会有 `malloc` 和 `delete` 带来的问题。保存的数据也是
    `lock free`。
- `scheduler`:
    - `rust`: 直接使用了 `tokio` 提供的 `scheduler` 来调度并发任务。仔细看了 `tokio scheduler` 的原理，
    感觉和 `c++` 中自己实现的 `scheduler` 思路类似， 都是多个 `processors`, 每个都有自己的 `queue`,
    而且 `tokio` 的实现更完善一些，不应该有这么大差异。
    - `c++`: 自己实现的 `scheduler`。 思路是多个 `processors`, 每个都有自己的 `queue`。
- `grpc`:
    - `rust`: 直接使用 `tonic` 框架。
    - `c++`: 使用的是 `c++ grpc`, 同时封装了一些 `zero copy` 的优化。考虑到主要耗时的计算逻辑是
    `Embedding` 参数的查询与更新，`grpc` 请求 `qps` 并不是很高，因此 `grpc` 请求这一因素应该影响不大。

待仔细研究。

TODO:
- 仔细对比 `ps` 中的耗时监控。
- 仔细对比 `map` 的单线程读写性能。
- 仔细对比 `map` 的多线程读写性能。

### 第三轮排查

#### `simd` 加速

突然想到 `c++` 版本中 `ps` `EmbeddingLookup` 与 `PushGrad` 都用了 `simd` 加速来计算 `sum`,  `sum` 是
一个高频操作。利用 `__mm256` 系列指令同时计算 `8` 个 `float` 的运算结果。理论上有 `8` 倍的加速，好
像和速度差异差不多。如果用 `__mm512` 系列则能够同时计算 `16` 个 `float`。

`c++` 中实现如下

    void Sum8FloatValues(float* dst, const float* src) {
      __m256 m_dst = _mm256_loadu_ps(dst);
      __m256 m_src = _mm256_loadu_ps(src);
      __m256 m_val = _mm256_add_ps(m_dst, m_src);
      _mm256_storeu_ps(dst, m_val);
    }

    void Sum(const float* src, float* dst, int32_t n) {
      int32_t c = n / 8;
      int32_t offset = 0;

      for (int32_t i = 0; i < c; ++i) {
        Sum8FloatValues(dst + offset, src + offset);
        offset += 8;
      }

      while (offset < n) {
        dst[offset] += src[offset];
        ++offset;
      }
    }


参考:
- [Nine Rules for SIMD Acceleration of Your Rust Code (Part 1)](https://towardsdatascience.com/nine-rules-for-simd-acceleration-of-your-rust-code-part-1-c16fe639ce21)
- [Nine Rules for SIMD Acceleration of your Rust Code (Part 2)](https://towardsdatascience.com/nine-rules-for-simd-acceleration-of-your-rust-code-part-2-6a104b3be6f3)

使用 `simd` 需要 `rust nightly`, 按如下步骤安装

    rustup install nightly
    rustup update nightly
    rustup override set nightly
    
    cargo update

    cargo install cargo-simd-detect --force
    cargo simd-detect
    
    export RUSTFLAGS="-C target-feature=+avx2"
    
    export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f -C linker=gcc"
    
注意: 使用 `nightly` 编译会报如下错误，需要设置参数 `export RUSTFLAGS="-C linker=gcc"`

    = note: rust-lld: error: sniper/target/debug/build/tensorflow-sys-a6e7af09a3d8a4cc/out/libtensorflow.so: invalid local symbol '_ZN9grpc_core7ExecCtx9exec_ctx_E' in global part of symbol table
              collect2: error: ld returned 1 exit status
              
或者在 .cargo/config.toml 中设置

    [target.x86_64-unknown-linux-gnu]
    rustflags = ["-C", "target-cpu=native", "-C", "target-feature=+avx512f", "-C", "linker=gcc"]

#### 测试 `simd` 性能

对比简单的 `32` 个 `f32` `sum` 性能, `Simd<f32, 32>` 表示 `32` 个 `float`, 再对比 `f32x4`, `f32x8`, `f32x16` 的结果。

具体实现见: `util/tests/tests.rs` 中 `test_simd_sum`。

结果如下，可以看出, 几种不同 `simd` 结构的速度差不多，相比普通的 `sum` 都有 `6` 倍多的加速。

    2024-09-25T21:05:18 [INFO] util/tests/tests.rs:115 - normal sum time spend: 4007 milliseconds, count: 10000000
    2024-09-25T21:05:31 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x4 sum time spend: 12415 milliseconds, count: 10000000
    2024-09-25T21:05:37 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x8 sum time spend: 6764 milliseconds, count: 10000000
    2024-09-25T21:05:41 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x16 sum time spend: 4065 milliseconds, count: 10000000
    2024-09-25T21:05:44 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x32 sum time spend: 2399 milliseconds, count: 10000000
    2024-09-25T21:05:45 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x16 sum time spend: 691 milliseconds, count: 10000000
    2024-09-25T21:05:45 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x32 sum time spend: 705 milliseconds, count: 10000000
    2024-09-25T21:05:50 [INFO] util/tests/tests.rs:174 - run_simd_sum_f32_mm256, use mm256 directly, sum time spend: 4533 milliseconds, count: 10000000
    2024-09-25T21:05:55 [INFO] util/tests/tests.rs:190 - run_simd_sum_f32_avx512, user avx512, sum time spend: 4924 milliseconds, count: 10000000
    
不同计算逻辑加速比如下

| Method                  | Time (ms) | Speedup |
|-------------------------|-----------|---------|
| normal sum              |      4007 |   1.00x |
| simd f32x4 sum          |     12415 |   0.32x |
| simd f32x8 sum          |      6764 |   0.59x |
| simd f32x16 sum         |      4065 |   0.99x |
| simd f32x32 sum         |      2399 |   1.67x |
| simd no copy f32x16 sum |       691 |   5.80x |
| simd no copy f32x32 sum |       705 |   5.68x |
| simd mm256              |      4533 |   0.88x |
| simd avx512             |      4924 |   0.81x |


备注:
- normal sum: 普通 sum。
- simd f32x4 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd f32x8 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd f32x16 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd f32x32 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd no copy f32x16 sum: 直接将结果 `sum` 到第一个参数。
- simd no copy f32x32 sum: 直接将结果 `sum` 到第一个参数。
- simd mm256: 直接使用 `__mm256` 指令计算，不使用 `Simd`, 但是会将结果 `load` 到原数组。
- simd avx512: 直接使用 `__avx512` 指令计算，不使用 `Simd`, 但是会将结果 `load` 到原数组。


可以看出，没有复制的计算逻辑是最快的，比普通 `sum` 快了 `6` 倍。其他的 `simd` 中 `N` 越大，则速度越快。而直接使用
指令计算的结果和普通 `sum` 差不多。

`f32x4` 甚至慢了 `3` 倍。

速度只涨到了 `61462 examples/sec`, 有点奇怪，`EmbeddingLookup` 耗时统计和之前比稍微快一点，`p95` 是 `164698`。但是和预期差很多。

`ps` 上耗时监控如下

    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 20000, p50: 115117, p95: 135947, p99: 137798, max: 135000
    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 510000, p50: 13949, p95: 105329, p99: 131676, max: 140000
    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 20000, p50: 0, p95: 940, p99: 1042, max: 6000
    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 20000, p50: 115111, p95: 135946, p99: 137798, max: 135000


##### 不使用 `simd` 的耗时监控如下

`trainer` 耗时监控, 速度只有 `31856 examples/sec`, 有点奇怪，之前能到 `40000 examples/sec`。

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 945  P50: 311674.082314  P95: 373167.408231  P99: 375977.000000  Max: 375977.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 946  P50: 42036.000000  P95: 51813.725490  P99: 64271.000000  Max: 64271.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 947  P50: 10260.135135  P95: 19540.625000  P99: 21908.125000  Max: 27457.000000
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:264 - after_run] - 2024-09-24 20:25:44.570218: step 1300, auc = 0.6708 (31.1 it/sec; 31856.0 examples/sec)
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:232 - after_run] - 2024-09-24 20:25:44.570601: step 1300, xentropy_mean:0 = 0.27869231 (31.1 it/sec; 31853.9 examples/sec)
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:232 - after_run] - 2024-09-24 20:25:44.570662: step 1300, prob_mean:0 = 0.08517376 (31.1 it/sec; 31853.8 examples/sec)
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:232 - after_run] - 2024-09-24 20:25:44.570710: step 1300, real_mean:0 = 0.08789062 (31.1 it/sec; 31853.7 examples/sec)
    
`ps` 耗时监控, 看起来和 `simd` 版本差不多，感觉是 `simd` 没生效。

    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 10000, p50: 115100, p95: 135947, p99: 137800, max: 155999
    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 220000, p50: 12917, p95: 98373, p99: 130281, max: 155999
    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 10000, p50: 0, p95: 878, p99: 1028, max: 999
    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 10000, p50: 115098, p95: 135947, p99: 137800, max: 155999

`hub` 耗时监控如下

    2024-09-24T20:38:49 [INFO] sniper/util/src/histogram.rs:659 - HubParseProto statistics in microseconds, total: 4900000, p50: 0, p95: 933, p99: 1039, max: 8000
    2024-09-24T20:39:19 [INFO] sniper/util/src/histogram.rs:659 - HubReadMessage statistics in microseconds, total: 4880000, p50: 891, p95: 2978, p99: 6894, max: 1622015


对比 `c++` 版本的 `EmbeddingLookup` `p95` 耗时，还不到 `10000`, 相差了 `10` 倍。


##### 不同 `simd` 实现的结果

###### `sum_f32_vectors_simd_flex::<16>`

`sum` 实现如下

    pub fn sum_f32_vectors_simd_flex<const N: usize>(a: &mut Vec<f32>, b: &Vec<f32>)
    where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
        std::simd::Simd<f32, N>: SimdFloat,
    {
        let len = a.len();
        let simd_len = len - (len % N);

        for i in (0..simd_len).step_by(N) {
            let a_chunk = Simd::<f32, N>::from_slice(&a[i..i+N]);
            let b_chunk = Simd::<f32, N>::from_slice(&b[i..i+N]);
            let sum = a_chunk + b_chunk;
            sum.copy_to_slice(&mut a[i..i+N]);
        }

        // Handle remaining elements
        for i in simd_len..len {
            a[i] += b[i];
        }
    }

经检查是参数给错了，`simd` 参数给成了 `4`, 应该给 `16`，和 `embedding_size` 一样，修改后速度能到 `66000 exampls/sec`,
与预期还是有差距。

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1916  P50: 139392.971246  P95: 164148.000000  P99: 164148.000000  Max: 164148.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1920  P50: 42103.000000  P95: 49243.059193  P99: 49926.977475  Max: 56925.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1917  P50: 15857.534247  P95: 34858.333333  P99: 47638.333333  Max: 95223.000000
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:264 - after_run] - 2024-09-25 00:55:27.114341: step 4600, auc = 0.6717 (64.2 it/sec; 65703.3 examples/sec)
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:232 - after_run] - 2024-09-25 00:55:27.114695: step 4600, xentropy_mean:0 = 0.27121678 (64.2 it/sec; 65696.7 examples/sec)
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:232 - after_run] - 2024-09-25 00:55:27.114824: step 4600, prob_mean:0 = 0.08899237 (64.2 it/sec; 65694.2 examples/sec)
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:232 - after_run] - 2024-09-25 00:55:27.114867: step 4600, real_mean:0 = 0.08007812 (64.2 it/sec; 65694.1 examples/sec)


`ps` 耗时监控如下

    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 10000, p50: 54313, p95: 128356, p99: 136274, max: 104000
    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 230000, p50: 6843, p95: 51800, p99: 94071, max: 104000
    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 10000, p50: 0, p95: 957, p99: 1044, max: 2000
    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 10000, p50: 54182, p95: 128022, p99: 136208, max: 104000


######  `sum_f32_vectors_simd_no_copy::<16>`

`sum` 实现如下

    pub fn sum_f32_vectors_simd_no_copy<const N: usize>(a: &mut Simd<f32, N>, b: &Vec<f32>)
    where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
        std::simd::Simd<f32, N>: SimdFloat,
    {
        let b_chunk = Simd::<f32, N>::from_slice(b.as_slice());
        *a += b_chunk;
    }


`trainer` 监控耗时如下

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1829  P50: 135684.803002  P95: 159176.000000  P99: 159176.000000  Max: 159176.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1828  P50: 42006.000000  P95: 51919.191919  P99: 61435.000000  Max: 61435.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1829  P50: 13885.297619  P95: 21867.650677  P99: 31383.000000  Max: 115997.000000
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:264 - after_run] - 2024-09-25 21:01:31.881627: step 9900, auc = 0.6665 (61.1 it/sec; 62592.1 examples/sec)
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:232 - after_run] - 2024-09-25 21:01:31.881838: step 9900, xentropy_mean:0 = 0.29910940 (61.1 it/sec; 62591.8 examples/sec)
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:232 - after_run] - 2024-09-25 21:01:31.881893: step 9900, prob_mean:0 = 0.10830545 (61.1 it/sec; 62591.8 examples/sec)
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:232 - after_run] - 2024-09-25 21:01:31.881932: step 9900, real_mean:0 = 0.09472656 (61.1 it/sec; 62591.8 examples/sec)


`ps` 耗时监控如下

    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 10000, p50: 51209, p95: 60430, p99: 61249, max: 78000
    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 220000, p50: 3384, p95: 43359, p99: 57832, max: 78000
    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 10000, p50: 0, p95: 914, p99: 1035, max: 1000
    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 10000, p50: 51208, p95: 60428, p99: 61247, max: 78000

`hub` 耗时监控如下

    2024-09-25T21:02:36 [INFO] sniper/util/src/histogram.rs:659 - HubReadMessage statistics in microseconds, total: 4040000, p50: 714, p95: 1064, p99: 1532, max: 3666077
    2024-09-25T21:02:36 [INFO] sniper/util/src/histogram.rs:659 - HubParseProto statistics in microseconds, total: 4050000, p50: 0, p95: 971, p99: 1046, max: 7000

从 `ps` 耗时监控来看，比之前快了一倍，训练速度能到 `6万`。

还是有点奇怪，速度差了很多。待继续排查。

还有个怀疑点: `c++` 中 `EmbeddingLookup` 结果是申请了一个 `var_cont * batch_size * embedding_size` 大小的数组保存结果，
所有变量只申请了一次内存，而 `rust` 中每个变量用了 `Vec<Vec<f32>>` 两层 `Vec` 来保存，因此有 `batch_size + 1` 次申请，
最终结果又需要复制到 `TensorMessage` 中，可能也有影响。

##### `EmbeddingLookup` 结果申请一次内存

### 经过第三轮排查与优化后与 `c++` 版本对比

添加了 `simd` 逻辑后速度并没有提高太多, 有点奇怪。并且 `simd` 复制与不复制的逻辑都试过。
`c++` 中也有复制逻辑，即 `sum` 后把 `simd` 的结果 `load` 到内存中，也有点奇怪。性能不应该差这么多。

可能要对比下 `c++` 中的逻辑。

### 第四轮排查

#### 测试 `c++` 中 `simd` 性能

`c++` 中 `simd` 加速 `sum` 实现如下

    #include <immintrin.h>

    void sum_8_f32(float* dst, const float* src) {
      __m256 m_dst = _mm256_loadu_ps(dst);
      __m256 m_src = _mm256_loadu_ps(src);
      __m256 m_val = _mm256_add_ps(m_dst, m_src);
      _mm256_storeu_ps(dst, m_val);
    }

    void sum_f32s(const float* src, float* dst, int32_t n) {
      int32_t c = n / 8;
      int32_t offset = 0;

      for (int32_t i = 0; i < c; ++i) {
        sum_8_f32(dst + offset, src + offset);
        offset += 8;
      }

      while (offset < n) {
        dst[offset] += src[offset];
        ++offset;
      }
    }
    
    
测试代码见: `sniper/trainer/trainer/core/test/test.cc`

结果如下, 普通逻辑就比 `rust` 快了 `4` 倍。

`mm256` 比普通 `sum` 快了约 `1` 倍，`mm512` 比普通 `sum` 快了约 `3` 倍。

`simd` 版本相比 `rust` 版本快了 `10` 倍。
    
    I0925 23:55:40.397699 49243 test.cc:32] run_sum_normal, time spend: 1399 milliseconds, count: 10000000
    I0925 23:55:40.939545 49243 test.cc:43] run_sum_simd_mm256, time spend: 541 milliseconds, count: 10000000
    I0925 23:55:41.313560 49243 test.cc:55] run_sum_simd_mm512, time spend: 374 milliseconds, count: 10000000


#### `rust` 编译 `release`

普通 `sum` 版本 `rust` 就比 `c++` 慢了 `4` 倍，有点奇怪，都是最简答的 `inline` 函数。
突然想到 `cargo build` 默认编的是 `debug` 版本，编 `release` 试下。

    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:115 - normal sum time spend: 300 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x4 sum time spend: 82 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x8 sum time spend: 44 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x16 sum time spend: 47 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x32 sum time spend: 45 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x16 sum time spend: 0 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x32 sum time spend: 0 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:174 - run_simd_sum_f32_mm256, use mm256 directly, sum time spend: 100 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:190 - run_simd_sum_f32_avx512, user avx512, sum time spend: 95 milliseconds, count: 10000000
    
不同计算逻辑加速比如下

| Method                      | Time (ms) | Speedup |
|-----------------------------|-----------|---------|
| Normal sum                  | 300       | 1.00x   |
| SIMD f32x4 (copy to slice)  | 82        | 3.66x   |
| SIMD f32x8 (copy to slice)  | 44        | 6.82x   |
| SIMD f32x16 (copy to slice) | 47        | 6.38x   |
| SIMD f32x32 (copy to slice) | 45        | 6.67x   |
| SIMD f32x16 (no copy)       | 0         | N/A     |
| SIMD f32x32 (no copy)       | 0         | N/A     |
| SIMD mm256                  | 100       | 3.00x   |
| SIMD AVX512                 | 95        | 3.16x   |

从结果看提高很大，甚至普通版本也比 `c++` 快了 `4` 倍。

并且直接使用 `mm256` 和 `avx512` 比 `Simd` 还要慢一些。避免复制的逻辑依然是最快的。


#### `simd` 加速 `Embedding` 效果

