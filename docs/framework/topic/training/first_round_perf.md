# 第一轮排查：定位瓶颈

## 训练速度慢

### 现在是读取单条样本，`hub` 里拼 `batch`

之前是离线拼好的 `batch`，会导致一些速度差异，`batch` 数据很快。

### 是否和 `lru` 有关

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

### `EmbeddingLookup` 时对整个 `EmbeddingManager` 加锁，锁的粒度太粗。

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


