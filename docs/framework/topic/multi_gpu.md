# 多卡训练

基于 `horovod` 实现多卡训练。

## 训练日志

`2` 卡的训练每张卡在 `6万` 左右，加起来和单卡差不多。默认经过 `30` step 各 `GPU` 同步一次参数。两张卡加起来速度
和单卡差不多，说明瓶颈还是在 `hub` 端。需要再调调参数或者再优化下。

`trainer` 日志如下

    [1,1]<stderr>:I1002 00:21:28.179744  1326 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1553  P50: 41644.387001  P95: 57511.261261  P99: 68023.000000  Max: 68023.000000
    [1,1]<stderr>:OpsPushGrad statistics => count: 1549  P50: 44087.000000  P95: 69190.729483  P99: 73898.936170  Max: 75266.000000
    [1,1]<stderr>:OpsReadSample statistics => count: 1553  P50: 13049.498747  P95: 22693.478261  P99: 32596.666667  Max: 91941.000000
    [1,0]<stderr>:2024-10-02 00:21:29,533 - INFO [hooks.py:264 - after_run] - 2024-10-02 00:21:29.533615: step 2800, auc = 0.6706 (63.9 it/sec; 65430.9 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:29,533 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:29.533833: step 2800, xentropy_mean:0 = 0.28447193 (63.9 it/sec; 65441.6 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:29,533 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:29.533905: step 2800, prob_mean:0 = 0.09642899 (63.9 it/sec; 65442.1 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:29,533 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:29.533949: step 2800, real_mean:0 = 0.08886719 (63.9 it/sec; 65442.4 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:29,534 - INFO [trainer.py:300 - train] - current lr: 0.050000
    [1,0]<stderr>:INFO:tensorflow:global_step/sec: 63.9066
    [1,0]<stderr>:2024-10-02 00:21:29,538 - INFO [basic_session_run_hooks.py:692 - _log_and_record] - global_step/sec: 63.9066
    [1,1]<stderr>:2024-10-02 00:21:29,691 - INFO [hooks.py:264 - after_run] - 2024-10-02 00:21:29.691529: step 2800, auc = 0.6708 (58.1 it/sec; 59447.6 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:29,691 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:29.691831: step 2800, xentropy_mean:0 = 0.28501779 (58.1 it/sec; 59450.7 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:29,691 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:29.691902: step 2800, prob_mean:0 = 0.08909728 (58.1 it/sec; 59450.9 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:29,691 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:29.691952: step 2800, real_mean:0 = 0.09082031 (58.1 it/sec; 59450.9 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:29,692 - INFO [trainer.py:300 - train] - current lr: 0.050000
    [1,0]<stderr>:2024-10-02 00:21:31,320 - INFO [hooks.py:264 - after_run] - 2024-10-02 00:21:31.320221: step 2900, auc = 0.6704 (56.0 it/sec; 57315.3 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:31,320 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:31.320481: step 2900, xentropy_mean:0 = 0.33368897 (56.0 it/sec; 57314.0 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:31,320 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:31.320562: step 2900, prob_mean:0 = 0.09811372 (56.0 it/sec; 57313.8 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:31,320 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:31.320624: step 2900, real_mean:0 = 0.11035156 (56.0 it/sec; 57313.2 examples/sec)
    [1,0]<stderr>:2024-10-02 00:21:31,320 - INFO [trainer.py:300 - train] - current lr: 0.050000
    [1,1]<stderr>:2024-10-02 00:21:31,323 - INFO [hooks.py:264 - after_run] - 2024-10-02 00:21:31.323569: step 2900, auc = 0.6705 (61.3 it/sec; 62743.2 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:31,323 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:31.323829: step 2900, xentropy_mean:0 = 0.35707310 (61.3 it/sec; 62745.2 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:31,323 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:31.323890: step 2900, prob_mean:0 = 0.09155443 (61.3 it/sec; 62745.6 examples/sec)
    [1,1]<stderr>:2024-10-02 00:21:31,323 - INFO [hooks.py:232 - after_run] - 2024-10-02 00:21:31.323934: step 2900, real_mean:0 = 0.11523438 (61.3 it/sec; 62745.8 examples/sec)