# 训练速度优化总结

本文档总结了对训练速度进行的多轮排查和优化过程。

## 第一轮排查：定位瓶颈

详细内容请参考：[第一轮排查详情: 定位瓶颈](framework/topic/training/first_round_perf.md)

## 第二轮排查：EmbeddingLookup 慢

主要发现：
- EmbeddingLookup 操作是主要瓶颈
- 添加 LRU 缓存后速度变慢
- 读取单条样本并在 hub 中拼接 batch 的方式影响了速度

`lock free`: 速度从 `2千 examples/sec` 增加到 `4万 examples/sec`。

详细内容请参考：[第二轮排查详情: EmbeddingLookup 慢](framework/topic/training/second_round_embedding_lookup_slow.md)

## 第三轮排查：feed_queue 耗时

主要发现：
- feed_queue 中各步骤的耗时分析
- 与 C++ 版本的性能对比

详细内容请参考：[第三轮排查详情: feed_queue 耗时](framework/topic/training/third_round_feed_queue_time.md)

## 第四轮排查：SIMD 加速

主要发现：
- 使用 SIMD 指令进行加速的效果
- 不同 SIMD 实现的性能比较

详细内容请参考：[第四轮排查详情: SIMD 加速](framework/topic/training/fourth_round_simd_acceleration.md)

## 第五轮排查：C++ SIMD 性能

主要发现：
- C++ 中 SIMD 的性能测试结果
- SIMD 加速 Embedding 的效果
- `cargo build --release`

速度从 `6万 examples/sec` 增加到 `14 万 examples/sec`。

详细内容请参考：[第五轮排查详情: C++ SIMD 性能](framework/topic/training/fifth_round_cpp_simd_performance.md)

## 第六轮排查：`ReadSample` 耗时

主要发现：
- `ReadSample` 耗时

详细内容请参考：[第六轮排查详情: `ReadSample` 耗时](framework/topic/training/seventh_round_read_sample_slow.md)

## 第七轮排查：`PushGrad` 耗时

主要发现：
- `PushGrad` 耗时

详细内容请参考：[第七轮排查详情: `PushGrad` 耗时](framework/topic/training/seventh_round_push_grad_slow.md)

## 总体结论

通过多轮排查和优化，我们发现了影响训练速度的几个关键因素，并尝试了多种优化方法。主要的优化点包括 EmbeddingLookup 操作的改进、feed_queue 的优化、以及使用 SIMD 指令进行加速。

尽管取得了一定的进展，但与 C++ 版本相比，仍有性能差距。未来可能需要进一步优化底层实现，包括内存管理、并发控制等方面，以缩小与 C++ 版本的性能差距。
