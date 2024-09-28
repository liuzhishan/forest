# training

发现训练速度很慢，每秒只有 `2000` 多条样本，而 `c++` 版本能达到 `24 万` 左右。差异比较大。

`c++` 版本速度可见文档:  [`c++`版本速度](framework/topic/training/cpp_version_speed.md)。

具体排查步骤如下:
- [第一轮排查: 定位瓶颈](framework/topic/training/first_round_perf.md)
- [第二轮排查: `EmbeddingLookup` 性能](framework/topic/training/second_round_embedding_lookup_slow.md)
- [第三轮排查: `feed_queue` 耗时](framework/topic/training/third_round_feed_queue_time.md)
- [第四轮排查: `SIMD` 加速](framework/topic/training/fourth_round_simd_acceleration.md)
- [第五轮排查: `C++` SIMD 性能](framework/topic/training/fifth_round_cpp_simd_performance.md)
- [第六轮排查: `ReadSample` 性能](framework/topic/training/sixth_round_read_sample_slow.md)
- [第七轮排查: `PushGrad` 性能](framework/topic/training/seventh_round_push_grad_slow.md)
- [总结](framework/topic/training/summary.md)