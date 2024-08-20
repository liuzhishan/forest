# Concurrent dispatch

`ps` 根据请求的不同需要执行许多不同的任务，如保存不同的 `varname` 对应到参数，查询不同 `varname`
对应的 `embedding`, 这些任务执行的时间不相同，比如 `embedding` 查询需要毫秒的时间，但是保存模型
可能需要分钟级的时间。并且这些任务也需要进行并行的调度。 因此需要一个统一的 `dispatcher` 来调度
这些任务。

如何实现一个并行的调度器 ?

## c++ concurrent dispatch

最朴素的想法是每个任务都新建一个 `std::thread` 来运行。但是 `std::thread` 的创建、消耗、切换都有
不小的开销，因此不是一个很好的方案。

如何避免这些开销? 一个很自然的想法就是采用线程池, 复用已有的线程。每个需要执行的任务用 `Task` 表示，
对于每种不同类型的任务，都提前创建好线程池，线程池中维护一个 `Worker` 队列，需要执行的 `Task` 则被
分配给队列中空闲的 `Worker`，并且通过 `conditional_variable` 通知 `Worker`, `Worker` 负责执行具体
的 `Task`。

每个任务结束后可能还有一些后处理的工作，比如加载模型参数需要等所有子任务都成功后才可以判断为
成功。可以通过回调函数来实现，将其设置为 `Task` 的成员变量。

主要涉及到以下三个模块:
- Worker
- WorkerPool
- Task

具体实现如下。

TODO: 补充代码示例。

## rust concurrent dispatch

`rust` 中可以直接使用 `tokio::spawn` 执行一个并发任务，如果需要返回结果则使用 `await` 等待即可。
回调逻辑可以直接写在 `spawn` 任务中。 比 `c++` 中要简单很多。
