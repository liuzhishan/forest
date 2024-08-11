# HashMap

`ps` 需要用 `HashMap` 来保存所有的参数，并且由于读写请求来自不同的线程，需要支持并发访问。

并发 `hashmap` 有如下一些候选:

- `dashmap`
- `Vec`: 对于 `hash` 模型，其 `sign` 的范围固定，可以直接用数组保存参数。
