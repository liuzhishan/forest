# tonic

## server 如何实现 mutable state ?

`tonic` 中 `prost` 生成的 `server trait` 都是 `&self` 为参数的接口，但是 `server` 中有些数据需要管理，必须 `ps` 中的参数, 如果作为 `server` 的成员变量，则相应的接口必须以 `&mut self` 为参数来实现对这些参数的改动。如何实现 ?

### 单例

使用 `OnceLock` 可以实现单例。


#### 如何返回单例中的引用 ?

单例可以通过以下方式获取

```
    let env = Env::instance().lock().unwrap();
```

但问题是 `env` 必须在当前函数中存在，因此会导致当前函数逻辑处理完之前其他线程都不能访问 `Env`。
需要一个办法直接获取到 `Env` 中的引用或者可变引用。

参考 `dashmap` 中的 `Ref` 和 `RefMut`, 可以自己实现对单例 `Env` 成员类似的引用。

但是发现也有问题。不能在 `tokio::spawn` 中使用。


### Arc<RefCell<T>>

### Arc<RwLock<T>>

最终采用 `Arc<RwLock<T>>` 实现。将需要管理的状态都作为 `Ps` 的成员变量，`Arc<RwLock<T>>` 都可以通过 `&self` 来访问，需要修改内部变量时用 `write()` 即可。
