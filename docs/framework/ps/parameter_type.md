# 参数类型

从魔性训练的过程中我们可以看到，参数类型主要分为 sparse 特征对应的 embedding 参数和网络层参数。

## embedding 参数

sparse 特征即原始数据经过处理的到的离散特征，通常用 int64 数组来表示。即每个 sparse 特征对应一个 int64 数组。
并且这些 int64 数组的长度也是不固定的。 在训练时我们需要保持网络结构形状的固定，那么这些 sparse 特征是如何进入
到神经网络中训练的呢? 我们采取的方式就是将每个 int64 映射到一个固定大小的 embedding 向量。然后对每个特征对应的
int64 数组的 embedding 向量进行 sum，或者其他处理，最终每个特征得到一个固定大小的向量，从而使得网络的输入层形状
保持固定。如下图所示(TODO: 示意图)

这一过程相当于 int64 数组中的每一个 sign 看做一个 one hot 的特征，每一个 sign 是一个无限大的字典的 key，embedding
参数则是每个 key 对应的值。embedding 参数查找的过程就是从这个无限大的字典中根据 sign 查找 embedding 的过程。

而对于 sign 的处理有以下几种不同的方案。

### 原始值

sign 的 值域是 int64，且我们也并不知道每个特征中具体的不重复的 sign 个数, 如果我们不对每个 sign 做其他处理，
直接根据 sign 的值来查找 embedding，则需要一个 hashmap 来保存参数。

这一方案有如下优缺点:

优点:
- 不需要对 sign 进行其他处理。

缺点:
- 根据 sign 查找 hashmap 需要时间, 而一个 batch 的训练中需要同时查询非常多的 sign 对应的参数，对性能提出了极大的挑战。
- 保存参数需要的内存和 cpu 是和 sign 的个数成正比的，在 sign 比较多的时候会消耗大量内存，并且参数的查找与更新
也需要大量的计算，ps 可能会成为训练的瓶颈。
- sign 分布的不均匀会造成热点 key 的出现，进一步加剧性能方面的问题。

### hash 处理

直接根据 sign 来保存参数比较消耗资源，是否有其他办法来解决查找 sign 的问题？ 

有一个折中的方案则是对 sign 进行 hash 处理后再进行查找。我们对每一个特征提前设置一个固定的 size，这个 size 应该尽量接近
特征中不重复的 sign 的个数，然后对于每一个 sign，我们都先对 sign 进行 hash，然后再对 size 进行取模。经过取模处理后
sign 的值域就是固定的 [0, size), 处理起来会更简单。相比无 hash 模型，有 hash 模型有如下优势:

- sign 值域的范围是固定的 [0, size), 因此可以用数组来保存参数，sign 取模后的值可以直接根据下标进行访问来获取
embedding 参数，性能比 hashmap 要好很多。
- 取模后的 sign 个数也减少了很多，极大的减少了保存参数需要的内存消耗, 更节省资源。

但是，这一方案也有如下缺点:
- hash 可能会冲突。即两个不同的 sign 经过 hash 后得到同一个值，会导致查找参数时本来应该不同的参数却得到了相同的参数，
一定程度上会影响效果。不同 hash 函数的冲突率不一样，可能还需要对比不同的 hash 函数才能确定比较好的 hash 函数。

### double hash

为了避免 hash 冲突，我们可以考虑将一个 int64 拆成高低两个 int32, 对这两个 int32 分别进行 hash，然后将结果 sum 或者
concat 起来。其冲突率可以按如下公式计算。

假设按一个 hash 函数处理的冲突率是 f1, 两个 hash 函数的冲突率分别是 f1, f2。则拆分后的两个 int32 同时冲突的概率则是
f1 * f2, 远远小于原来的冲突率 f1。如下所示(TODO: 示意图)

### 不 hash 直接取模

对 sign 进行 hash 理论上可以让 sign 分布的更分散一些。但是取模后的结果查找 embedding 最重要的因素还是是否会冲突，
是否 hash 影响并不是很大, 因此直接对 size 进行取模也是一种处理方案。

### 总结

以上四种处理方案，各有其优缺点。我们第一版采取方案二(hash 处理) 的方式来处理，其他方案待之后再进行对比。

### rust 实现

对于不同的处理方式，我们定义一个 trait 来表示 sign 相关的处理。

    /// Different process method of sign
    pub trait SignConverter {
        fn convert(self, sign: u64) -> u64;
    }

## 网络层参数

对于网络层参数，通常是一个节点名称对应一个矩阵。网络层的参数计算与更新主要发生在 GPU 上，即 trainer 上，ps 上只需要
完成保存和查找功能即可满足训练的需求。因此直接根据节点名称来保存参数即可。我们采取 hashmap 来保存。

### rust 实现

## 并行处理

因为计算量非常大，我们必须要并行处理参数的更新与查找。

### rust 实现