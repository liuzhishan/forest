# ps (Pamameter Server)

在广告模型中，sparse 特征的数量非常多，通常可以达到几十亿甚至上百亿的规模，这些 sparse 特征对应的 embedding
参数量就非常大，难以保存到单张 GPU 卡甚至单台机器中。我们用 sign (格式为 int64_t) 表示每个具体的 sparse 特征。
在模型训练时，每一个 batch 都需要根据 sparse 特征的 sign, 从几十亿参数中找到其对应的参数, 进行前向计算与梯度
计算，并更新对应的参数。因此这些计算量也非常大，通常的做法是采用独立的 parameter server 模块来负责参数相关的计算,
包括查找、更新、保存等。这些计算主要是采用 cpu 进行，可以采用多台 cpu 机器来进行相关的计算。

ps 在训练框架中的作用如下所示(TODO: 示意图)。

如上图所示，我们需要思考如何解决以下问题:

- sparse 特征对应的参数与网络层参数的含义与更新方式都不同，如何区分两种参数？ 如何更新两种不同的参数？
- 采取什么格式来保存 sparse 特征对应的参数？ 如何设计数据结构能够达到高性能的查找与更新？
- 参数如何创建与保存？保存采取什么格式？ 参数保存需要考虑什么问题？
- 模型如果需要从之前的 checkpoint 继续训练，如何加载之前保存的参数？
- 不同参数如何在多台 ps 机器中进行分配?
- 不同的优化器计算逻辑不同，如何方便的支持不同的优化器?
- ps 保存的参数如何与 hub 以及 trainer 中的 batch 进行关联？
- ps 与 trainer、hub 如何进行高性能的通信？

接下来本章节将主要从以下几方面详细介绍 ps 的各个组成部分:

- 参数类型 
- 参数更新
- 参数保存
- 参数加载
- 负载均衡
- 优化器
- ps 与 trainer
- ps 与 hub