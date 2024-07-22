# ps 和 hub

对于 embedding 参数, hub 将组织成 batch 的 sparse 特征发送给 ps, 每个 batch 被分配一个唯一的 batch_id。ps 
接收到 sparse 特征后， 根据 sparse 特征对应的 field 查找每个 sign 对应的 embedding 参数，并根据 batch_id 进行
sum。结果被保存到以 batch_id 为 key 的 map 中。当 trainer 根据 batch_id 和 field 进行 embedding 查找时，
即可直接从预先计算的结果中获取。
