# hub

`hub` is the `IO` module of sniper. Its main functionality is to read data, and communicating
with `ps` and `trainer`.

If the input data is assemblied by batch, then `hub` will generate a batch_id to repesent the
batch data. If not, `hub` will assembly the data into batch first, and then generate a batch_id.
Then it will send the data to `ps` for `embedding` operations, and send batch_id to `trainer` at
the same time.

The components are as follows:
- `hub_server`: main logic, start all worker threads.
- `request_handler`: handle `rpc` request, process, and return result.
- `task`: task node, handle of data processing logic.
- `pipeline`: data processing pipeline, every data format should have a unique pipeline.
