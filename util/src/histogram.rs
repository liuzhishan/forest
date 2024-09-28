//! Histogram is used to calcuate the statistics of time spend, such as median, min, max, average.
//!
//! The histogram data is computed every time we want to record a time spend value. It will be
//! computed very frequently, and concurrently. So we need a high performance solution to compute
//! the histogram statistics.
//!
//! Min, average of histogram is easy to compute, and the main chanllenge of the indicators is
//! percentile. We use an extremely fast approximation method to solve the problem.
//!
//! Normally to compute percentile we need to sort all the data, and find the percentile by index.
//! But sorint every time a value is pushed to the vector is expensive. The value is typically time
//! spend in milliseconds, representing as an `u64`, although it would be smaller than max of `u32`.
//! To avoid sorting, we can devide the value range into roughly 100 buckets. Then we can decide which
//! bucket the value should be put into quickly, and increment the bucket value to be count of numbers
//! in the bucket. We can use the bucket values to linearly interpolation to compute the approximate
//! percentile value.
//!
//! More detail will be explained below.
//!
//! As for the concurrent problem, to aoiding too much lock when count the numbers, we aggregate the
//! statistics data in each thread (more precisely, the spawned task), and then send to one merge
//! channel periodically.

use std::{
    default,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use log::{error, info};

use strum::{EnumCount, EnumDiscriminants, EnumString, FromRepr, ToString};
use tokio::time::sleep;
use tokio::{select, sync::mpsc};

use coarsetime::{Duration, Instant, Updater};

use likely_stable::{likely, unlikely};

use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};

use crate::error_bail;

/// All histogram type to be recorded.
#[derive(
    Default, Clone, FromRepr, Debug, PartialEq, EnumCount, EnumDiscriminants, EnumString, ToString,
)]
#[repr(usize)]
pub enum HistogramType {
    #[default]
    None,

    /// ps
    PsCreate,
    PsFeedSample,
    PsPull,
    PsPush,
    PsEmbeddingLookup,
    PsEmbeddingLookupNewVec,
    PsEmbeddingLookupSum,
    PsEmbeddingLookupOneVariable,
    PsEmbeddingLookupDispatch,
    PsEmbeddingLookupWaiting,
    PsPushGrad,
    PsSave,
    PsRestore,
    PsComplete,
    PsFeedCached,
    PsLookupCached,

    /// hub
    HubStartSample,
    HubReadSample,
    HubNext,
    HubProcess,
    HubFeedSample,
    HubReadMessage,
    HubCountMessage,
    HubBatchAssembler,
    HubCountAfterItemFilter,
    HubCountAfterLabelExtractor,
    HubCountSamplePos,
    HubCountSampleNeg,
    HubDecompress,
    HubParseProto,

    /// trainer op
    OpsCreate,
    OpsFeedSample,
    OpsPull,
    OpsPush,
    OpsEmbeddingLookup,
    OpsPushGrad,
    OpsSave,
    OpsRestore,
    OpsComplete,
    OpsStartSample,
    OpsReadSample,
}

impl From<usize> for HistogramType {
    fn from(value: usize) -> Self {
        match HistogramType::from_repr(value) {
            Some(x) => x,
            None => HistogramType::None,
        }
    }
}

impl From<HistogramType> for usize {
    fn from(x: HistogramType) -> Self {
        x as usize
    }
}

/// Statistics values of histogram.
#[derive(Default)]
pub struct HistogramResult {
    /// Median of data.
    pub median: f64,

    /// Percentile 95 of data.
    pub p95: f64,

    /// Percentile 99 of data.
    pub p99: f64,

    /// Average of data.
    pub average: f64,

    /// Standard deviation.
    pub standard_deviation: f64,

    /// Max.
    pub max_value: f64,
}

/// Buckets for computing percentile.
///
/// To speedup percentile computing, the main idea is to split each `u64` value into buckets, and record
/// the total number of values in each buckets. Then we can find which bucket the percentile fall into quickly,
/// in linear time of buckets count. And then use the boundary of the bucket to linearly approximate the percentile.
///
/// But how many buckets should we use? How many values should one bucket contains?
///
/// The total number of `u64` is `1 << 64`, and the total buckets of percentile is 100. Because the time
/// spend is an `u32` normally not too big, we can split more buckets if the value is small, and less buckets
/// when the value is big.
///
/// Let's do some math!
///
/// In [1]: import math
/// In [2]: a = math.pow(2, 64)
///
/// In [3]: a
/// Out[3]: 1.8446744073709552e+19
///
/// In [4]: math.log(a, 1.5)
/// Out[4]: 109.4087226464931
///
/// In [5]: math.log(a, 1.4)
/// Out[5]: 131.8427338947933
///
/// In [6]: math.log(a, 1.6)
/// Out[6]: 94.3852702308447
///
/// If we use geometric sequence to split the all `u64` values, and use `1.5` as the common ratio, all `u64`
/// values would fall into roughly 109 buckets. The boundary of each bucket is precomputed once. When we get
/// an `u64` value, we can quickly find which bucket it fall into, and increase the count of this bucket.
///
/// The buckets look like below:
///
/// | 1 | 2 | 3   |   5   |     8     | ..... |     math.pow(2, 63)     |
///
/// How can we find which bucket the value fall into?
///
/// Suppose we have an `u64` value `x`, which is `000011001` in binary format. We know the highest bit of `1` is
/// at position 4, starting at 0 from the lowest bit. It must meet the condition `(1 << 4) < x (1 << 5)`. Because
/// thare exactly `64` bits in an `u64`, right boudary of each bucket is at most `1.5x` of left boudary, so each
/// bit of an `u64` must fall into exactly one bucket. For the `64` bits of an `u64`, we can precompute each bit
/// position. As shown below With the buckets:
///
/// buckets:           | 1 | 2 | 3   |   5   |     8     | ..... |     math.pow(2, 63)     |
/// u64 bits position:   1   1         1           1       .....              1
///
/// When we get an `u64` value, we can get it's highest postion of `1` bit in just one bit operation, and then
/// get the bucket index from computed bucket index quickly. Then incrase the total number in the bucket.
#[derive(Clone)]
struct HistogramBucket {
    /// Buckets for all `u64` range.
    bucket_values: Vec<u64>,

    /// Bit position of all `1`s in an `u64`.
    bit_position: Vec<usize>,

    /// Number of integer that fall into the bucket.
    buckets: Vec<u64>,
}

impl HistogramBucket {
    pub fn new() -> Self {
        // Bigger than 109.
        let total_bucket = 110;

        let mut bucket_values: Vec<u64> = vec![0; total_bucket];

        // Initialize first two elements, the rest will be computed incremently.
        bucket_values[0] = 1;
        bucket_values[1] = 2;

        for i in 2..total_bucket {
            bucket_values[i] = (bucket_values[i - 1] as f64 * 1.5).ceil() as u64;
        }

        let mut bit_position: Vec<usize> = vec![0; 64];

        bit_position[0] = 0;

        // Find the right bit postition for all `1`s in `u64`.
        let mut pos: usize = 1;
        for i in 1..64 {
            let x = 1u64 << i;

            while pos < total_bucket {
                if bucket_values[pos - 1] <= x && x < bucket_values[pos] {
                    break;
                } else {
                    pos += 1;
                }
            }

            bit_position[i] = pos;
        }

        // Buckets for counting integer numbers.
        let buckets: Vec<u64> = vec![0; total_bucket];

        Self {
            bucket_values,
            buckets,
            bit_position,
        }
    }
    /// Get the bucket index by value.
    ///
    /// We use bit operation to decide the index.
    #[inline]
    pub fn get_index(&self, value: u64) -> usize {
        if value == 0 {
            0
        } else {
            self.bit_position[63 - value.leading_zeros() as usize]
        }
    }

    /// Get the left boundary of bucket.
    #[inline]
    fn get_left_boundary(&self, pos: usize) -> u64 {
        if pos <= 0 {
            0
        } else {
            self.bucket_values[pos]
        }
    }

    /// Get the right boundary of bucket.
    #[inline]
    fn get_right_boundary(&self, pos: usize) -> u64 {
        if pos == 0 {
            0
        } else if pos >= self.bucket_values.len() {
            // Unlikely to happen.
            if self.bucket_values.len() > 0 {
                self.bucket_values[self.bucket_values.len() - 1]
            } else {
                0
            }
        } else {
            self.bucket_values[pos + 1]
        }
    }

    /// Compute the percentile using buckets.
    ///
    /// The param `p` is the target percentile we want to find.
    ///
    /// First we get the right count by p and total. Then we find the right bucket by `acumulative_sum` which
    /// the threshold fall into. In the bucket, we know there are `self.buckets[i]` intergers, and we can compute
    /// the approximate position in the bucket by threshold. Then using the left and right boundary of the bucket,
    /// we can compute the approximate percentile using linear interpolation.
    pub fn get_percentile(&self, p: f64, total: u64) -> f64 {
        let threshold: f64 = total as f64 * p / 100.0;
        let mut cumulative_sum: f64 = 0.0;

        for (i, v) in self.buckets.iter().enumerate() {
            let number: f64 = *v as f64;

            cumulative_sum += number;

            if cumulative_sum >= threshold {
                let left_sum = cumulative_sum - number;

                // Get the approximate position in the bucket.
                let pos: f64 = if *v != 0 {
                    (threshold - left_sum) / number
                } else {
                    0.0
                };

                let left = self.get_left_boundary(i) as f64;
                let right = self.get_right_boundary(i) as f64;

                // Using linear interpolation to approximate the percentile.
                let r: f64 = left + (right - left) * pos;

                return r.round();
            }
        }

        0.0
    }

    pub fn clear(&mut self) {
        for i in 0..self.buckets.len() {
            self.buckets[i] = 0;
        }
    }

    /// Return the last value of `bucket_values`.
    pub fn last_value(&self) -> u64 {
        self.bucket_values[self.bucket_values.len() - 1]
    }

    /// Find the index of bucket by `v`, and increase the bucket value by 1.
    pub fn add(&mut self, v: u64) {
        let index = self.get_index(v);

        if index < self.buckets.len() {
            self.buckets[index] += 1;
        }
    }

    /// Merge with other `HistogramBucket`, sum the value in `self.buckets`.
    pub fn merge(&mut self, other: &HistogramBucket) {
        if self.buckets.len() == other.buckets.len() {
            for i in 0..self.buckets.len() {
                self.buckets[i] += other.buckets[i];
            }
        }
    }
}

/// Data structure for high performance histogram statistics.
#[derive(Clone)]
pub struct HistogramDetail {
    /// HistogramType.
    pub histogram_type: HistogramType,

    /// Frequency to send to merge channel.
    pub frequency: u64,

    /// Min value.
    pub min: u64,

    /// Max value.
    pub max: u64,

    /// Total count of values.
    pub num: u64,

    /// Sum of all values.
    pub sum: u64,

    /// Sum of squares.
    pub sum_squares: u64,

    /// Buckets for computing percentile.
    buckets: HistogramBucket,
}

impl HistogramDetail {
    pub fn new(histogram_type: HistogramType) -> Self {
        let frequency = Self::get_frequency(&histogram_type);
        let buckets = HistogramBucket::new();
        let min = buckets.last_value();

        Self {
            histogram_type: histogram_type.clone(),
            frequency,
            min,
            max: 0,
            num: 0,
            sum: 0,
            sum_squares: 0,
            buckets,
        }
    }

    /// Get different frequency by histogram_type.
    ///
    /// The default result is 10000.
    fn get_frequency(histogram_type: &HistogramType) -> u64 {
        match histogram_type {
            HistogramType::HubFeedSample => 10000,
            _ => 10000,
        }
    }

    /// Clear all variables.
    pub fn clear(&mut self) {
        self.min = self.buckets.last_value();
        self.max = 0;

        self.num = 0;

        self.sum = 0;
        self.sum_squares = 0;

        self.buckets.clear();
    }

    /// If the total number is 0.
    pub fn is_empty(&self) -> bool {
        self.num == 0
    }

    /// Add one value, update all statistics variables.
    pub fn add(&mut self, v: u64) {
        self.min = self.min.min(v);
        self.max = self.max.max(v);

        self.num += 1;

        // Maybe overflow, must use `wrapping_add`.
        self.sum = self.sum.wrapping_add(v);

        // Maybe overflow, must use `wrapping_add` and `wrapping_mul`.
        self.sum_squares = self.sum_squares.wrapping_add(v.wrapping_mul(v));

        self.buckets.add(v);
    }

    /// Merge other with self, update all variables of self.
    ///
    /// We merge each statistics variable seperately, according to their meaning.
    /// For example, `self.max.max(other.max)` for `self.max`.
    pub fn merge(&mut self, other: &HistogramDetail) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);

        // Maybe overflow, must use `wrapping_add`.
        self.num = self.num.wrapping_add(other.num);

        // Maybe overflow, must use `wrapping_add`.
        self.sum = self.sum.wrapping_add(other.sum);
        self.sum_squares = self.sum_squares.wrapping_add(other.sum_squares);

        self.buckets.merge(&other.buckets);
    }

    /// Get the median of all values.
    pub fn median(&self) -> f64 {
        self.buckets.get_percentile(50.0, self.num)
    }

    /// Get percentile 95 of all values.
    pub fn p95(&self) -> f64 {
        self.buckets.get_percentile(95.0, self.num)
    }

    /// Get percentile 99 of all values.
    pub fn p99(&self) -> f64 {
        self.buckets.get_percentile(99.0, self.num)
    }

    /// Get the average of all values.
    pub fn average(&self) -> f64 {
        if self.num > 0 {
            self.sum as f64 / self.num as f64
        } else {
            0.0
        }
    }

    /// Get the standard deviation of all values.
    pub fn standard_deviation(&self) -> f64 {
        if self.num == 0 {
            0.0
        } else {
            let x = self
                .sum_squares
                .wrapping_mul(self.num)
                .wrapping_sub(self.sum.wrapping_mul(self.sum));
            let variance = x as f64 / ((self.num.wrapping_mul(self.num)) as f64);

            variance.sqrt()
        }
    }

    /// Check whether `num` is multiple of `frequency`.
    pub fn is_enough_to_send(&self) -> bool {
        if self.frequency > 0 && self.num % self.frequency == 0 {
            true
        } else {
            false
        }
    }

    /// Get int of `histogram_type`.
    #[inline]
    pub fn get_histogram_type_int(&self) -> usize {
        self.histogram_type.clone().into()
    }

    /// Print the statistics.
    #[inline]
    pub fn to_string(&self) -> String {
        format!(
            "{} statistics in microseconds, total: {}, p50: {}, p95: {}, p99: {}, max: {}",
            self.histogram_type.to_string(),
            self.num,
            self.median(),
            self.p95(),
            self.p99(),
            self.max,
        )
    }
}

/// Histogram statistics.
///
/// The value is aggrated locally, and send to a merge channel periodicity.
#[derive(Clone)]
pub struct Histogram {
    /// Record of histogram data.
    details: Vec<HistogramDetail>,

    /// mpsc channel sender between threads.
    sender: mpsc::Sender<HistogramDetail>,
}

impl Histogram {
    pub fn new(sender: mpsc::Sender<HistogramDetail>) -> Self {
        let details = Self::get_all_histogram_details();

        Self { details, sender }
    }

    /// Get `HistogramDetail` for all `HistogramType`, stored in a `Vec`.
    pub fn get_all_histogram_details() -> Vec<HistogramDetail> {
        let size = HistogramType::COUNT;
        let mut details = Vec::with_capacity(size);

        for i in 0..size {
            details.push(HistogramDetail::new(HistogramType::from(i)));
        }

        details
    }

    #[inline]
    pub fn clear(&mut self) {
        for detail in self.details.iter_mut() {
            detail.clear();
        }
    }

    /// Send `HistogramDetail` to aggregate channel.
    fn send_detail(&mut self, index: usize) {
        if index >= self.details.len() {
            // Unlikely to happen.
            error!(
                "out of range, enum_int: {}, self.details.len(): {}",
                index,
                self.details.len()
            );
            return;
        }

        let sender = self.sender.clone();
        let detail = self.details[index].clone();

        // Must clear the detail.
        self.details[index].clear();

        tokio::spawn(async move {
            match sender.send(detail).await {
                Ok(_) => {}
                Err(err) => {
                    error!("send histogram detail failed! err: {}", err);
                }
            }
        });
    }

    pub fn add(&mut self, histogram_type: HistogramType, v: u64) {
        let enum_int: usize = histogram_type.into();

        if likely(enum_int < self.details.len()) {
            self.details[enum_int].add(v);

            let detail = &self.details[enum_int];

            if detail.is_enough_to_send() {
                self.send_detail(enum_int);
            }
        } else {
            // Unlikely to happen.
            error!(
                "out of range, enum_int: {}, self.details.len(): {}",
                enum_int,
                self.details.len()
            );
        }
    }
}

/// Print `Histogram` every `interval_seconds`.
#[derive(Clone)]
struct PrintHistogram {
    /// How many seconds to print statistics result.
    interval_seconds: u32,

    /// Aggregate result for different item.
    aggregator: Arc<Vec<Mutex<HistogramDetail>>>,

    /// Different role concern different `HistogramType`, only print the interested detail.
    followed_histogram_indexes: Vec<usize>,
}

impl PrintHistogram {
    pub fn new(
        interval_seconds: u32,
        aggregator: Arc<Vec<Mutex<HistogramDetail>>>,
        followed_histogram_indexes: Vec<usize>,
    ) -> Self {
        Self {
            interval_seconds,
            aggregator,
            followed_histogram_indexes,
        }
    }

    /// Print detail every `self.interval_seconds` seconds, using current time to determine whether
    /// it's tiem to print.
    pub async fn run(self, subsys: SubsystemHandle) -> Result<()> {
        let duration = std::time::Duration::from_secs(self.interval_seconds as u64);
        let mut interval = tokio::time::interval(duration);

        loop {
            select! {
                _ = interval.tick() => {
                    for index in self.followed_histogram_indexes.iter() {
                        if unlikely(*index >= self.aggregator.len()) {
                            continue;
                        }

                        let x = self.aggregator[*index].lock().unwrap();

                        if x.num > 0 {
                            info!("{}", x.to_string());
                        }
                    }
                },
                _ = subsys.on_shutdown_requested() => {
                    info!("PrintHistogram shutdown!");
                    return Ok(());
                }
            }
        }
    }
}

/// Aggregate multiple histogram statistics from different threads.
pub struct HistogramAggregator {
    /// mpsc channel receiver for `HistogramDetail`.
    receiver: mpsc::Receiver<HistogramDetail>,

    /// Aggregate result for different item.
    aggregator: Arc<Vec<Mutex<HistogramDetail>>>,

    /// Print `Histogram`.
    printer: PrintHistogram,
}

impl HistogramAggregator {
    /// The `receiver` and `histogram_types` must be provided when construct `HistogramAggregator`.
    pub fn new(
        receiver: mpsc::Receiver<HistogramDetail>,
        histogram_types: &Vec<HistogramType>,
    ) -> Self {
        let aggregator = Self::get_all_histogram_details();

        let followed_histogram_indexes: Vec<usize> = histogram_types
            .iter()
            .map(|x| Into::<usize>::into(x.clone()))
            .collect();

        let printer = PrintHistogram::new(30, aggregator.clone(), followed_histogram_indexes);

        Self {
            receiver,
            aggregator,
            printer,
        }
    }

    /// Get `HistogramDetail` for all `HistogramType`, stored in a `Arc<Vec>`.
    pub fn get_all_histogram_details() -> Arc<Vec<Mutex<HistogramDetail>>> {
        let size = HistogramType::COUNT;
        let mut details = Vec::with_capacity(size);

        for i in 0..size {
            details.push(Mutex::new(HistogramDetail::new(HistogramType::from(i))));
        }

        Arc::new(details)
    }

    /// Merge the detail with correspoding histogram by `HistogramType`.
    fn update_detail(&mut self, detail: &HistogramDetail) {
        let index = detail.get_histogram_type_int();

        if unlikely(index >= self.aggregator.len()) {
            error!(
                "out of range, index: {}, total enum: {}",
                index,
                HistogramType::COUNT
            );
            return;
        }

        let mut x = self.aggregator[index].lock().unwrap();
        x.merge(detail);
    }

    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        let printer = self.printer.clone();
        subsys.start(SubsystemBuilder::new("print_histogram", |a| printer.run(a)));

        loop {
            tokio::select! {
                detail = self.receiver.recv() => {
                    match detail {
                        Some(x) => {
                            self.update_detail(&x);
                        },
                        None => {
                        }
                    }
                },
                _ = subsys.on_shutdown_requested() => {
                    info!("HistogramAggregator shutdown!");
                    return Ok(());
                }
            }
        }
    }
}

/// Add time spend to `histogram`.
#[inline]
pub fn record_time(histogram: &mut Histogram, histogram_type: HistogramType, last: &mut Instant) {
    let elapsed = Instant::now().duration_since(last.clone());

    histogram.add(histogram_type, elapsed.as_micros());

    *last = Instant::now();
}

/// Fake type to workaround with `Default`.
///
/// TODO: rm this trait.
pub trait WithHistogram {
    fn with_histogram(histogram: Histogram) -> Self;
}
