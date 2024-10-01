#![feature(stdarch_x86_avx512)]
#![feature(portable_simd)]
use core::simd::prelude::*;

use std::arch::x86_64::*;
use std::mem::size_of;

use std::simd::LaneCount;
use std::simd::SimdElement;
use std::simd::StdFloat;
use std::simd::SupportedLaneCount;

use std::ops::{Add, AddAssign};

use coarsetime::Instant;
use hashbrown::HashMap;
use log::{error, info};

use rand::Rng;
use rand::RngCore;
use std::any::type_name;
use std::sync::Once;
use std::sync::{Arc, Mutex};
use util::simd::sum_f32_vectors_simd_flex;
use util::simd::sum_f32_vectors_simd_no_copy;

use anyhow::bail;
use anyhow::Result;

use std::io::{BufRead, BufReader, Read, Write};

use util::{error_bail, gen_random_f32_list, init_log, Flags};

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(|| {
        init_log();
    });
}

#[test]
fn test_util_setup() {
    info!("test util");
    setup();
}

#[test]
fn test_simd() {
    setup();

    const LANES: usize = 32;
    const THIRTEENS: Simd<u8, LANES> = Simd::<u8, LANES>::from_array([13; LANES]);
    const TWENTYSIXS: Simd<u8, LANES> = Simd::<u8, LANES>::from_array([26; LANES]);
    const ZEES: Simd<u8, LANES> = Simd::<u8, LANES>::from_array([b'Z'; LANES]);

    let mut data = Simd::<u8, LANES>::from_slice(b"URYYBJBEYQVQBUBCRVGFNYYTBVATJRYY");

    data += THIRTEENS;

    let mask = data.simd_gt(ZEES);
    data = mask.select(data - TWENTYSIXS, data);

    let output = String::from_utf8_lossy(data.as_array());

    info!("output: {}", output);
}

#[inline]
fn sum_float_list(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    for i in 0..n {
        a[i] += b[i];
    }
}

#[inline]
fn sum_float_list_simd_32<const N: usize>(a: &mut Vec<f32>, b: &[f32])
where
    LaneCount<N>: SupportedLaneCount,
    Simd<f32, N>: SimdFloat,
{
    let mut a_data = Simd::<f32, N>::from_slice(a.as_slice());
    let b_data = Simd::<f32, N>::from_slice(b);

    a_data += b_data;

    a_data.copy_to_slice(a.as_mut_slice());
}

/// Time a function in `milliseconds`.
#[inline]
fn time_loop<F>(n: usize, f: F) -> u64
where
    F: Fn() -> (),
{
    let now = Instant::now();

    for i in 0..n {
        f();
    }

    let time_spend = (Instant::now() - now).as_millis();
    time_spend
}

fn run_normal_sum(n: usize, a: &mut Vec<f32>, b: &Vec<f32>) {
    let now = Instant::now();

    for i in 0..n {
        sum_float_list(a.as_mut_slice(), b.as_slice());
    }

    let time_spend = (Instant::now() - now).as_millis();

    info!(
        "normal sum time spend: {} milliseconds, count: {}",
        time_spend, n
    );
}

/// Sum `Vec<f32>` using `mm256` simd intrinsics.
pub fn sum_f32_vectors_simd_mm256(a: &mut Vec<f32>, b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    // Ensure the vectors are aligned properly
    let (prefix, middle, suffix) = unsafe { a.align_to_mut::<__m256>() };
    let (prefix_b, middle_b, suffix_b) = unsafe { b.align_to::<__m256>() };

    // Handle unaligned prefix
    for (x, y) in prefix.iter_mut().zip(prefix_b.iter()) {
        *x += *y;
    }

    // Main SIMD loop
    unsafe {
        for (chunk_a, chunk_b) in middle.iter_mut().zip(middle_b.iter()) {
            *chunk_a = _mm256_add_ps(*chunk_a, *chunk_b);
        }
    }

    // Handle unaligned suffix
    for (x, y) in suffix.iter_mut().zip(suffix_b.iter()) {
        *x += *y;
    }
}

/// Sum `Vec<f32>` using `avx512` simd intrinsics.
pub fn sum_f32_vectors_simd_avx512(a: &mut Vec<f32>, b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let (prefix, middle, suffix) = unsafe { a.align_to_mut::<__m512>() };
    let (prefix_b, middle_b, suffix_b) = unsafe { b.align_to::<__m512>() };

    // Handle unaligned prefix
    for (x, y) in prefix.iter_mut().zip(prefix_b.iter()) {
        *x += *y;
    }

    // Main SIMD loop
    unsafe {
        for (chunk_a, chunk_b) in middle.iter_mut().zip(middle_b.iter()) {
            *chunk_a = _mm512_add_ps(*chunk_a, *chunk_b);
        }
    }

    // Handle unaligned suffix
    for (x, y) in suffix.iter_mut().zip(suffix_b.iter()) {
        *x += *y;
    }
}

fn run_simd_sum_flex<const N: usize>(n: usize, a: &mut Vec<f32>, b: &Vec<f32>)
where
    LaneCount<N>: SupportedLaneCount,
    Simd<f32, N>: Add + AddAssign,
{
    let now = Instant::now();

    for i in 0..n {
        sum_f32_vectors_simd_flex(a, b.as_slice());
    }

    let time_spend = (Instant::now() - now).as_millis();

    info!(
        "run_simd_sum_flex, copy to slice, {}x{} sum time spend: {} milliseconds, count: {}",
        type_name::<f32>(),
        N,
        time_spend,
        n
    );
}

fn run_simd_sum_f32_no_copy<const N: usize>(n: usize, a: &mut Vec<f32>, b: &Vec<f32>)
where
    LaneCount<N>: SupportedLaneCount,
    Simd<f32, N>: SimdFloat,
{
    let mut a_simd = Simd::<f32, N>::from_slice(a.as_slice());
    let mut a_mut = a.clone();

    let now = Instant::now();

    for i in 0..n {
        sum_f32_vectors_simd_no_copy::<N>(&mut a_simd, b);
    }

    let time_spend = (Instant::now() - now).as_millis();

    info!(
        "run_simd_sum_f32_no_copy, no copy, {}x{} sum time spend: {} milliseconds, count: {}",
        type_name::<f32>(),
        N,
        time_spend,
        n
    );
}

fn run_simd_sum_f32_mm256(n: usize, a: &mut Vec<f32>, b: &Vec<f32>) {
    let now = Instant::now();

    for i in 0..n {
        sum_f32_vectors_simd_mm256(a, b.as_slice());
    }

    let time_spend = (Instant::now() - now).as_millis();

    info!(
        "run_simd_sum_f32_mm256, use mm256 directly, sum time spend: {} milliseconds, count: {}",
        time_spend, n
    );
}

fn run_simd_sum_f32_avx512(n: usize, a: &mut Vec<f32>, b: &Vec<f32>) {
    let now = Instant::now();

    for i in 0..n {
        sum_f32_vectors_simd_avx512(a, b.as_slice());
    }

    let time_spend = (Instant::now() - now).as_millis();

    info!(
        "run_simd_sum_f32_avx512, user avx512, sum time spend: {} milliseconds, count: {}",
        time_spend, n
    );
}

#[test]
fn test_simd_sum() {
    setup();

    const N: usize = 32;

    let a = gen_random_f32_list(N);
    let mut a_mut = a.clone();

    let b = gen_random_f32_list(N);

    const LOOP_COUNT: usize = 10000000;

    // normal sum
    run_normal_sum(LOOP_COUNT, &mut a_mut, &b);

    // f32x4
    run_simd_sum_flex::<4>(LOOP_COUNT, &mut a_mut, &b);

    // f32x8
    run_simd_sum_flex::<8>(LOOP_COUNT, &mut a_mut, &b);

    // f32x16
    run_simd_sum_flex::<16>(LOOP_COUNT, &mut a_mut, &b);

    // Simd<f32, 32>
    run_simd_sum_flex::<32>(LOOP_COUNT, &mut a_mut, &b);

    // f32x16
    run_simd_sum_f32_no_copy::<16>(LOOP_COUNT, &mut a_mut, &b);

    // f32x32
    run_simd_sum_f32_no_copy::<32>(LOOP_COUNT, &mut a_mut, &b);

    // mm256
    run_simd_sum_f32_mm256(LOOP_COUNT, &mut a_mut, &b);

    // avx512
    run_simd_sum_f32_avx512(LOOP_COUNT, &mut a_mut, &b);
}
