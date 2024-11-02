//! Functions using simd.
use std::simd::prelude::SimdFloat;
use std::simd::Simd;
use std::simd::StdFloat;

use anyhow::{bail, Result};
use log::error;

use likely_stable::unlikely;

use crate::error_bail;

/// Using simd to speedup f32 vector sum.
///
/// The length must be exactly equal to `N`. The value is directly added to the first parameter.
#[inline]
pub fn sum_f32_vectors_simd_no_copy<const N: usize>(a: &mut Simd<f32, N>, b: &[f32])
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    std::simd::Simd<f32, N>: SimdFloat,
{
    let b_chunk = Simd::<f32, N>::from_slice(b);
    *a += b_chunk;
}

/// Using simd to speedup v32 vector sum.
///
/// The length must be exactly equal to `N`.
#[inline]
pub fn sum_f32_vectors_simd<const N: usize>(a: &mut [f32], b: &[f32])
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    std::simd::Simd<f32, N>: SimdFloat,
{
    let mut a_chunk = Simd::<f32, N>::from_slice(a);
    let b_chunk = Simd::<f32, N>::from_slice(b);

    a_chunk += b_chunk;

    a_chunk.copy_to_slice(a);
}

/// Using simd to speedup f32 vector sum.
///
/// Support the situation where length is multiply of `N`.
///
/// The length of `a` and `b` must be same.
#[inline]
pub fn sum_f32_vectors_simd_flex<const N: usize>(a: &mut [f32], b: &[f32])
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    std::simd::Simd<f32, N>: SimdFloat,
{
    let len = a.len();
    let simd_len = len - (len % N);

    for i in (0..simd_len).step_by(N) {
        let a_chunk = Simd::<f32, N>::from_slice(&a[i..i + N]);
        let b_chunk = Simd::<f32, N>::from_slice(&b[i..i + N]);
        let sum = a_chunk + b_chunk;
        sum.copy_to_slice(&mut a[i..i + N]);
    }

    // Handle remaining elements
    for i in simd_len..len {
        a[i] += b[i];
    }
}

// Scalar fallback implementation remains the same.
//
/// The length of `a` and `b` must be same.
fn sum_f32_vectors_scalar(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

pub fn sum_f32_vectors_flex(a: &mut [f32], b: &[f32]) {
    if is_x86_feature_detected!("avx2") {
        sum_f32_vectors_simd_flex::<8>(a, b)
    } else if is_x86_feature_detected!("sse2") {
        sum_f32_vectors_simd_flex::<4>(a, b)
    } else {
        sum_f32_vectors_scalar(a, b)
    }
}

/// Update weights using the Adagrad optimization algorithm with SIMD acceleration.
///
/// # Parameters
/// - `N`: The width of the SIMD vector. Must be a supported lane count.
///
/// # Arguments
/// - `w`: Mutable slice of weights to be updated.
/// - `g`: Mutable slice of accumulated squared gradients.
/// - `gradient`: Slice of current gradients.
/// - `learning_rate`: The learning rate for the update.
/// - `epsilon`: Small constant for numerical stability.
///
/// # Panics
/// Panics if the lengths of `w`, `g`, and `gradient` are not equal.
pub fn adagrad_update<const N: usize>(
    w: &mut [f32],
    g: &mut [f32],
    gradient: &[f32],
    learning_rate: f32,
    epsilon: f32,
) -> Result<()>
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
    Simd<f32, N>: SimdFloat,
{
    // Ensure all input slices have the same length
    if unlikely(w.len() != g.len()) {
        error_bail!("Weights and accumulated gradients must have the same length");
    }

    if unlikely(w.len() != gradient.len()) {
        error_bail!("Weights and gradient must have the same length");
    }

    let len = w.len();
    let simd_len = len - (len % N);

    // SIMD update for chunks of size N
    for i in (0..simd_len).step_by(N) {
        // Load chunks of data into SIMD vectors
        let w_chunk = Simd::from_slice(&w[i..i + N]);
        let g_chunk = Simd::from_slice(&g[i..i + N]);
        let grad_chunk = Simd::from_slice(&gradient[i..i + N]);

        // Update accumulated squared gradient
        let new_g_chunk = g_chunk + grad_chunk * grad_chunk;
        new_g_chunk.copy_to_slice(&mut g[i..i + N]);

        // Compute update
        let lr = Simd::splat(learning_rate);
        let eps = Simd::splat(epsilon);
        let update = (lr * grad_chunk) / (new_g_chunk.sqrt() + eps);

        // Apply update to weights
        let new_w_chunk = w_chunk - update;
        new_w_chunk.copy_to_slice(&mut w[i..i + N]);
    }

    // Handle remaining elements with scalar operations
    for i in simd_len..len {
        // Update accumulated squared gradient
        g[i] += gradient[i] * gradient[i];

        // Compute and apply update
        let update = (learning_rate * gradient[i]) / (g[i].sqrt() + epsilon);
        w[i] -= update;
    }

    Ok(())
}
