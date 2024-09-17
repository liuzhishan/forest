use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use log::{error, info};

use sync_unsafe_cell::SyncUnsafeCell;

/// Arc unsafe cell of `Vec`, no lock.
///
/// Used for speeding up ps parameters accessing.
pub struct ArcUnsafeVec<T: Sized> {
    /// Inner value.
    value: Arc<SyncUnsafeCell<Vec<SyncUnsafeCell<T>>>>,
}

unsafe impl<T: Sized + Sync> Sync for ArcUnsafeVec<T> {}

impl<T: Sized> Clone for ArcUnsafeVec<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone()
        }
    }
}

impl<T: Sized> ArcUnsafeVec<T> {
    /// Constructs a new `ArcUnsafeVec` which will wrap the inner value.
    #[inline]
    pub fn new() -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(Vec::new()))
        }
    }

    /// Construct a new `ArcUnsafeVec` with `capacity`.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(Vec::with_capacity(capacity)))
        }
    }

    /// Get an immutable reference of the inner `Vec`.
    #[inline]
    pub fn get_vec(&self) -> &Vec<SyncUnsafeCell<T>> {
        unsafe { & *self.value.get() }
    }

    /// Get a mutable reference of the inner `Vec`.
    #[inline]
    pub fn get_vec_mut(&self) -> &mut Vec<SyncUnsafeCell<T>> {
        unsafe { &mut *self.value.get() }
    }

    /// Get immutable reference of inner element of `Vec` at index.
    ///
    /// If index is bigger than `Vec` size, return `None`.
    #[inline]
    pub fn get_element(&self, index: usize) -> Option<&T> {
        let vec = self.get_vec();

        if index < vec.len() {
            unsafe { Some(& *(self.get_vec()[index].get())) }
        } else {
            None
        }
    }

    /// Get mutable reference of inner element of `Vec` at index.
    ///
    /// If index is bigger than `Vec` size, return `None`.
    #[inline]
    pub fn get_element_mut(&self, index: usize) -> Option<&mut T> {
        let vec = self.get_vec();

        if index < vec.len() {
            unsafe { Some(&mut *(self.get_vec_mut()[index].get())) }
        } else {
            None
        }
    }

    /// Get immutable reference of inner element of `Vec` at index.
    ///
    /// Uncheck index.
    #[inline]
    pub fn get_element_unchecked(&self, index: usize) -> &T {
        unsafe { & *(self.get_vec()[index].get()) }
    }

    /// Get mutable reference of inner element of `Vec` at index.
    ///
    /// Uncheck index.
    #[inline]
    pub fn get_element_mut_unchecked(&self, index: usize) -> &mut T {
        unsafe { &mut *(self.get_vec_mut()[index].get()) }
    }

    /// Get len of inner `Vec`.
    #[inline]
    pub fn len(&self) -> usize {
        self.get_vec().len()
    }

    /// Push an element to inner `Vec`.
    #[inline]
    pub fn push(&self, v: T) {
        self.get_vec_mut().push(SyncUnsafeCell::new(v));
    }
}
