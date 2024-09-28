use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use log::{error, info};
use std::ops::Range;

use sync_unsafe_cell::SyncUnsafeCell;

/// Arc unsafe cell of `Vec`, no lock.
///
/// Used for embedding lookup result storing.
#[derive(Default)]
pub struct ArcUnsafeVec<T: Sized> {
    /// Inner value.
    value: Arc<SyncUnsafeCell<Vec<T>>>,
}

unsafe impl<T: Sized + Sync> Sync for ArcUnsafeVec<T> {}

impl<T: Sized> Clone for ArcUnsafeVec<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

impl<T: Sized> ArcUnsafeVec<T> {
    /// Constructs a new `ArcUnsafeVec` which will wrap the inner value.
    #[inline]
    pub fn new() -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(Vec::new())),
        }
    }

    #[inline]
    pub fn from_vec(v: Vec<T>) -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(v)),
        }
    }

    /// Construct a new `ArcUnsafeVec` with `capacity`.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(Vec::with_capacity(capacity))),
        }
    }

    /// Get an immutable reference of the inner `Vec`.
    #[inline]
    pub fn as_vec(&self) -> &Vec<T> {
        unsafe { &*self.value.get() }
    }

    /// Get a mutable reference of the inner `Vec`.
    #[inline]
    pub fn as_vec_mut(&self) -> &mut Vec<T> {
        unsafe { &mut *self.value.get() }
    }

    /// Get immutable reference of inner element of `Vec` at index.
    ///
    /// If index is bigger than `Vec` size, return `None`.
    #[inline]
    pub fn get_element(&self, index: usize) -> Option<&T> {
        let vec = self.as_vec();

        if index < vec.len() {
            unsafe { Some(&self.as_vec()[index]) }
        } else {
            None
        }
    }

    /// Get mutable reference of inner element of `Vec` at index.
    ///
    /// If index is bigger than `Vec` size, return `None`.
    #[inline]
    pub fn get_element_mut(&self, index: usize) -> Option<&mut T> {
        let vec = self.as_vec();

        if index < vec.len() {
            unsafe { Some(&mut self.as_vec_mut()[index]) }
        } else {
            None
        }
    }

    /// Get immutable reference of inner element of `Vec` at index.
    ///
    /// Uncheck index.
    #[inline]
    pub fn get_element_unchecked(&self, index: usize) -> &T {
        unsafe { &self.as_vec()[index] }
    }

    /// Get mutable reference of inner element of `Vec` at index.
    ///
    /// Uncheck index.
    #[inline]
    pub fn get_element_mut_unchecked(&self, index: usize) -> &mut T {
        unsafe { &mut self.as_vec_mut()[index] }
    }

    /// Get len of inner `Vec`.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_vec().len()
    }

    /// Push an element to inner `Vec`.
    #[inline]
    pub fn push(&self, v: T) {
        self.as_vec_mut().push(v);
    }

    /// Get slice between `start..end`.
    #[inline]
    pub fn get_slice(&self, start: usize, end: usize) -> &[T] {
        &self.as_vec()[start..end]
    }

    /// Get slice between `start..end`.
    #[inline]
    pub fn get_mut_slice(&self, start: usize, end: usize) -> &mut [T] {
        &mut self.as_vec_mut()[start..end]
    }

    /// Get slice of the inner `Vec`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.as_vec().as_slice()
    }

    /// Get mutable slice of the inner `Vec`.
    #[inline]
    pub fn as_mut_slice(&self) -> &mut [T] {
        self.as_vec_mut().as_mut_slice()
    }
}

/// Arc unsafe cell of `Vec`, no lock.
///
/// The inner value of `Vec` is also `SyncUnsafeCell`.
///
/// Used for speeding up ps parameters accessing.
pub struct ArcInnerUnsafeVec<T: Sized> {
    /// Inner value.
    value: Arc<SyncUnsafeCell<Vec<SyncUnsafeCell<T>>>>,
}

unsafe impl<T: Sized + Sync> Sync for ArcInnerUnsafeVec<T> {}

impl<T: Sized> Clone for ArcInnerUnsafeVec<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

impl<T: Sized> ArcInnerUnsafeVec<T> {
    /// Constructs a new `ArcInnerUnsafeVec` which will wrap the inner value.
    #[inline]
    pub fn new() -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(Vec::new())),
        }
    }

    /// Construct a new `ArcInnerUnsafeVec` with `capacity`.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new(Vec::with_capacity(capacity))),
        }
    }

    /// Get an immutable reference of the inner `Vec`.
    #[inline]
    pub fn get_vec(&self) -> &Vec<SyncUnsafeCell<T>> {
        unsafe { &*self.value.get() }
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
            unsafe { Some(&*(self.get_vec()[index].get())) }
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
        unsafe { &*(self.get_vec()[index].get()) }
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
