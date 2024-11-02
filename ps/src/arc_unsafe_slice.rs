use std::sync::Arc;

use sync_unsafe_cell::SyncUnsafeCell;

/// Arc unsafe cell of `Slice`, no lock.
///
/// Used for push grad.
pub struct ArcUnsafeSlice<T: Sized> {
    /// Inner value.
    value: Arc<SyncUnsafeCell<(*const T, usize)>>,
}

unsafe impl<T: Sized + Send> Send for ArcUnsafeSlice<T> {}
unsafe impl<T: Sized + Sync> Sync for ArcUnsafeSlice<T> {}

impl<T: Sized> Clone for ArcUnsafeSlice<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

impl<T: Sized> ArcUnsafeSlice<T> {
    /// Create a new `ArcUnsafeSlice` from a slice.
    pub fn new(ptr: *const T, len: usize) -> Self {
        Self {
            value: Arc::new(SyncUnsafeCell::new((ptr, len))),
        }
    }

    /// Get the slice.
    pub fn get(&self) -> &[T] {
        unsafe {
            let (ptr, len) = *self.value.get();
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Get the length of the slice.
    pub fn len(&self) -> usize {
        unsafe {
            let (_, len) = *self.value.get();
            len
        }
    }

    /// Get the pointer of the slice.
    pub fn ptr(&self) -> *const T {
        unsafe {
            let (ptr, _) = *self.value.get();
            ptr
        }
    }
}
