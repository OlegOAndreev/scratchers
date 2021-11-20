use std::sync::atomic;
use std::sync::atomic::{AtomicU64};

use parking_lot::{Condvar, Mutex};

// This struct is similar to Barrier or crossbeam::WaitGroup, but uses atomic in fast path.
pub struct AtomicLatch {
    count: AtomicU64,
    mutex: Mutex<()>,
    condvar: Condvar,
}

impl AtomicLatch {
    pub fn new(count: u64) -> Self {
        Self {
            count: AtomicU64::new(count),
            mutex: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }

    pub fn add(&self) {
        self.count.fetch_add(1, atomic::Ordering::AcqRel);
    }

    pub fn done(&self) {
        // We need to use Acquire-Release here because all producers running done() must synchronize
        // with consumers running wait().
        let v = self.count.fetch_sub(1, atomic::Ordering::Release);
        if v == 0 {
            panic!("done called more than count times")
        } else if v == 1 {
            // NOTE: This guard is critical for preventing TOCTOU error, without it the thread
            // running wait() may sleep between checking count and wait()ing and miss the
            // notification.
            let _guard = self.mutex.lock();
            self.condvar.notify_one();
        }
    }

    pub fn wait(&self) {
        let mut guard = self.mutex.lock();
        if self.count.load(atomic::Ordering::Acquire) == 0 {
            return;
        }
        self.condvar.wait(&mut guard);
    }
}
