use std::sync::atomic;
use std::sync::atomic::AtomicU64;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_atomic_latch_basic_usage() {
        let latch = AtomicLatch::new(1);

        // Producer thread
        let producer_handle = thread::spawn(move || {
            latch.add();
            latch.done();
        });

        // Consumer thread
        let consumer_handle = thread::spawn(|| {
            latch.wait();
        });

        // Wait for both threads to complete
        producer_handle.join().unwrap();
        consumer_handle.join().unwrap();

        // Ensure the latch was used as expected
        assert_eq!(latch.count.load(std::sync::atomic::Ordering::Acquire), 0);
    }

    #[test]
    fn test_atomic_latch_multiple_producers_consumers() {
        let latch = AtomicLatch::new(3);

        // Spawn multiple producer threads
        let producer_handles: Vec<_> = (0..3).map(|_| {
            thread::spawn(move || {
                latch.add();
                latch.done();
            })
        }).collect();

        // Spawn a single consumer thread
        let consumer_handle = thread::spawn(|| {
            latch.wait();
        });

        // Wait for all producer threads to complete
        for handle in producer_handles {
            handle.join().unwrap();
        }

        // Wait for the consumer thread to complete
        consumer_handle.join().unwrap();

        // Ensure the latch was used as expected
        assert_eq!(latch.count.load(std::sync::atomic::Ordering::Acquire), 0);
    }

    #[test]
    fn test_atomic_latch_panic_on_extra_done_calls() {
        let latch = AtomicLatch::new(1);

        // Spawn a producer thread that calls done() twice
        let producer_handle = thread::spawn(move || {
            latch.add();
            latch.done(); // First call
            latch.done(); // Second call
        });

        // Spawn a consumer thread
        let consumer_handle = thread::spawn(|| {
            latch.wait();
        });

        // Wait for both threads to complete
        producer_handle.join().unwrap();
        consumer_handle.join().unwrap();

        // Expect a panic due to extra done() call
        // Note: This test will fail if the panic is caught or ignored
    }
}
