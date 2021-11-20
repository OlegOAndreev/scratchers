use std::{alloc, mem};
use std::sync::atomic;
use std::sync::atomic::AtomicBool;

use crate::atomic_latch::AtomicLatch;

// An adapter for ISPC task system, which spawns tasks via Rayon.

struct Handle {
    allocs: crossbeam::queue::SegQueue<Allocation>,
    latch: AtomicLatch,
    dropped: AtomicBool,
}

struct Allocation {
    ptr: *mut u8,
    layout: alloc::Layout,
}

impl Handle {
    fn new() -> Self {
        Self {
            allocs: crossbeam::queue::SegQueue::new(),
            latch: AtomicLatch::new(0),
            dropped: AtomicBool::new(false),
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.dropped.swap(true, atomic::Ordering::SeqCst) {
            panic!("Handle dropped multiple times");
        }
        while let Some(allocation) = self.allocs.pop() {
            unsafe { alloc::dealloc(allocation.ptr, allocation.layout) };
        }
    }
}

type TaskFn = extern "C" fn(
    data: *mut libc::c_void,
    thread_index: libc::c_int,
    thread_count: libc::c_int,
    task_index: libc::c_int,
    task_count: libc::c_int,
    task_index0: libc::c_int,
    task_index1: libc::c_int,
    task_index2: libc::c_int,
    task_count0: libc::c_int,
    task_count1: libc::c_int,
    task_count2: libc::c_int,
);

#[derive(Copy, Clone)]
struct TaskData {
    data: *mut libc::c_void,
    latch: *const AtomicLatch,
}

unsafe impl Send for TaskData {}

unsafe impl Sync for TaskData {}

#[allow(non_snake_case)]
#[no_mangle]
unsafe extern "C" fn ISPCAlloc(
    handle_ptr: *mut *mut libc::c_void,
    size: i64,
    align: i32,
) -> *mut libc::c_void {
    let handle = get_or_alloc_handle(handle_ptr);
    let layout = alloc::Layout::from_size_align(
        size as usize,
        align as usize,
    ).unwrap();
    let ptr = alloc::alloc(layout);
    handle.allocs.push(Allocation { ptr, layout });
    mem::forget(handle);
    ptr as *mut libc::c_void
}

#[allow(non_snake_case)]
#[no_mangle]
unsafe extern "C" fn ISPCLaunch(
    handle_ptr: *mut *mut libc::c_void,
    f: *mut libc::c_void,
    data: *mut libc::c_void,
    count0: libc::c_int,
    count1: libc::c_int,
    count2: libc::c_int,
) {
    let handle = get_or_alloc_handle(handle_ptr);
    let task_fn: TaskFn = mem::transmute(f);
    let task_data = TaskData {
        data,
        latch: &handle.latch as *const AtomicLatch,
    };
    let task_count = count0 * count1 * count2;
    for task_index0 in 0..count0 {
        for task_index1 in 0..count1 {
            for task_index2 in 0..count2 {
                let task_index = task_index0 + count0 * (task_index1 + count1 * task_index2);
                handle.latch.add();
                rayon::spawn(move || {
                    // NOTE: We have to do this, otherwise the closure will try to capture
                    // inidividual fields and fail due to them being non-Send+Sync.
                    let task_data = task_data;
                    task_fn(
                        task_data.data,
                        rayon::current_thread_index().unwrap() as libc::c_int,
                        rayon::current_num_threads() as libc::c_int,
                        task_index,
                        task_count,
                        task_index0,
                        count0,
                        task_index1,
                        count1,
                        task_index2,
                        count2,
                    );
                    let latch = &*task_data.latch as &AtomicLatch;
                    latch.done();
                });
            }
        }
    }

    mem::forget(handle)
}

#[allow(non_snake_case)]
#[no_mangle]
unsafe extern "C" fn ISPCSync(handle: *mut libc::c_void) {
    let handle = Box::<Handle>::from_raw(handle as *mut Handle);
    handle.latch.wait();
    // Handle::drop() deals with the cleanup.
}

// Do not forget to mem::forget(handle) at the end of the function!
unsafe fn get_or_alloc_handle(handle_ptr: *mut *mut libc::c_void) -> Box<Handle> {
    if (*handle_ptr).is_null() {
        let ptr = Box::into_raw(Box::new(Handle::new()));
        *handle_ptr = ptr as *mut libc::c_void;
    };
    Box::from_raw(*handle_ptr as *mut Handle)
}

