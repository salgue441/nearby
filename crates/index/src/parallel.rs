//! A minimal parallel map on scoped standard-library threads.

use std::num::NonZeroUsize;
use std::panic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Computes `f(0), f(1), ..., f(count - 1)` on up to `threads` threads and
/// returns the results in index order.
///
/// Indexes are handed out one at a time, so items of uneven cost still keep
/// every thread busy. With one thread or at most one item, `f` runs on the
/// calling thread and nothing is spawned. A panic in `f` is propagated.
pub(crate) fn parallel_map<T: Send>(
    count: usize,
    threads: NonZeroUsize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let threads = threads.get().min(count);
    if threads <= 1 {
        return (0..count).map(f).collect();
    }

    let next = AtomicUsize::new(0);
    let (next, f) = (&next, &f);
    let mut results: Vec<(usize, T)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= count {
                            return done;
                        }
                        done.push((i, f(i)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap_or_else(|p| panic::resume_unwind(p)))
            .collect()
    });
    results.sort_unstable_by_key(|&(i, _)| i);
    results.into_iter().map(|(_, value)| value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn threads(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[test]
    fn results_come_back_in_index_order() {
        let expected: Vec<usize> = (0..1_000).map(|i| i * i).collect();
        for t in [1, 2, 7, 64] {
            assert_eq!(
                parallel_map(1_000, threads(t), |i| i * i),
                expected,
                "{t} threads"
            );
        }
    }

    #[test]
    fn handles_no_items_and_more_threads_than_items() {
        assert_eq!(parallel_map(0, threads(8), |i| i), Vec::<usize>::new());
        assert_eq!(parallel_map(3, threads(8), |i| i + 1), vec![1, 2, 3]);
    }

    #[test]
    fn runs_inline_with_one_thread() {
        let caller = thread::current().id();
        let ids = parallel_map(4, threads(1), |_| thread::current().id());
        assert!(ids.iter().all(|&id| id == caller));
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn propagates_panics() {
        parallel_map(10, threads(4), |i| assert!(i != 5, "boom"));
    }
}
