//! Recovery through the public API: whatever mix of writes, flushes,
//! compactions and reopenings happens, reopening rebuilds exactly the points
//! an in-memory model holds.

use std::collections::BTreeMap;
use std::path::Path;

use nearby_core::PointId;
use nearby_storage::{CollectionStore, Durability, Op, StoreOptions};

const DIM: usize = 3;

type Points = BTreeMap<PointId, Vec<f32>>;

fn options() -> StoreOptions {
    StoreOptions {
        durability: Durability::NoSync,
    }
}

fn reopen(dir: &Path) -> (CollectionStore, Points) {
    let mut points = Points::new();
    let store = CollectionStore::open(dir, options(), |op| match op {
        Op::Upsert { id, vector } => {
            points.insert(id, vector.to_vec());
        }
        Op::Delete { id } => {
            points.remove(&id);
        }
    })
    .unwrap();
    (store, points)
}

/// A tiny deterministic generator, so the test needs no extra crate.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

#[test]
fn random_histories_recover_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = CollectionStore::create(dir.path(), DIM, options()).unwrap();
    let mut model = Points::new();
    let mut rng = Lcg(42);

    for step in 0..2_000 {
        match rng.below(100) {
            0..=59 => {
                let id = rng.below(200);
                let vector = vec![step as f32, id as f32, 0.5];
                store
                    .write(&[Op::Upsert {
                        id,
                        vector: &vector,
                    }])
                    .unwrap();
                model.insert(id, vector);
            }
            60..=89 => {
                let id = rng.below(200);
                store.write(&[Op::Delete { id }]).unwrap();
                model.remove(&id);
            }
            90..=94 => store.flush().unwrap(),
            95..=97 => store.compact().unwrap(),
            _ => {
                drop(store);
                let (reopened, points) = reopen(dir.path());
                assert_eq!(points, model, "after reopening at step {step}");
                store = reopened;
            }
        }
    }
    drop(store);
    assert_eq!(reopen(dir.path()).1, model);
}

#[test]
fn a_batch_write_is_replayed_whole() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = CollectionStore::create(dir.path(), DIM, options()).unwrap();
    let ops: Vec<Op<'_>> = (0..100)
        .map(|id| Op::Upsert {
            id,
            vector: &[1.0, 2.0, 3.0],
        })
        .collect();
    store.write(&ops).unwrap();
    assert_eq!(store.wal_ops(), 100);
    drop(store);
    assert_eq!(reopen(dir.path()).1.len(), 100);
}
