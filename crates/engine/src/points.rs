//! The id map: which slot holds each point's current vector.

use std::collections::HashMap;

use nearby_core::PointId;

/// Every vector a collection's index holds, by slot, plus a map from point
/// id to the slot with its current vector.
///
/// Indexes can only append, so replacing or deleting a point never touches
/// the index: the old slot is just marked dead, and searches skip it. Dead
/// slots are reclaimed by building a compacted table and a fresh index.
#[derive(Debug, Clone)]
pub(crate) struct PointTable {
    dim: usize,
    /// Slot `s`'s vector is `vectors[s * dim..(s + 1) * dim]`.
    vectors: Vec<f32>,
    /// `ids[s]` is the point slot `s` was written for.
    ids: Vec<PointId>,
    live: Vec<bool>,
    /// The live slot of every point.
    slots: HashMap<PointId, u32>,
}

impl PointTable {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            live: Vec::new(),
            slots: HashMap::new(),
        }
    }

    /// Number of live points.
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Number of slots, live or dead.
    pub(crate) fn slot_count(&self) -> usize {
        self.ids.len()
    }

    /// Number of dead slots.
    pub(crate) fn dead(&self) -> usize {
        self.slot_count() - self.len()
    }

    /// Stores `vector` as point `id` in a new slot, which it returns. A slot
    /// the point had before becomes dead.
    pub(crate) fn upsert(&mut self, id: PointId, vector: &[f32]) -> u32 {
        debug_assert_eq!(vector.len(), self.dim, "caller validates vectors");
        let slot = u32::try_from(self.ids.len()).expect("fewer than 2^32 slots");
        self.vectors.extend_from_slice(vector);
        self.ids.push(id);
        self.live.push(true);
        if let Some(old) = self.slots.insert(id, slot) {
            self.live[old as usize] = false;
        }
        slot
    }

    /// Removes point `id`. Returns whether it existed.
    pub(crate) fn delete(&mut self, id: PointId) -> bool {
        let Some(slot) = self.slots.remove(&id) else {
            return false;
        };
        self.live[slot as usize] = false;
        true
    }

    /// Point `id`'s current vector.
    pub(crate) fn get(&self, id: PointId) -> Option<&[f32]> {
        self.slots.get(&id).map(|&slot| self.vector(slot))
    }

    /// The point in `slot`, if the slot is live.
    pub(crate) fn live_id(&self, slot: u32) -> Option<PointId> {
        let s = slot as usize;
        (*self.live.get(s)?).then(|| self.ids[s])
    }

    fn vector(&self, slot: u32) -> &[f32] {
        let s = slot as usize;
        &self.vectors[s * self.dim..(s + 1) * self.dim]
    }

    /// Every live point with its slot, in slot order.
    pub(crate) fn live_slots(&self) -> impl Iterator<Item = (u32, &[f32])> {
        (0..self.slot_count() as u32)
            .filter(|&slot| self.live[slot as usize])
            .map(|slot| (slot, self.vector(slot)))
    }

    /// Every slot's point id and whether it is live, in slot order.
    pub(crate) fn slots(&self) -> impl Iterator<Item = (PointId, bool)> {
        self.ids.iter().copied().zip(self.live.iter().copied())
    }

    /// Rebuilds a table from [`PointTable::slots`] and each slot's vector.
    ///
    /// # Panics
    ///
    /// If `vectors` doesn't yield one `dim`-long vector per slot.
    pub(crate) fn from_slots<'a>(
        dim: usize,
        slots: &[(PointId, bool)],
        vectors: impl Iterator<Item = &'a [f32]>,
    ) -> Self {
        let mut table = Self::new(dim);
        for (&(id, live), vector) in slots.iter().zip(vectors) {
            assert_eq!(vector.len(), dim, "vectors must have the table's dimension");
            let slot = u32::try_from(table.ids.len()).expect("fewer than 2^32 slots");
            table.vectors.extend_from_slice(vector);
            table.ids.push(id);
            table.live.push(live);
            if live {
                table.slots.insert(id, slot);
            }
        }
        assert_eq!(table.ids.len(), slots.len(), "one vector per slot");
        table
    }

    /// A table with the same live points and no dead slots. Points keep
    /// their relative order, so slot `s` here is the `s`-th live slot there.
    pub(crate) fn compacted(&self) -> Self {
        let mut table = Self::new(self.dim);
        for (slot, vector) in self.live_slots() {
            table.upsert(self.ids[slot as usize], vector);
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upserts_replace_and_deletes_remove() {
        let mut t = PointTable::new(2);
        assert_eq!(t.upsert(10, &[1.0, 1.0]), 0);
        assert_eq!(t.upsert(20, &[2.0, 2.0]), 1);
        assert_eq!(
            t.upsert(10, &[3.0, 3.0]),
            2,
            "a replacement gets a new slot"
        );
        assert_eq!(t.get(10), Some(&[3.0, 3.0][..]));
        assert_eq!((t.len(), t.slot_count(), t.dead()), (2, 3, 1));
        assert_eq!(t.live_id(0), None, "the replaced slot is dead");
        assert_eq!(t.live_id(2), Some(10));

        assert!(t.delete(20));
        assert!(!t.delete(20), "already gone");
        assert!(!t.delete(99), "never existed");
        assert_eq!(t.get(20), None);
        assert_eq!((t.len(), t.dead()), (1, 2));
        assert_eq!(t.live_id(7), None, "out of range");
    }

    #[test]
    fn compaction_keeps_live_points_in_order() {
        let mut t = PointTable::new(1);
        for id in 0..5 {
            t.upsert(id, &[id as f32]);
        }
        t.delete(1);
        t.upsert(3, &[30.0]);
        let c = t.compacted();
        assert_eq!((c.len(), c.dead()), (4, 0));
        let ids: Vec<PointId> = (0..4).map(|s| c.live_id(s).unwrap()).collect();
        assert_eq!(ids, vec![0, 2, 4, 3]);
        assert_eq!(c.get(3), Some(&[30.0][..]));
    }

    #[test]
    fn slots_round_trip_through_from_slots() {
        let mut t = PointTable::new(1);
        t.upsert(1, &[1.0]);
        t.upsert(2, &[2.0]);
        t.upsert(1, &[3.0]);
        t.delete(2);
        let slots: Vec<(PointId, bool)> = t.slots().collect();
        assert_eq!(slots, vec![(1, false), (2, false), (1, true)]);
        let vectors = [[1.0], [2.0], [3.0]];
        let r = PointTable::from_slots(1, &slots, vectors.iter().map(|v| &v[..]));
        assert_eq!((r.len(), r.dead()), (1, 2));
        assert_eq!(r.get(1), Some(&[3.0][..]));
        assert_eq!(r.live_id(2), Some(1));
    }

    #[test]
    fn live_slots_skip_dead_ones() {
        let mut t = PointTable::new(1);
        t.upsert(1, &[1.0]);
        t.upsert(2, &[2.0]);
        t.delete(1);
        let live: Vec<(u32, Vec<f32>)> = t.live_slots().map(|(s, v)| (s, v.to_vec())).collect();
        assert_eq!(live, vec![(1, vec![2.0])]);
    }
}
