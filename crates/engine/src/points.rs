//! The id map: which slot holds each point's current vector.

use std::collections::HashMap;

use nearby_core::PointId;

use crate::payload::Payload;

/// Every slot a collection's index holds, a map from point id to the slot
/// with its current vector, and each point's payload. The vectors themselves
/// live in the index.
///
/// Indexes can only append, so replacing or deleting a point never touches
/// the index: the old slot is just marked dead, and searches skip it. Dead
/// slots are reclaimed by building a compacted table and a fresh index.
#[derive(Debug, Clone, Default)]
pub(crate) struct PointTable {
    /// `ids[s]` is the point slot `s` was written for.
    ids: Vec<PointId>,
    live: Vec<bool>,
    /// The live slot of every point.
    slots: HashMap<PointId, u32>,
    /// The payload of every live point that has one.
    payloads: HashMap<PointId, Payload>,
}

impl PointTable {
    pub(crate) fn new() -> Self {
        Self::default()
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

    /// Gives point `id` a new slot, which it returns, and `payload`. A slot
    /// the point had before becomes dead, and its old payload is dropped.
    pub(crate) fn upsert(&mut self, id: PointId, payload: Option<Payload>) -> u32 {
        match payload {
            Some(payload) => self.payloads.insert(id, payload),
            None => self.payloads.remove(&id),
        };
        let slot = u32::try_from(self.ids.len()).expect("fewer than 2^32 slots");
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
        self.payloads.remove(&id);
        self.live[slot as usize] = false;
        true
    }

    /// The slot holding point `id`'s current vector.
    pub(crate) fn slot(&self, id: PointId) -> Option<u32> {
        self.slots.get(&id).copied()
    }

    /// Point `id`'s payload, if it is live and has one.
    pub(crate) fn payload(&self, id: PointId) -> Option<&Payload> {
        self.payloads.get(&id)
    }

    /// Every payload, by point id.
    pub(crate) fn payloads(&self) -> &HashMap<PointId, Payload> {
        &self.payloads
    }

    /// The point in `slot`, if the slot is live.
    pub(crate) fn live_id(&self, slot: u32) -> Option<PointId> {
        let s = slot as usize;
        (*self.live.get(s)?).then(|| self.ids[s])
    }

    /// Every live slot, in order.
    pub(crate) fn live_slots(&self) -> impl Iterator<Item = u32> {
        (0..self.slot_count() as u32).filter(|&slot| self.live[slot as usize])
    }

    /// Every slot's point id and whether it is live, in slot order.
    pub(crate) fn slots(&self) -> impl Iterator<Item = (PointId, bool)> {
        self.ids.iter().copied().zip(self.live.iter().copied())
    }

    /// Rebuilds a table from [`PointTable::slots`] and
    /// [`PointTable::payloads`]. Payloads of points that aren't live are
    /// dropped.
    pub(crate) fn from_slots(
        slots: &[(PointId, bool)],
        mut payloads: HashMap<PointId, Payload>,
    ) -> Self {
        let mut table = Self::new();
        for &(id, live) in slots {
            let slot = table.upsert(id, None);
            if !live {
                table.live[slot as usize] = false;
                table.slots.remove(&id);
            }
        }
        payloads.retain(|id, _| table.slots.contains_key(id));
        table.payloads = payloads;
        table
    }

    /// A table with the same live points and no dead slots. Points keep
    /// their relative order, so slot `s` here is the `s`-th live slot there.
    pub(crate) fn compacted(&self) -> Self {
        let mut table = Self::new();
        for slot in self.live_slots() {
            table.upsert(self.ids[slot as usize], None);
        }
        table.payloads.clone_from(&self.payloads);
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upserts_replace_and_deletes_remove() {
        let mut t = PointTable::new();
        assert_eq!(t.upsert(10, None), 0);
        assert_eq!(t.upsert(20, None), 1);
        assert_eq!(t.upsert(10, None), 2, "a replacement gets a new slot");
        assert_eq!(t.slot(10), Some(2));
        assert_eq!((t.len(), t.slot_count(), t.dead()), (2, 3, 1));
        assert_eq!(t.live_id(0), None, "the replaced slot is dead");
        assert_eq!(t.live_id(2), Some(10));

        assert!(t.delete(20));
        assert!(!t.delete(20), "already gone");
        assert!(!t.delete(99), "never existed");
        assert_eq!(t.slot(20), None);
        assert_eq!((t.len(), t.dead()), (1, 2));
        assert_eq!(t.live_id(7), None, "out of range");
    }

    #[test]
    fn compaction_keeps_live_points_in_order() {
        let mut t = PointTable::new();
        for id in 0..5 {
            t.upsert(id, None);
        }
        t.delete(1);
        t.upsert(3, None);
        let c = t.compacted();
        assert_eq!((c.len(), c.dead()), (4, 0));
        let ids: Vec<PointId> = (0..4).map(|s| c.live_id(s).unwrap()).collect();
        assert_eq!(ids, vec![0, 2, 4, 3]);
        assert_eq!(c.slot(3), Some(3));
    }

    #[test]
    fn slots_round_trip_through_from_slots() {
        let mut t = PointTable::new();
        t.upsert(1, None);
        t.upsert(2, None);
        t.upsert(1, None);
        t.delete(2);
        let slots: Vec<(PointId, bool)> = t.slots().collect();
        assert_eq!(slots, vec![(1, false), (2, false), (1, true)]);
        let r = PointTable::from_slots(&slots, HashMap::new());
        assert_eq!((r.len(), r.dead()), (1, 2));
        assert_eq!(r.slot(1), Some(2));
        assert_eq!(r.live_id(2), Some(1));
    }

    #[test]
    fn payloads_follow_their_point() {
        use crate::payload::Value;

        let tagged = |tag: &str| {
            Some(Payload::from([(
                "tag".to_owned(),
                Value::String(tag.to_owned()),
            )]))
        };
        let mut t = PointTable::new();
        t.upsert(1, tagged("a"));
        t.upsert(2, tagged("b"));
        t.upsert(3, None);
        assert_eq!(t.payload(1), tagged("a").as_ref());
        assert_eq!(t.payload(3), None);

        t.upsert(1, None);
        assert_eq!(t.payload(1), None, "a replacement drops the old payload");
        t.delete(2);
        assert_eq!(t.payload(2), None);
        t.upsert(3, tagged("c"));
        assert_eq!(t.compacted().payload(3), tagged("c").as_ref());

        let slots: Vec<(PointId, bool)> = t.slots().collect();
        let mut payloads = t.payloads().clone();
        payloads.insert(2, tagged("stale").unwrap());
        let r = PointTable::from_slots(&slots, payloads);
        assert_eq!(r.payload(3), tagged("c").as_ref());
        assert_eq!(r.payload(2), None, "payloads of dead points are dropped");
    }

    #[test]
    fn live_slots_skip_dead_ones() {
        let mut t = PointTable::new();
        t.upsert(1, None);
        t.upsert(2, None);
        t.delete(1);
        assert_eq!(t.live_slots().collect::<Vec<_>>(), vec![1]);
    }
}
