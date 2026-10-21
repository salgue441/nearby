//! Payloads stored by field: one column of values per field name, indexed
//! by slot.
//!
//! A filtered search checks the payload of every slot it visits. Stored as
//! one map per point, each check follows several pointers into memory the
//! search hasn't touched, and costs more than the distance it guards. Here
//! a field's values sit in one dense array, so a check reads one 16-byte
//! cell, and the few columns a filter uses stay in cache. Strings are
//! interned per column, so comparing them compares integers.
//!
//! A field costs 16 bytes for every slot up to the last one that has it,
//! including slots whose points don't. Compaction rebuilds the columns,
//! dropping deleted values and strings no point uses anymore.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::filter::{Filter, Range};
use crate::payload::{Payload, Value};

/// Every slot's payload, column by column.
#[derive(Debug, Clone, Default)]
pub(crate) struct PayloadColumns {
    columns: BTreeMap<String, Column>,
}

/// One field's value in every slot.
#[derive(Debug, Clone, Default)]
struct Column {
    /// `cells[s]` is the value in slot `s`. Slots past the end are missing.
    cells: Vec<Cell>,
    /// `Cell::String(i)` holds `strings[i]`.
    strings: Vec<Arc<str>>,
    string_ids: HashMap<Arc<str>, u32>,
}

/// A [`Value`] with its string interned, or no value.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) enum Cell {
    #[default]
    Missing,
    Bool(bool),
    Number(f64),
    String(u32),
}

impl PayloadColumns {
    /// Stores `payload` as the payload of `slot`, which must have none.
    pub(crate) fn insert(&mut self, slot: u32, payload: &Payload) {
        for (field, value) in payload {
            if !self.columns.contains_key(field) {
                self.columns.insert(field.clone(), Column::default());
            }
            let column = self.columns.get_mut(field).expect("inserted above");
            column.set(slot, value);
        }
    }

    /// Drops the payload of `slot`.
    pub(crate) fn remove(&mut self, slot: u32) {
        for column in self.columns.values_mut() {
            if let Some(cell) = column.cells.get_mut(slot as usize) {
                *cell = Cell::Missing;
            }
        }
    }

    /// The payload of `slot`, if it has any fields.
    pub(crate) fn get(&self, slot: u32) -> Option<Payload> {
        let payload: Payload = self
            .columns
            .iter()
            .filter_map(|(field, column)| Some((field.clone(), column.value(slot)?)))
            .collect();
        (!payload.is_empty()).then_some(payload)
    }

    /// `filter` resolved against these columns, ready to check slots.
    pub(crate) fn compile(&self, filter: &Filter) -> CompiledFilter<'_> {
        let column = |field: &str| self.columns.get(field);
        match filter {
            Filter::Eq { field, value } => column(field)
                .and_then(|c| Some(CompiledFilter::Eq(&c.cells, c.lookup(value)?)))
                .unwrap_or(CompiledFilter::Const(false)),
            Filter::In { field, values } => {
                column(field).map_or(CompiledFilter::Const(false), |c| {
                    CompiledFilter::In(
                        &c.cells,
                        values.iter().filter_map(|v| c.lookup(v)).collect(),
                    )
                })
            }
            Filter::Range { field, range } => column(field)
                .map_or(CompiledFilter::Const(false), |c| {
                    CompiledFilter::Range(&c.cells, *range)
                }),
            Filter::Exists(field) => column(field).map_or(CompiledFilter::Const(false), |c| {
                CompiledFilter::Exists(&c.cells)
            }),
            Filter::And(filters) => {
                CompiledFilter::And(filters.iter().map(|f| self.compile(f)).collect())
            }
            Filter::Or(filters) => {
                CompiledFilter::Or(filters.iter().map(|f| self.compile(f)).collect())
            }
            Filter::Not(filter) => CompiledFilter::Not(Box::new(self.compile(filter))),
        }
    }
}

impl Column {
    fn set(&mut self, slot: u32, value: &Value) {
        let cell = match value {
            Value::Bool(b) => Cell::Bool(*b),
            Value::Number(x) => Cell::Number(*x),
            Value::String(s) => Cell::String(self.intern(s)),
        };
        let slot = slot as usize;
        if self.cells.len() <= slot {
            self.cells.resize(slot + 1, Cell::Missing);
        }
        self.cells[slot] = cell;
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_ids.get(s) {
            return id;
        }
        let id = u32::try_from(self.strings.len()).expect("fewer than 2^32 distinct strings");
        let s: Arc<str> = Arc::from(s);
        self.strings.push(Arc::clone(&s));
        self.string_ids.insert(s, id);
        id
    }

    fn value(&self, slot: u32) -> Option<Value> {
        match cell_at(&self.cells, slot) {
            Cell::Missing => None,
            Cell::Bool(b) => Some(Value::Bool(b)),
            Cell::Number(x) => Some(Value::Number(x)),
            Cell::String(id) => Some(Value::String(self.strings[id as usize].to_string())),
        }
    }

    /// The cell holding `value`, or `None` if no slot can hold it: a string
    /// this column has never stored.
    fn lookup(&self, value: &Value) -> Option<Cell> {
        match value {
            Value::Bool(b) => Some(Cell::Bool(*b)),
            Value::Number(x) => Some(Cell::Number(*x)),
            Value::String(s) => self.string_ids.get(s.as_str()).map(|&id| Cell::String(id)),
        }
    }
}

fn cell_at(cells: &[Cell], slot: u32) -> Cell {
    cells.get(slot as usize).copied().unwrap_or_default()
}

/// A [`Filter`] resolved against one [`PayloadColumns`]: field names become
/// columns and strings become interned ids once per search, instead of once
/// per slot checked.
#[derive(Debug)]
pub(crate) enum CompiledFilter<'a> {
    /// Matches every slot or none, such as a condition on a field no point
    /// has.
    Const(bool),
    Eq(&'a [Cell], Cell),
    In(&'a [Cell], Vec<Cell>),
    Range(&'a [Cell], Range),
    Exists(&'a [Cell]),
    And(Vec<Self>),
    Or(Vec<Self>),
    Not(Box<Self>),
}

impl CompiledFilter<'_> {
    /// Whether the payload of `slot` matches: the same answer as
    /// [`Filter::matches`] gives for it.
    pub(crate) fn matches(&self, slot: u32) -> bool {
        match self {
            Self::Const(matches) => *matches,
            Self::Eq(cells, target) => cell_at(cells, slot) == *target,
            Self::In(cells, targets) => targets.contains(&cell_at(cells, slot)),
            Self::Range(cells, range) => {
                matches!(cell_at(cells, slot), Cell::Number(x) if range.contains(x))
            }
            Self::Exists(cells) => cell_at(cells, slot) != Cell::Missing,
            Self::And(filters) => filters.iter().all(|f| f.matches(slot)),
            Self::Or(filters) => filters.iter().any(|f| f.matches(slot)),
            Self::Not(filter) => !filter.matches(slot),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(fields: &[(&str, Value)]) -> Payload {
        fields
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect()
    }

    fn text(s: &str) -> Value {
        Value::String(s.to_owned())
    }

    fn field(name: &str) -> String {
        name.to_owned()
    }

    #[test]
    fn payloads_round_trip() {
        let a = payload(&[("lang", text("en")), ("year", Value::Number(2024.0))]);
        let b = payload(&[("lang", text("en")), ("draft", Value::Bool(true))]);
        let mut columns = PayloadColumns::default();
        columns.insert(0, &a);
        columns.insert(3, &b);
        assert_eq!(columns.get(0), Some(a));
        assert_eq!(columns.get(1), None, "a slot without a payload");
        assert_eq!(columns.get(3), Some(b));
        assert_eq!(columns.get(9), None, "past every column");

        columns.remove(0);
        assert_eq!(columns.get(0), None);
        assert_eq!(columns.columns["lang"].strings.len(), 1, "interned once");
    }

    #[test]
    fn compiled_filters_agree_with_filters() {
        let payloads = [
            Some(payload(&[
                ("lang", text("en")),
                ("year", Value::Number(2020.0)),
            ])),
            Some(payload(&[
                ("lang", text("es")),
                ("year", Value::Number(2024.0)),
            ])),
            Some(payload(&[
                ("lang", Value::Number(1.0)),
                ("draft", Value::Bool(true)),
            ])),
            Some(payload(&[("year", text("2024"))])),
            None,
        ];
        let mut columns = PayloadColumns::default();
        for (slot, p) in payloads.iter().enumerate() {
            if let Some(p) = p {
                columns.insert(slot as u32, p);
            }
        }

        let year_from = |gte: f64| Filter::Range {
            field: field("year"),
            range: Range {
                gte: Some(gte),
                ..Range::default()
            },
        };
        let filters = [
            Filter::Eq {
                field: field("lang"),
                value: text("en"),
            },
            Filter::Eq {
                field: field("lang"),
                value: text("fr"),
            },
            Filter::Eq {
                field: field("lang"),
                value: Value::Number(1.0),
            },
            Filter::Eq {
                field: field("nowhere"),
                value: text("en"),
            },
            Filter::In {
                field: field("lang"),
                values: vec![text("es"), text("fr"), Value::Number(1.0)],
            },
            Filter::In {
                field: field("lang"),
                values: vec![],
            },
            year_from(2021.0),
            Filter::Range {
                field: field("nowhere"),
                range: Range::default(),
            },
            Filter::Exists(field("draft")),
            Filter::Not(Box::new(Filter::Exists(field("year")))),
            Filter::And(vec![year_from(2000.0), Filter::Exists(field("lang"))]),
            Filter::Or(vec![
                Filter::Eq {
                    field: field("draft"),
                    value: Value::Bool(true),
                },
                Filter::Not(Box::new(year_from(2022.0))),
            ]),
            Filter::And(vec![]),
            Filter::Or(vec![]),
        ];
        for filter in &filters {
            let compiled = columns.compile(filter);
            for (slot, p) in payloads.iter().enumerate() {
                assert_eq!(
                    compiled.matches(slot as u32),
                    filter.matches(p.as_ref()),
                    "{filter:?} on slot {slot}"
                );
            }
            assert_eq!(
                compiled.matches(99),
                filter.matches(None),
                "{filter:?} past every column"
            );
        }
    }

    #[test]
    fn removed_payloads_stop_matching() {
        let mut columns = PayloadColumns::default();
        columns.insert(0, &payload(&[("lang", text("en"))]));
        let exists = Filter::Exists(field("lang"));
        assert!(columns.compile(&exists).matches(0));
        columns.remove(0);
        assert!(!columns.compile(&exists).matches(0));
    }
}
