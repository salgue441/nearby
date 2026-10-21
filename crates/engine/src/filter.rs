//! Filters: conditions on payloads that limit which points a search returns.

use serde::{Deserialize, Serialize};

use crate::payload::{Payload, Value};

/// A condition on a point's payload. In JSON:
///
/// | Filter | Matches points whose payload |
/// |---|---|
/// | `{"field": "lang", "eq": "en"}` | has `lang` equal to `"en"` |
/// | `{"field": "lang", "in": ["en", "es"]}` | has `lang` equal to any of them |
/// | `{"field": "year", "gte": 2020, "lt": 2024}` | has a number `year` within every bound given (`gt`, `gte`, `lt`, `lte`) |
/// | `{"field": "lang", "exists": true}` | has a `lang` field (`false`: hasn't) |
/// | `{"and": [f, ...]}` | matches every filter (an empty list matches all) |
/// | `{"or": [f, ...]}` | matches any filter (an empty list matches none) |
/// | `{"not": f}` | doesn't match `f` |
///
/// A point without a payload has no fields, so it only matches filters
/// like `not` and `exists: false`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawFilter", into = "RawFilter")]
pub enum Filter {
    /// The field equals the value.
    Eq {
        /// The field's name.
        field: String,
        /// The value it must equal.
        value: Value,
    },
    /// The field equals one of the values.
    In {
        /// The field's name.
        field: String,
        /// The values it may equal.
        values: Vec<Value>,
    },
    /// The field is a number within the range.
    Range {
        /// The field's name.
        field: String,
        /// The bounds it must satisfy.
        range: Range,
    },
    /// The field is present, with any value.
    Exists(String),
    /// Every filter matches.
    And(Vec<Filter>),
    /// At least one filter matches.
    Or(Vec<Filter>),
    /// The filter doesn't match.
    Not(Box<Filter>),
}

/// Bounds on a number; absent bounds don't constrain it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Range {
    /// Exclusive lower bound.
    pub gt: Option<f64>,
    /// Inclusive lower bound.
    pub gte: Option<f64>,
    /// Exclusive upper bound.
    pub lt: Option<f64>,
    /// Inclusive upper bound.
    pub lte: Option<f64>,
}

impl Range {
    pub(crate) fn contains(&self, x: f64) -> bool {
        self.gt.is_none_or(|b| x > b)
            && self.gte.is_none_or(|b| x >= b)
            && self.lt.is_none_or(|b| x < b)
            && self.lte.is_none_or(|b| x <= b)
    }

    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl Filter {
    /// Whether a point with `payload` (`None`: no payload) matches.
    pub fn matches(&self, payload: Option<&Payload>) -> bool {
        let field = |name: &str| payload.and_then(|p| p.get(name));
        match self {
            Self::Eq { field: name, value } => field(name) == Some(value),
            Self::In {
                field: name,
                values,
            } => field(name).is_some_and(|v| values.contains(v)),
            Self::Range { field: name, range } => {
                matches!(field(name), Some(Value::Number(x)) if range.contains(*x))
            }
            Self::Exists(name) => field(name).is_some(),
            Self::And(filters) => filters.iter().all(|f| f.matches(payload)),
            Self::Or(filters) => filters.iter().any(|f| f.matches(payload)),
            Self::Not(filter) => !filter.matches(payload),
        }
    }
}

/// The JSON shape of a [`Filter`]: every key optional, so one struct can
/// read any form; [`Filter`]'s `TryFrom` checks exactly one form is used.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eq: Option<Value>,
    #[serde(rename = "in", skip_serializing_if = "Option::is_none")]
    any_of: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gte: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lte: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exists: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    and: Option<Vec<Filter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    or: Option<Vec<Filter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    not: Option<Box<Filter>>,
}

impl TryFrom<RawFilter> for Filter {
    type Error = String;

    fn try_from(raw: RawFilter) -> Result<Self, String> {
        let range = Range {
            gt: raw.gt,
            gte: raw.gte,
            lt: raw.lt,
            lte: raw.lte,
        };
        let conditions = usize::from(raw.eq.is_some())
            + usize::from(raw.any_of.is_some())
            + usize::from(!range.is_empty())
            + usize::from(raw.exists.is_some());
        let combinators = usize::from(raw.and.is_some())
            + usize::from(raw.or.is_some())
            + usize::from(raw.not.is_some());
        match (raw.field, conditions, combinators) {
            (Some(field), 1, 0) => Ok(if let Some(value) = raw.eq {
                Self::Eq { field, value }
            } else if let Some(values) = raw.any_of {
                Self::In { field, values }
            } else if let Some(exists) = raw.exists {
                let filter = Self::Exists(field);
                if exists {
                    filter
                } else {
                    Self::Not(Box::new(filter))
                }
            } else {
                Self::Range { field, range }
            }),
            (Some(_), _, 0) => Err(
                "a field filter needs exactly one of `eq`, `in`, `exists`, or range bounds"
                    .to_owned(),
            ),
            (None, 0, 1) => Ok(if let Some(filters) = raw.and {
                Self::And(filters)
            } else if let Some(filters) = raw.or {
                Self::Or(filters)
            } else {
                Self::Not(raw.not.expect("one combinator is set"))
            }),
            _ => Err(
                "a filter is either `field` with one condition, or exactly one of `and`, `or`, `not`"
                    .to_owned(),
            ),
        }
    }
}

impl From<Filter> for RawFilter {
    fn from(filter: Filter) -> Self {
        match filter {
            Filter::Eq { field, value } => Self {
                field: Some(field),
                eq: Some(value),
                ..Self::default()
            },
            Filter::In { field, values } => Self {
                field: Some(field),
                any_of: Some(values),
                ..Self::default()
            },
            Filter::Range { field, range } => Self {
                field: Some(field),
                gt: range.gt,
                gte: range.gte,
                lt: range.lt,
                lte: range.lte,
                ..Self::default()
            },
            Filter::Exists(field) => Self {
                field: Some(field),
                exists: Some(true),
                ..Self::default()
            },
            Filter::And(filters) => Self {
                and: Some(filters),
                ..Self::default()
            },
            Filter::Or(filters) => Self {
                or: Some(filters),
                ..Self::default()
            },
            Filter::Not(filter) => Self {
                not: Some(filter),
                ..Self::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(json: &str) -> Filter {
        serde_json::from_str(json).unwrap()
    }

    fn payload() -> Payload {
        serde_json::from_str(r#"{"lang": "en", "year": 2021, "draft": false}"#).unwrap()
    }

    #[test]
    fn conditions_match_fields() {
        let p = payload();
        let cases = [
            (r#"{"field": "lang", "eq": "en"}"#, true),
            (r#"{"field": "lang", "eq": "es"}"#, false),
            (r#"{"field": "year", "eq": 2021}"#, true),
            (r#"{"field": "draft", "eq": false}"#, true),
            (r#"{"field": "lang", "in": ["es", "en"]}"#, true),
            (r#"{"field": "lang", "in": []}"#, false),
            (r#"{"field": "year", "gte": 2021, "lt": 2022}"#, true),
            (r#"{"field": "year", "gt": 2021}"#, false),
            (r#"{"field": "year", "lte": 2021}"#, true),
            (r#"{"field": "lang", "gte": 0}"#, false),
            (r#"{"field": "missing", "eq": 1}"#, false),
            (r#"{"field": "lang", "exists": true}"#, true),
            (r#"{"field": "missing", "exists": false}"#, true),
        ];
        for (json, expected) in cases {
            assert_eq!(filter(json).matches(Some(&p)), expected, "{json}");
        }
    }

    #[test]
    fn combinators_compose() {
        let p = payload();
        let cases = [
            (
                r#"{"and": [{"field": "lang", "eq": "en"}, {"field": "year", "gt": 2000}]}"#,
                true,
            ),
            (
                r#"{"and": [{"field": "lang", "eq": "en"}, {"field": "year", "gt": 2030}]}"#,
                false,
            ),
            (
                r#"{"or": [{"field": "lang", "eq": "fr"}, {"field": "draft", "eq": false}]}"#,
                true,
            ),
            (r#"{"not": {"field": "lang", "eq": "en"}}"#, false),
            (r#"{"and": []}"#, true),
            (r#"{"or": []}"#, false),
        ];
        for (json, expected) in cases {
            assert_eq!(filter(json).matches(Some(&p)), expected, "{json}");
        }
    }

    #[test]
    fn points_without_a_payload_have_no_fields() {
        assert!(!filter(r#"{"field": "lang", "eq": "en"}"#).matches(None));
        assert!(filter(r#"{"not": {"field": "lang", "eq": "en"}}"#).matches(None));
        assert!(filter(r#"{"field": "lang", "exists": false}"#).matches(None));
    }

    #[test]
    fn malformed_filters_are_rejected() {
        for json in [
            "{}",
            r#"{"field": "a"}"#,
            r#"{"field": "a", "eq": 1, "in": [1]}"#,
            r#"{"eq": 1}"#,
            r#"{"and": [], "or": []}"#,
            r#"{"field": "a", "eq": 1, "and": []}"#,
            r#"{"field": "a", "like": "x"}"#,
            r#"{"field": "a", "eq": [1]}"#,
        ] {
            assert!(serde_json::from_str::<Filter>(json).is_err(), "{json}");
        }
    }

    #[test]
    fn filters_serialize_back_to_the_same_json() {
        for json in [
            r#"{"field":"lang","eq":"en"}"#,
            r#"{"field":"lang","in":["en","es"]}"#,
            r#"{"field":"year","gte":2020,"lt":2024}"#,
            r#"{"field":"lang","exists":true}"#,
            r#"{"and":[{"not":{"field":"a","eq":true}}]}"#,
        ] {
            let parsed: Filter = serde_json::from_str(json).unwrap();
            let back = serde_json::to_string(&parsed).unwrap();
            assert_eq!(
                serde_json::from_str::<Filter>(&back).unwrap(),
                parsed,
                "{json}"
            );
        }
    }
}
