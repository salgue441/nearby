//! Payloads: named values stored with a point and matched by filters.

use std::collections::BTreeMap;

use nearby_core::{Error, Result};
use serde::{Deserialize, Serialize, Serializer};

/// A point's payload: field names mapped to values, e.g.
/// `{"lang": "en", "year": 2024, "draft": false}`.
pub type Payload = BTreeMap<String, Value>;

/// Largest encoded payload accepted per point, in bytes.
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024;

/// One payload value. Serialized as the plain JSON value, with whole
/// numbers written as integers.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// `true` or `false`.
    Bool(bool),
    /// Any finite number. Integers are stored exactly up to 2^53.
    Number(f64),
    /// UTF-8 text.
    String(String),
}

/// Whole numbers below this magnitude are exact in an `f64`.
const EXACT_INTEGERS: f64 = 9_007_199_254_740_992.0;

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Bool(b) => serializer.serialize_bool(*b),
            Self::Number(n) if n.fract() == 0.0 && n.abs() < EXACT_INTEGERS => {
                serializer.serialize_i64(*n as i64)
            }
            Self::Number(n) => serializer.serialize_f64(*n),
            Self::String(s) => serializer.serialize_str(s),
        }
    }
}

const TAG_BOOL: u8 = 0;
const TAG_NUMBER: u8 = 1;
const TAG_STRING: u8 = 2;

/// Encodes `payload` for storage, rejecting values storage can't keep.
///
/// ```text
/// field count u32, then for each field in name order:
/// name length u32, name bytes, tag u8 (0 bool, 1 number, 2 string), and
/// a bool u8, a number f64, or a string length u32 and its bytes
/// ```
///
/// # Errors
///
/// [`Error::InvalidParameter`] if a number isn't finite or the encoding
/// exceeds [`MAX_PAYLOAD_BYTES`].
pub(crate) fn encode(payload: &Payload) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    put_len(&mut out, payload.len());
    for (name, value) in payload {
        put_len(&mut out, name.len());
        out.extend_from_slice(name.as_bytes());
        match value {
            Value::Bool(b) => {
                out.push(TAG_BOOL);
                out.push(u8::from(*b));
            }
            Value::Number(n) => {
                if !n.is_finite() {
                    return Err(Error::InvalidParameter(format!(
                        "payload field `{name}` is not a finite number"
                    )));
                }
                out.push(TAG_NUMBER);
                out.extend_from_slice(&n.to_le_bytes());
            }
            Value::String(s) => {
                out.push(TAG_STRING);
                put_len(&mut out, s.len());
                out.extend_from_slice(s.as_bytes());
            }
        }
    }
    if out.len() > MAX_PAYLOAD_BYTES {
        return Err(Error::InvalidParameter(format!(
            "payload is {} bytes encoded, more than the {MAX_PAYLOAD_BYTES} allowed",
            out.len()
        )));
    }
    Ok(out)
}

/// Decodes bytes written by [`encode`]. Empty bytes are no payload.
///
/// # Errors
///
/// [`Error::Corruption`] if the bytes aren't a valid encoding.
pub(crate) fn decode(bytes: &[u8]) -> Result<Option<Payload>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    let mut r = Bytes(bytes);
    let corrupt = || Error::corruption("payload", "invalid encoding");
    let count = r.len().ok_or_else(corrupt)?;
    let mut payload = Payload::new();
    for _ in 0..count {
        let name = r.string().ok_or_else(corrupt)?;
        let value = match r.take(1).ok_or_else(corrupt)?[0] {
            TAG_BOOL => Value::Bool(r.take(1).ok_or_else(corrupt)?[0] != 0),
            TAG_NUMBER => {
                let raw = r.take(8).ok_or_else(corrupt)?;
                Value::Number(f64::from_le_bytes(raw.try_into().expect("8 bytes")))
            }
            TAG_STRING => Value::String(r.string().ok_or_else(corrupt)?),
            _ => return Err(corrupt()),
        };
        payload.insert(name, value);
    }
    if !r.0.is_empty() {
        return Err(corrupt());
    }
    Ok(Some(payload))
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    let len = u32::try_from(len).expect("payload lengths are bounded");
    out.extend_from_slice(&len.to_le_bytes());
}

/// A cursor over encoded bytes; `None` means they ran out or were invalid.
struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(n)?;
        self.0 = tail;
        Some(head)
    }

    fn len(&mut self) -> Option<usize> {
        let raw = self.take(4)?;
        Some(u32::from_le_bytes(raw.try_into().ok()?) as usize)
    }

    fn string(&mut self) -> Option<String> {
        let len = self.len()?;
        String::from_utf8(self.take(len)?.to_vec()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Payload {
        Payload::from([
            ("lang".to_owned(), Value::String("en".to_owned())),
            ("year".to_owned(), Value::Number(2024.0)),
            ("draft".to_owned(), Value::Bool(false)),
            ("score".to_owned(), Value::Number(-0.25)),
        ])
    }

    #[test]
    fn payloads_round_trip() {
        let bytes = encode(&sample()).unwrap();
        assert_eq!(decode(&bytes).unwrap(), Some(sample()));
        assert_eq!(decode(&[]).unwrap(), None, "empty bytes: no payload");
        let empty = encode(&Payload::new()).unwrap();
        assert_eq!(decode(&empty).unwrap(), Some(Payload::new()));
    }

    #[test]
    fn the_same_payload_always_encodes_the_same() {
        let mut reversed = Payload::new();
        for (k, v) in sample().into_iter().rev() {
            reversed.insert(k, v);
        }
        assert_eq!(encode(&reversed).unwrap(), encode(&sample()).unwrap());
    }

    #[test]
    fn unstorable_payloads_are_rejected() {
        let nan = Payload::from([("x".to_owned(), Value::Number(f64::NAN))]);
        assert!(encode(&nan).unwrap_err().to_string().contains("finite"));
        let huge = Payload::from([("x".to_owned(), Value::String("a".repeat(70_000)))]);
        assert!(encode(&huge).unwrap_err().to_string().contains("allowed"));
    }

    #[test]
    fn damaged_encodings_are_corruption() {
        let bytes = encode(&sample()).unwrap();
        for bad in [
            &bytes[..bytes.len() - 1],
            &[1, 0, 0, 0, 9][..],
            &[0, 0, 0, 0, 7][..],
        ] {
            assert!(
                matches!(decode(bad), Err(Error::Corruption { .. })),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn values_read_as_plain_json() {
        let payload: Payload = serde_json::from_str(r#"{"a": 1, "b": "x", "c": true}"#).unwrap();
        assert_eq!(payload["a"], Value::Number(1.0));
        assert_eq!(payload["b"], Value::String("x".to_owned()));
        assert_eq!(payload["c"], Value::Bool(true));
        assert!(serde_json::from_str::<Payload>(r#"{"a": [1]}"#).is_err());
        assert_eq!(
            serde_json::to_string(&sample()).unwrap(),
            r#"{"draft":false,"lang":"en","score":-0.25,"year":2024}"#
        );
    }
}
