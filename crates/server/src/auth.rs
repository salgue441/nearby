//! API-key authentication: `Authorization: Bearer <key>` on every request.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// The secret clients must present.
#[derive(Clone)]
pub struct ApiKey(Arc<str>);

impl ApiKey {
    /// Wraps a key, which must not be empty or contain whitespace (it
    /// travels in an HTTP header).
    ///
    /// # Errors
    ///
    /// If the key is empty or contains whitespace or control characters.
    pub fn new(key: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(!key.is_empty(), "the API key must not be empty");
        anyhow::ensure!(
            key.bytes().all(|b| b.is_ascii_graphic()),
            "the API key may only contain visible ASCII characters"
        );
        Ok(Self(key.into()))
    }

    /// Whether `headers` carry this key as a bearer token.
    fn accepts(&self, headers: &HeaderMap) -> bool {
        headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|token| constant_time_eq(token.as_bytes(), self.0.as_bytes()))
    }
}

/// Never prints the key itself.
impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(..)")
    }
}

/// Middleware rejecting requests without the right key with `401`.
pub async fn require_api_key(State(key): State<ApiKey>, request: Request, next: Next) -> Response {
    if key.accepts(request.headers()) {
        return next.run(request).await;
    }
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "missing or invalid API key" })),
    )
        .into_response();
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

/// Compares two byte strings in time that depends only on their lengths,
/// so response times don't reveal how much of a guessed key was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(authorization: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, authorization.parse().unwrap());
        headers
    }

    #[test]
    fn only_the_exact_bearer_token_is_accepted() {
        let key = ApiKey::new("s3cret").unwrap();
        assert!(key.accepts(&headers("Bearer s3cret")));
        for wrong in [
            "Bearer s3cre",
            "Bearer s3cret2",
            "bearer s3cret",
            "s3cret",
            "Basic s3cret",
        ] {
            assert!(!key.accepts(&headers(wrong)), "{wrong}");
        }
        assert!(!key.accepts(&HeaderMap::new()));
    }

    #[test]
    fn keys_must_be_header_safe() {
        assert!(ApiKey::new("").is_err());
        assert!(ApiKey::new("has space").is_err());
        assert!(ApiKey::new("line\nbreak").is_err());
        assert!(ApiKey::new("ok-Key_123").is_ok());
    }

    #[test]
    fn debug_output_hides_the_key() {
        let key = ApiKey::new("s3cret").unwrap();
        assert_eq!(format!("{key:?}"), "ApiKey(..)");
    }

    #[test]
    fn constant_time_eq_compares_whole_strings() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }
}
