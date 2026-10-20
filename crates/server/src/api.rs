//! The HTTP + JSON API.
//!
//! | Method | Path | Body | Response |
//! |---|---|---|---|
//! | `GET` | `/health` | | `{"status": "ok"}` |
//! | `GET` | `/collections` | | `{"collections": ["docs", ...]}` |
//! | `PUT` | `/collections/{name}` | collection config | `201`, collection info |
//! | `GET` | `/collections/{name}` | | collection info |
//! | `DELETE` | `/collections/{name}` | | `204` |
//! | `PUT` | `/collections/{name}/points` | `{"points": [{"id": 1, "vector": [...]}]}` | `{"upserted": n}` |
//! | `POST` | `/collections/{name}/points/delete` | `{"ids": [1, 2]}` | `{"deleted": n}` |
//! | `GET` | `/collections/{name}/points/{id}` | | `{"id": 1, "vector": [...]}` |
//! | `POST` | `/collections/{name}/search` | `{"vector": [...], "k": 10}` | `{"results": [{"id": 1, "distance": 0.5}]}` |
//!
//! Errors come back as `{"error": "..."}` with a 4xx or 5xx status.

use std::sync::{Arc, PoisonError};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, FromRequest, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use nearby_core::{Error, PointId};
use nearby_engine::{CollectionConfig, CollectionInfo, Database};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Largest `k` a search may ask for.
const MAX_K: usize = 10_000;

/// Shared state of every handler.
type Db = Arc<Database>;

/// The API's routes, serving `db`, accepting request bodies of up to
/// `max_body_bytes`.
pub fn router(db: Db, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", get(list_collections))
        .route(
            "/collections/{name}",
            put(create_collection)
                .get(collection_info)
                .delete(drop_collection),
        )
        .route("/collections/{name}/points", put(upsert_points))
        .route("/collections/{name}/points/delete", post(delete_points))
        .route("/collections/{name}/points/{id}", get(get_point))
        .route("/collections/{name}/search", post(search))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(db)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

#[derive(Debug, Serialize)]
struct CollectionList {
    collections: Vec<String>,
}

async fn list_collections(State(db): State<Db>) -> Json<CollectionList> {
    Json(CollectionList {
        collections: db.list(),
    })
}

async fn create_collection(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(config): ApiJson<CollectionConfig>,
) -> Result<(StatusCode, Json<CollectionInfo>), ApiError> {
    let info = blocking(move || {
        let handle = db.create_collection(&name, config)?;
        let info = handle.read().unwrap_or_else(PoisonError::into_inner).info();
        Ok(info)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(info)))
}

async fn collection_info(
    State(db): State<Db>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, ApiError> {
    let handle = db.collection(&name)?;
    let info = handle.read().unwrap_or_else(PoisonError::into_inner).info();
    Ok(Json(info))
}

async fn drop_collection(
    State(db): State<Db>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    blocking(move || db.drop_collection(&name)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertRequest {
    points: Vec<PointBody>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PointBody {
    id: PointId,
    vector: Vec<f32>,
}

async fn upsert_points(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<UpsertRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let upserted = request.points.len();
    blocking(move || {
        let handle = db.collection(&name)?;
        let points: Vec<(PointId, &[f32])> = request
            .points
            .iter()
            .map(|p| (p.id, p.vector.as_slice()))
            .collect();
        let mut collection = handle.write().unwrap_or_else(PoisonError::into_inner);
        collection.upsert(&points)
    })
    .await?;
    Ok(Json(json!({ "upserted": upserted })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeleteRequest {
    ids: Vec<PointId>,
}

async fn delete_points(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<DeleteRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let deleted = blocking(move || {
        let handle = db.collection(&name)?;
        let mut collection = handle.write().unwrap_or_else(PoisonError::into_inner);
        collection.delete(&request.ids)
    })
    .await?;
    Ok(Json(json!({ "deleted": deleted })))
}

async fn get_point(
    State(db): State<Db>,
    Path((name, id)): Path<(String, PointId)>,
) -> Result<Json<PointBody>, ApiError> {
    let handle = db.collection(&name)?;
    let collection = handle.read().unwrap_or_else(PoisonError::into_inner);
    let vector = collection.get(id).ok_or(Error::PointNotFound(id))?;
    Ok(Json(PointBody {
        id,
        vector: vector.to_vec(),
    }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<Hit>,
}

#[derive(Debug, Serialize)]
struct Hit {
    id: PointId,
    distance: f32,
}

async fn search(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if request.k > MAX_K {
        return Err(ApiError::bad_request(format!("k must be at most {MAX_K}")));
    }
    let hits = blocking(move || {
        let handle = db.collection(&name)?;
        let collection = handle.read().unwrap_or_else(PoisonError::into_inner);
        collection.search(&request.vector, request.k)
    })
    .await?;
    Ok(Json(SearchResponse {
        results: hits
            .into_iter()
            .map(|p| Hit {
                id: p.id,
                distance: p.distance,
            })
            .collect(),
    }))
}

/// Runs engine work, which can take a while (an fsync, an index build), on
/// a thread meant for blocking, so it never stalls the async workers.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> nearby_core::Result<T> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| ApiError::internal(format!("request task failed: {e}")))?
        .map_err(ApiError::from)
}

/// [`Json`], but with malformed bodies reported in the API's error format.
#[derive(Debug, FromRequest)]
#[from_request(via(Json), rejection(ApiError))]
struct ApiJson<T>(T);

/// An error response: a status code and `{"error": message}`.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
        }
    }

    fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message,
        }
    }
}

impl From<Error> for ApiError {
    fn from(err: Error) -> Self {
        let status = match &err {
            Error::CollectionNotFound(_) | Error::PointNotFound(_) => StatusCode::NOT_FOUND,
            Error::CollectionExists(_) => StatusCode::CONFLICT,
            Error::DimensionMismatch { .. }
            | Error::InvalidVector
            | Error::InvalidParameter(_)
            | Error::Config(_) => StatusCode::BAD_REQUEST,
            Error::Corruption { .. } | Error::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        if status.is_server_error() {
            tracing::error!(%err, "request failed");
        }
        Self {
            status,
            message: err.to_string(),
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self {
            status: rejection.status(),
            message: rejection.body_text(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Method, Request};
    use http_body_util::BodyExt;
    use nearby_engine::{Durability, EngineOptions};
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    /// A router over a database in a temporary directory.
    struct TestServer {
        app: Router,
        /// Deleted when the server is dropped.
        dir: tempfile::TempDir,
    }

    fn options() -> EngineOptions {
        EngineOptions {
            durability: Durability::NoSync,
            ..EngineOptions::default()
        }
    }

    impl TestServer {
        fn new() -> Self {
            Self::in_dir(tempfile::tempdir().unwrap())
        }

        fn in_dir(dir: tempfile::TempDir) -> Self {
            let db = Database::open(dir.path(), options()).unwrap();
            Self {
                app: router(Arc::new(db), 1 << 20),
                dir,
            }
        }

        /// Sends a request and returns the status and the JSON body (`null` if empty).
        async fn send(
            &self,
            method: Method,
            uri: &str,
            body: Option<Value>,
        ) -> (StatusCode, Value) {
            let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
            self.send_raw(method, uri, body).await
        }

        async fn send_raw(&self, method: Method, uri: &str, body: Body) -> (StatusCode, Value) {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(body)
                .unwrap();
            let response = self.app.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let json = if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes).unwrap()
            };
            (status, json)
        }

        async fn create_docs(&self) {
            let config = json!({ "dim": 2, "metric": "l2", "index": { "type": "flat" } });
            let (status, _) = self
                .send(Method::PUT, "/collections/docs", Some(config))
                .await;
            assert_eq!(status, StatusCode::CREATED);
        }
    }

    #[tokio::test]
    async fn health_check() {
        let server = TestServer::new();
        let (status, body) = server.send(Method::GET, "/health", None).await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "status": "ok" })));
    }

    #[tokio::test]
    async fn collection_lifecycle() {
        let server = TestServer::new();
        let (status, info) = server
            .send(Method::PUT, "/collections/docs", Some(json!({ "dim": 3 })))
            .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(info["name"], "docs");
        assert_eq!(info["config"]["metric"], "l2");
        assert_eq!(info["config"]["index"]["type"], "hnsw");
        assert_eq!(info["points"], 0);

        let (_, list) = server.send(Method::GET, "/collections", None).await;
        assert_eq!(list, json!({ "collections": ["docs"] }));
        let (status, _) = server
            .send(Method::PUT, "/collections/docs", Some(json!({ "dim": 3 })))
            .await;
        assert_eq!(status, StatusCode::CONFLICT);

        let (status, body) = server.send(Method::DELETE, "/collections/docs", None).await;
        assert_eq!((status, body), (StatusCode::NO_CONTENT, Value::Null));
        let (status, body) = server.send(Method::GET, "/collections/docs", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"], "collection `docs` not found");
    }

    #[tokio::test]
    async fn points_can_be_written_read_searched_and_deleted() {
        let server = TestServer::new();
        server.create_docs().await;
        let points = json!({ "points": [
            { "id": 1, "vector": [0.0, 0.0] },
            { "id": 2, "vector": [1.0, 0.0] },
            { "id": 3, "vector": [5.0, 5.0] },
        ]});
        let (status, body) = server
            .send(Method::PUT, "/collections/docs/points", Some(points))
            .await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "upserted": 3 })));

        let (status, body) = server
            .send(Method::GET, "/collections/docs/points/2", None)
            .await;
        assert_eq!(
            (status, body),
            (StatusCode::OK, json!({ "id": 2, "vector": [1.0, 0.0] }))
        );

        let query = json!({ "vector": [1.0, 0.0], "k": 2 });
        let (status, body) = server
            .send(Method::POST, "/collections/docs/search", Some(query))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({ "results": [
                { "id": 2, "distance": 0.0 },
                { "id": 1, "distance": 1.0 },
            ]})
        );

        let (_, body) = server
            .send(
                Method::POST,
                "/collections/docs/points/delete",
                Some(json!({ "ids": [2, 99] })),
            )
            .await;
        assert_eq!(body, json!({ "deleted": 1 }));
        let (status, body) = server
            .send(Method::GET, "/collections/docs/points/2", None)
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"], "point `2` not found");
        let (_, info) = server.send(Method::GET, "/collections/docs", None).await;
        assert_eq!(info["points"], 2);
    }

    #[tokio::test]
    async fn bad_requests_get_clear_errors() {
        let server = TestServer::new();
        server.create_docs().await;
        let cases = [
            (
                Method::PUT,
                "/collections/docs/points",
                json!({ "points": [{ "id": 1, "vector": [1.0] }] }),
                StatusCode::BAD_REQUEST,
                "dimension mismatch",
            ),
            (
                Method::POST,
                "/collections/docs/search",
                json!({ "vector": [1.0, 2.0], "k": 1_000_000 }),
                StatusCode::BAD_REQUEST,
                "k must be at most",
            ),
            (
                Method::POST,
                "/collections/missing/search",
                json!({ "vector": [1.0, 2.0] }),
                StatusCode::NOT_FOUND,
                "not found",
            ),
            (
                Method::PUT,
                "/collections/bad%20name",
                json!({ "dim": 2 }),
                StatusCode::BAD_REQUEST,
                "collection names",
            ),
            (
                Method::PUT,
                "/collections/other",
                json!({ "dim": 2, "metric": "manhattan" }),
                StatusCode::UNPROCESSABLE_ENTITY,
                "manhattan",
            ),
            (
                Method::POST,
                "/collections/docs/search",
                json!({ "vector": [1.0, 2.0], "limit": 3 }),
                StatusCode::UNPROCESSABLE_ENTITY,
                "limit",
            ),
            (
                // Too large for f32, so it becomes infinite, which is rejected.
                Method::PUT,
                "/collections/docs/points",
                json!({ "points": [{ "id": 1, "vector": [1e300, 0.0] }] }),
                StatusCode::BAD_REQUEST,
                "NaN or infinite",
            ),
        ];
        for (method, uri, body, expected_status, expected_message) in cases {
            let (status, response) = server.send(method, uri, Some(body)).await;
            assert_eq!(status, expected_status, "{uri}: {response}");
            let message = response["error"].as_str().unwrap();
            assert!(message.contains(expected_message), "{uri}: {message}");
        }
    }

    #[tokio::test]
    async fn malformed_json_is_rejected_in_the_error_format() {
        let server = TestServer::new();
        server.create_docs().await;
        let (status, body) = server
            .send_raw(
                Method::POST,
                "/collections/docs/search",
                Body::from("{not json"),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].is_string(), "{body}");
    }

    #[tokio::test]
    async fn oversized_bodies_are_refused() {
        let server = TestServer::new();
        server.create_docs().await;
        let vector = vec![0.5; 400_000]; // about 1.6 MB of JSON, over the 1 MiB limit
        let (status, _) = server
            .send(
                Method::PUT,
                "/collections/docs/points",
                Some(json!({ "points": [{ "id": 1, "vector": vector }] })),
            )
            .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn data_survives_a_restart() {
        let server = TestServer::new();
        server.create_docs().await;
        let points = json!({ "points": [{ "id": 7, "vector": [3.0, 4.0] }] });
        server
            .send(Method::PUT, "/collections/docs/points", Some(points))
            .await;
        let TestServer { app, dir } = server;
        drop(app);

        let server = TestServer::in_dir(dir);
        let (status, body) = server
            .send(Method::GET, "/collections/docs/points/7", None)
            .await;
        assert_eq!(
            (status, body),
            (StatusCode::OK, json!({ "id": 7, "vector": [3.0, 4.0] }))
        );
    }
}
