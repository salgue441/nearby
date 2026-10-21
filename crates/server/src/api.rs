//! The HTTP + JSON API.
//!
//! | Method | Path | Body | Response |
//! |---|---|---|---|
//! | `GET` | `/health` | | `{"status": "ok"}` |
//! | `GET` | `/collections` | | `{"collections": ["docs", ...]}` |
//! | `PUT` | `/collections/{name}` | collection config | `201`, collection info |
//! | `GET` | `/collections/{name}` | | collection info |
//! | `DELETE` | `/collections/{name}` | | `204` |
//! | `PUT` | `/collections/{name}/points` | `{"points": [{"id": 1, "vector": [...], "payload": {...}}]}` | `{"upserted": n}` |
//! | `POST` | `/collections/{name}/points/delete` | `{"ids": [1, 2]}` | `{"deleted": n}` |
//! | `GET` | `/collections/{name}/points/{id}` | | `{"id": 1, "vector": [...], "payload": {...}}` |
//! | `POST` | `/collections/{name}/search` | `{"vector": [...], "k": 10, "filter": {...}, "with_payload": true}` | `{"results": [{"id": 1, "distance": 0.5, "payload": {...}}]}` |
//! | `POST` | `/collections/{name}/search/batch` | `{"searches": [search, ...]}` | `{"results": [[...], ...]}` |
//! | `GET` | `/metrics` | | request and collection metrics, Prometheus text format |
//!
//! Payloads are optional objects of booleans, numbers and strings; filters
//! are described on `nearby_engine::Filter`.
//!
//! Errors come back as `{"error": "..."}` with a 4xx or 5xx status. With an
//! API key configured, every endpoint but `/health` requires
//! `Authorization: Bearer <key>` and answers `401` without it.

use std::num::NonZeroUsize;
use std::sync::{Arc, PoisonError};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, FromRequest, Path, State};
use axum::http::{StatusCode, header};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use nearby_core::{Error, PointId};
use nearby_engine::{Collection, CollectionConfig, CollectionInfo, Database, Filter, Payload};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::{ApiKey, require_api_key};
use crate::metrics::{self, Metrics};

/// Largest `k` a search may ask for.
const MAX_K: usize = 10_000;

/// Most searches one batch request may hold.
const MAX_BATCH: usize = 1024;

/// Shared state of every handler.
type Db = Arc<Database>;

/// The API's routes, serving `db`.
pub fn router(db: Db, options: ApiOptions) -> Router {
    let mut api = Router::new()
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
        .route("/collections/{name}/search/batch", post(search_batch));
    let metrics = Arc::new(Metrics::default());
    let exposition = Arc::clone(&metrics);
    api = api.route(
        "/metrics",
        get(move |State(db): State<Db>| async move {
            (
                [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
                exposition.render(&db),
            )
        }),
    );
    if let Some(key) = options.api_key {
        api = api.route_layer(middleware::from_fn_with_state(key, require_api_key));
    }
    // Health checks stay open, so load balancers need no key.
    api.route("/health", get(health))
        // A route layer sees the matched route pattern, which `track` labels by.
        .route_layer(middleware::from_fn_with_state(metrics, metrics::track))
        .layer(DefaultBodyLimit::max(options.max_body_bytes))
        .with_state(db)
}

/// How the API is served.
#[derive(Debug, Clone)]
pub struct ApiOptions {
    /// Largest request body accepted, in bytes.
    pub max_body_bytes: usize,
    /// When set, every endpoint but `/health` requires this key as a bearer token.
    pub api_key: Option<ApiKey>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload: Option<Payload>,
}

async fn upsert_points(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<UpsertRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let upserted = request.points.len();
    blocking(move || {
        let handle = db.collection(&name)?;
        let points: Vec<(PointId, &[f32], Option<&Payload>)> = request
            .points
            .iter()
            .map(|p| (p.id, p.vector.as_slice(), p.payload.as_ref()))
            .collect();
        let mut collection = handle.write().unwrap_or_else(PoisonError::into_inner);
        collection.upsert_with_payloads(&points)
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
        payload: collection.payload(id),
    }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    /// Only points whose payload matches are returned.
    #[serde(default)]
    filter: Option<Filter>,
    /// Whether each result carries its point's payload.
    #[serde(default)]
    with_payload: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<Payload>,
}

async fn search(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if request.k > MAX_K {
        return Err(ApiError::bad_request(format!("k must be at most {MAX_K}")));
    }
    let results = blocking(move || {
        let handle = db.collection(&name)?;
        let collection = handle.read().unwrap_or_else(PoisonError::into_inner);
        run_search(&collection, &request)
    })
    .await?;
    Ok(Json(SearchResponse { results }))
}

/// Runs one search request against `collection`.
fn run_search(collection: &Collection, request: &SearchRequest) -> nearby_core::Result<Vec<Hit>> {
    let hits = collection.search_where(&request.vector, request.k, request.filter.as_ref())?;
    Ok(hits
        .into_iter()
        .map(|p| Hit {
            id: p.id,
            distance: p.distance,
            payload: request
                .with_payload
                .then(|| collection.payload(p.id))
                .flatten(),
        })
        .collect())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchSearchRequest {
    searches: Vec<SearchRequest>,
}

#[derive(Debug, Serialize)]
struct BatchSearchResponse {
    /// One list of hits per search, in request order.
    results: Vec<Vec<Hit>>,
}

/// Runs many searches in one request, in parallel, under one read lock.
async fn search_batch(
    State(db): State<Db>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<BatchSearchRequest>,
) -> Result<Json<BatchSearchResponse>, ApiError> {
    let searches = request.searches;
    if searches.len() > MAX_BATCH {
        return Err(ApiError::bad_request(format!(
            "a batch holds at most {MAX_BATCH} searches, got {}",
            searches.len()
        )));
    }
    if let Some(i) = searches.iter().position(|s| s.k > MAX_K) {
        return Err(ApiError::bad_request(format!(
            "search {i}: k must be at most {MAX_K}"
        )));
    }
    let results = blocking(move || {
        let handle = db.collection(&name)?;
        let collection = handle.read().unwrap_or_else(PoisonError::into_inner);
        let threads = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let chunk = searches.len().div_ceil(threads).max(1);
        let collection = &*collection;
        let per_chunk: Vec<Vec<nearby_core::Result<Vec<Hit>>>> = std::thread::scope(|scope| {
            let workers: Vec<_> = searches
                .chunks(chunk)
                .map(|chunk| {
                    scope.spawn(move || chunk.iter().map(|s| run_search(collection, s)).collect())
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("search threads don't panic"))
                .collect()
        });
        Ok(per_chunk.into_iter().flatten().collect::<Vec<_>>())
    })
    .await?;
    let results = results
        .into_iter()
        .enumerate()
        .map(|(i, hits)| hits.map_err(|err| ApiError::from(err).context(&format!("search {i}"))))
        .collect::<Result<_, _>>()?;
    Ok(Json(BatchSearchResponse { results }))
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

    /// The same error, its message prefixed with what it concerns.
    fn context(self, context: &str) -> Self {
        Self {
            message: format!("{context}: {}", self.message),
            ..self
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
                app: router(
                    Arc::new(db),
                    ApiOptions {
                        max_body_bytes: 1 << 20,
                        api_key: None,
                    },
                ),
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
    async fn an_api_key_guards_everything_but_health() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        let app = router(
            Arc::new(db),
            ApiOptions {
                max_body_bytes: 1 << 20,
                api_key: Some(ApiKey::new("s3cret").unwrap()),
            },
        );
        let status = |uri: &str, key: Option<&str>| {
            let mut request = Request::builder().uri(uri);
            if let Some(key) = key {
                request = request.header("authorization", format!("Bearer {key}"));
            }
            let request = request.body(Body::empty()).unwrap();
            let app = app.clone();
            async move { app.oneshot(request).await.unwrap().status() }
        };
        assert_eq!(status("/health", None).await, StatusCode::OK);
        assert_eq!(status("/collections", None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(
            status("/collections", Some("wrong")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(status("/collections", Some("s3cret")).await, StatusCode::OK);
        assert_eq!(status("/metrics", None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(status("/metrics", Some("s3cret")).await, StatusCode::OK);
        assert_eq!(
            status("/collections/missing", None).await,
            StatusCode::UNAUTHORIZED,
            "no information leaks without the key"
        );
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
    async fn payloads_are_returned_and_filtered_on() {
        let server = TestServer::new();
        server.create_docs().await;
        let points = json!({ "points": [
            { "id": 1, "vector": [0.0, 0.0], "payload": { "lang": "en", "year": 2020 } },
            { "id": 2, "vector": [1.0, 0.0], "payload": { "lang": "es", "year": 2023 } },
            { "id": 3, "vector": [2.0, 0.0] },
        ]});
        server
            .send(Method::PUT, "/collections/docs/points", Some(points))
            .await;

        let (_, body) = server
            .send(Method::GET, "/collections/docs/points/2", None)
            .await;
        assert_eq!(
            body,
            json!({ "id": 2, "vector": [1.0, 0.0], "payload": { "lang": "es", "year": 2023 } })
        );
        let (_, body) = server
            .send(Method::GET, "/collections/docs/points/3", None)
            .await;
        assert_eq!(
            body,
            json!({ "id": 3, "vector": [2.0, 0.0] }),
            "no payload key"
        );

        let query = json!({
            "vector": [0.0, 0.0],
            "k": 3,
            "filter": { "or": [
                { "field": "lang", "eq": "es" },
                { "field": "year", "lt": 2021 },
            ]},
            "with_payload": true,
        });
        let (status, body) = server
            .send(Method::POST, "/collections/docs/search", Some(query))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({ "results": [
                { "id": 1, "distance": 0.0, "payload": { "lang": "en", "year": 2020 } },
                { "id": 2, "distance": 1.0, "payload": { "lang": "es", "year": 2023 } },
            ]})
        );

        let bad = json!({ "vector": [0.0, 0.0], "filter": { "field": "lang" } });
        let (status, body) = server
            .send(Method::POST, "/collections/docs/search", Some(bad))
            .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(
            body["error"].as_str().unwrap().contains("exactly one"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn batch_searches_answer_like_single_ones_in_order() {
        let server = TestServer::new();
        server.create_docs().await;
        let points: Vec<Value> = (0..50)
            .map(|i| json!({ "id": i, "vector": [i, 0], "payload": { "even": i % 2 == 0 } }))
            .collect();
        server
            .send(
                Method::PUT,
                "/collections/docs/points",
                Some(json!({ "points": points })),
            )
            .await;
        let searches: Vec<Value> = (0..20)
            .map(|i| {
                let mut search = json!({ "vector": [i * 2, 0], "k": 3 });
                if i % 3 == 0 {
                    search["filter"] = json!({ "field": "even", "eq": false });
                }
                search
            })
            .collect();
        let (status, batch) = server
            .send(
                Method::POST,
                "/collections/docs/search/batch",
                Some(json!({ "searches": searches })),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        for (i, search) in searches.into_iter().enumerate() {
            let (_, single) = server
                .send(Method::POST, "/collections/docs/search", Some(search))
                .await;
            assert_eq!(batch["results"][i], single["results"], "search {i}");
        }

        let bad = json!({ "searches": [{ "vector": [0, 0] }, { "vector": [0] }] });
        let (status, body) = server
            .send(Method::POST, "/collections/docs/search/batch", Some(bad))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"].as_str().unwrap().starts_with("search 1:"),
            "{body}"
        );

        let many: Vec<Value> = (0..=MAX_BATCH)
            .map(|_| json!({ "vector": [0, 0] }))
            .collect();
        let (status, _) = server
            .send(
                Method::POST,
                "/collections/docs/search/batch",
                Some(json!({ "searches": many })),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn metrics_count_requests_by_route_and_report_collections() {
        let server = TestServer::new();
        server.create_docs().await;
        let points =
            json!({ "points": [{ "id": 1, "vector": [0, 0] }, { "id": 2, "vector": [1, 1] }] });
        server
            .send(Method::PUT, "/collections/docs/points", Some(points))
            .await;
        for _ in 0..3 {
            server
                .send(Method::GET, "/collections/docs/points/1", None)
                .await;
        }
        server
            .send(Method::GET, "/collections/docs/points/9", None)
            .await;

        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let response = server.app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/plain")
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let route = r#"method="GET",route="/collections/{name}/points/{id}""#;
        for line in [
            format!(r#"nearby_requests_total{{{route},status="200"}} 3"#),
            format!(r#"nearby_requests_total{{{route},status="404"}} 1"#),
            r#"nearby_collection_points{collection="docs"} 2"#.to_owned(),
        ] {
            assert!(text.lines().any(|l| l == line), "missing {line}\n{text}");
        }
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
