//! Request and collection metrics, served in the Prometheus text format.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::Response;
use nearby_engine::{CollectionInfo, Database};

/// Upper bounds of the latency histogram's buckets, in seconds.
const BUCKETS: [f64; 12] = [
    0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
];

/// Counts requests and their latencies, by route.
#[derive(Debug, Default)]
pub struct Metrics {
    inner: Mutex<Registry>,
}

#[derive(Debug, Default)]
struct Registry {
    /// Requests by (method, route, status).
    requests: BTreeMap<(String, String, u16), u64>,
    /// Latencies by (method, route).
    latencies: BTreeMap<(String, String), Histogram>,
}

#[derive(Debug, Default)]
struct Histogram {
    /// `buckets[i]` counts observations at most `BUCKETS[i]`.
    buckets: [u64; BUCKETS.len()],
    count: u64,
    sum: f64,
}

impl Histogram {
    fn observe(&mut self, seconds: f64) {
        for (bucket, &bound) in self.buckets.iter_mut().zip(&BUCKETS) {
            if seconds <= bound {
                *bucket += 1;
            }
        }
        self.count += 1;
        self.sum += seconds;
    }
}

impl Metrics {
    /// Records one finished request.
    pub fn record(&self, method: &str, route: &str, status: u16, seconds: f64) {
        let mut registry = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        *registry
            .requests
            .entry((method.to_owned(), route.to_owned(), status))
            .or_default() += 1;
        registry
            .latencies
            .entry((method.to_owned(), route.to_owned()))
            .or_default()
            .observe(seconds);
    }

    /// Every metric in the Prometheus text exposition format, with each
    /// collection's size read from `db` now.
    pub fn render(&self, db: &Database) -> String {
        let mut out = String::new();
        {
            let registry = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            out.push_str("# HELP nearby_requests_total HTTP requests answered.\n");
            out.push_str("# TYPE nearby_requests_total counter\n");
            for ((method, route, status), count) in &registry.requests {
                let _ = writeln!(
                    out,
                    "nearby_requests_total{{method=\"{method}\",route=\"{route}\",status=\"{status}\"}} {count}"
                );
            }
            out.push_str("# HELP nearby_request_duration_seconds Time to answer HTTP requests.\n");
            out.push_str("# TYPE nearby_request_duration_seconds histogram\n");
            for ((method, route), h) in &registry.latencies {
                let labels = format!("method=\"{method}\",route=\"{route}\"");
                for (bound, count) in BUCKETS.iter().zip(&h.buckets) {
                    let _ = writeln!(
                        out,
                        "nearby_request_duration_seconds_bucket{{{labels},le=\"{bound}\"}} {count}"
                    );
                }
                let _ = writeln!(
                    out,
                    "nearby_request_duration_seconds_bucket{{{labels},le=\"+Inf\"}} {}",
                    h.count
                );
                let _ = writeln!(
                    out,
                    "nearby_request_duration_seconds_sum{{{labels}}} {}",
                    h.sum
                );
                let _ = writeln!(
                    out,
                    "nearby_request_duration_seconds_count{{{labels}}} {}",
                    h.count
                );
            }
        }
        let infos: Vec<_> = db
            .list()
            .iter()
            .filter_map(|name| {
                let handle = db.collection(name).ok()?;
                let collection = handle.read().unwrap_or_else(PoisonError::into_inner);
                Some(collection.info())
            })
            .collect();
        gauge(
            &mut out,
            "nearby_collection_points",
            "Points in a collection.",
            &infos,
            |i| i.points,
        );
        gauge(
            &mut out,
            "nearby_collection_segments",
            "Segments on disk for a collection.",
            &infos,
            |i| i.segments,
        );
        gauge(
            &mut out,
            "nearby_collection_wal_ops",
            "Writes in a collection's log since its last flush.",
            &infos,
            |i| i.wal_ops,
        );
        out
    }
}

/// Writes one gauge, with a sample per collection.
fn gauge(
    out: &mut String,
    name: &str,
    help: &str,
    infos: &[CollectionInfo],
    value: impl Fn(&CollectionInfo) -> usize,
) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    for info in infos {
        let _ = writeln!(
            out,
            "{name}{{collection=\"{}\"}} {}",
            info.name,
            value(info)
        );
    }
}

/// Middleware recording each request's route, status and latency. Routes
/// are the patterns (`/collections/{name}`), not the paths, so a metric
/// exists per endpoint rather than per collection or point.
pub async fn track(State(metrics): State<Arc<Metrics>>, request: Request, next: Next) -> Response {
    let method = request.method().as_str().to_owned();
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_owned(), |p| p.as_str().to_owned());
    let start = Instant::now();
    let response = next.run(request).await;
    metrics.record(
        &method,
        &route,
        response.status().as_u16(),
        start.elapsed().as_secs_f64(),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_are_cumulative() {
        let metrics = Metrics::default();
        for seconds in [0.000_1, 0.003, 0.003, 5.0] {
            metrics.record("POST", "/collections/{name}/search", 200, seconds);
        }
        metrics.record("POST", "/collections/{name}/search", 400, 0.000_1);
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), nearby_engine::EngineOptions::default()).unwrap();
        let text = metrics.render(&db);
        let labels = r#"method="POST",route="/collections/{name}/search""#;
        for line in [
            format!(r#"nearby_requests_total{{{labels},status="200"}} 4"#),
            format!(r#"nearby_requests_total{{{labels},status="400"}} 1"#),
            format!(r#"nearby_request_duration_seconds_bucket{{{labels},le="0.0005"}} 2"#),
            format!(r#"nearby_request_duration_seconds_bucket{{{labels},le="0.005"}} 4"#),
            format!(r#"nearby_request_duration_seconds_bucket{{{labels},le="2.5"}} 4"#),
            format!(r#"nearby_request_duration_seconds_bucket{{{labels},le="+Inf"}} 5"#),
            format!("nearby_request_duration_seconds_count{{{labels}}} 5"),
        ] {
            assert!(text.lines().any(|l| l == line), "missing {line}\n{text}");
        }
    }
}
