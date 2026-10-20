//! A small product catalog: payloads, filtered searches, updates and a
//! restart, through the engine's API.
//!
//! Each product's vector describes its style on four axes (sporty, formal,
//! warm, colorful), standing in for the embeddings a model would produce.
//! Its payload holds what the shop filters on: category, price and stock.
//!
//! ```sh
//! cargo run -p nearby-engine --example product_search
//! ```

use nearby_core::{Metric, PointId};
use nearby_engine::{
    CollectionConfig, Database, EngineOptions, Filter, IndexConfig, Payload, Range, Value,
};

/// A product in the catalog.
struct Product {
    id: PointId,
    name: &'static str,
    /// Sporty, formal, warm, colorful.
    style: [f32; 4],
    category: &'static str,
    price: f64,
    in_stock: bool,
}

const fn product(
    id: PointId,
    name: &'static str,
    style: [f32; 4],
    category: &'static str,
    price: f64,
    in_stock: bool,
) -> Product {
    Product {
        id,
        name,
        style,
        category,
        price,
        in_stock,
    }
}

const PRODUCTS: [Product; 8] = [
    product(
        1,
        "trail running shoes",
        [0.9, 0.0, 0.2, 0.6],
        "shoes",
        120.0,
        true,
    ),
    product(
        2,
        "leather oxfords",
        [0.0, 0.9, 0.1, 0.1],
        "shoes",
        180.0,
        true,
    ),
    product(
        3,
        "canvas sneakers",
        [0.5, 0.1, 0.1, 0.8],
        "shoes",
        60.0,
        false,
    ),
    product(4, "down parka", [0.3, 0.2, 1.0, 0.3], "coats", 250.0, true),
    product(
        5,
        "wool overcoat",
        [0.0, 0.8, 0.8, 0.1],
        "coats",
        320.0,
        true,
    ),
    product(6, "rain shell", [0.7, 0.1, 0.4, 0.5], "coats", 140.0, true),
    product(7, "track jacket", [0.9, 0.0, 0.3, 0.9], "coats", 80.0, true),
    product(
        8,
        "silk tie",
        [0.0, 1.0, 0.0, 0.5],
        "accessories",
        45.0,
        true,
    ),
];

fn main() -> nearby_core::Result<()> {
    let dir = tempfile::tempdir()?;
    let db = Database::open(dir.path(), EngineOptions::default())?;
    let config = CollectionConfig {
        dim: 4,
        metric: Metric::Cosine,
        index: IndexConfig::default(), // HNSW
    };
    let products = db.create_collection("products", config)?;

    let payloads: Vec<Payload> = PRODUCTS
        .iter()
        .map(|p| {
            Payload::from([
                ("name".to_owned(), Value::String(p.name.to_owned())),
                ("category".to_owned(), Value::String(p.category.to_owned())),
                ("price".to_owned(), Value::Number(p.price)),
                ("in_stock".to_owned(), Value::Bool(p.in_stock)),
            ])
        })
        .collect();
    let points: Vec<(PointId, &[f32], Option<&Payload>)> = PRODUCTS
        .iter()
        .zip(&payloads)
        .map(|(p, payload)| (p.id, p.style.as_slice(), Some(payload)))
        .collect();
    products.write().unwrap().upsert_with_payloads(&points)?;

    // Someone looking at the track jacket: what else is sporty and bright?
    let like_track_jacket = [0.9, 0.0, 0.3, 0.9];
    show("Similar to the track jacket", &db, &like_track_jacket, None)?;

    // Only shoes...
    let shoes = Filter::Eq {
        field: "category".to_owned(),
        value: Value::String("shoes".to_owned()),
    };
    show("Similar shoes", &db, &like_track_jacket, Some(&shoes))?;

    // ...that are in stock and under $150.
    let affordable_shoes_in_stock = Filter::And(vec![
        shoes,
        Filter::Eq {
            field: "in_stock".to_owned(),
            value: Value::Bool(true),
        },
        Filter::Range {
            field: "price".to_owned(),
            range: Range {
                lt: Some(150.0),
                ..Range::default()
            },
        },
    ]);
    show(
        "Similar shoes in stock under $150",
        &db,
        &like_track_jacket,
        Some(&affordable_shoes_in_stock),
    )?;

    // The canvas sneakers come back in stock; the oxfords are discontinued.
    // Writing a point replaces its payload too, so send the whole payload.
    let mut sneakers = payloads[2].clone();
    sneakers.insert("in_stock".to_owned(), Value::Bool(true));
    {
        let mut products = products.write().unwrap();
        products.upsert_with_payloads(&[(3, &PRODUCTS[2].style, Some(&sneakers))])?;
        products.delete(&[2])?;
    }
    show(
        "After restocking and a discontinuation",
        &db,
        &like_track_jacket,
        Some(&affordable_shoes_in_stock),
    )?;

    // Everything is durable: a restart (here, reopening the directory) keeps
    // vectors, payloads and deletes.
    drop(products);
    db.checkpoint_all()?;
    drop(db);
    let db = Database::open(dir.path(), EngineOptions::default())?;
    let formal = [0.0, 1.0, 0.3, 0.2];
    show("Formal wear, after a restart", &db, &formal, None)?;
    Ok(())
}

/// Prints the three products nearest to `style` that match `filter`.
fn show(
    title: &str,
    db: &Database,
    style: &[f32],
    filter: Option<&Filter>,
) -> nearby_core::Result<()> {
    let handle = db.collection("products")?;
    let products = handle.read().unwrap();
    println!("{title}:");
    for hit in products.search_where(style, 3, filter)? {
        let payload = products
            .payload(hit.id)
            .expect("every product has a payload");
        let field = |name: &str| match &payload[name] {
            Value::String(s) => s.clone(),
            Value::Number(n) => format!("${n}"),
            Value::Bool(b) => if *b { "in stock" } else { "sold out" }.to_owned(),
        };
        println!(
            "  {:<20} {:>5}  {:<9} distance {:.3}",
            field("name"),
            field("price"),
            field("in_stock"),
            hit.distance
        );
    }
    println!();
    Ok(())
}
