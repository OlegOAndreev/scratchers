use anyhow::{bail, Context, Result};
use rand::Rng;
use rusqlite::{params, Connection, OpenFlags};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{env, thread, time};
use std::sync::atomic::{AtomicUsize, Ordering};
use time::Duration;

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        bail!("Usage: sqlite <action>");
    }
    let action = &args[1];

    match action.as_str() {
        "concurrent-readers" => {
            let mut conn = Connection::open("db-with-journal")?;
            conn.query_row("PRAGMA journal_mode = persist", (), |_| Ok(()))
                .context("PRAGMA journal_mode failed")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = full", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            delete_row_ids(&mut conn, "normal", &mut row_ids)?;
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            drop(conn);

            let row_ids_copy = Arc::new(row_ids.clone());
            let mut handles = vec![];
            for _ in 0..4 {
                let row_ids_ref = row_ids_copy.clone();
                let h = thread::spawn(move || -> Result<()> {
                    let mut conn = Connection::open_with_flags("db-with-journal", OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                    do_select(&mut conn, "normal", &row_ids_ref)?;
                    Ok(())
                });
                handles.push(h);
            }
            for h in handles {
                h.join().unwrap()?;
            }
        }
        "insert" => {
            let mut conn = Connection::open("db-with-journal")?;
            conn.query_row("PRAGMA journal_mode = persist", (), |_| Ok(()))
                .context("PRAGMA journal_mode failed")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = full", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            delete_row_ids(&mut conn, "normal", &mut row_ids)?;
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "insert-each-line" => {
            let mut conn = Connection::open("db-with-journal")?;
            conn.query_row("PRAGMA journal_mode = persist", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = full", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert_each_line(&mut conn, "normal", 20000, &mut row_ids)?;
            delete_row_ids_each_line(&mut conn, "normal", &mut row_ids)?;
            do_insert_each_line(&mut conn, "normal", 20000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "insert-each-line-normal" => {
            let mut conn = Connection::open("db-with-journal")?;
            conn.query_row("PRAGMA journal_mode = persist", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert_each_line(&mut conn, "normal", 20000, &mut row_ids)?;
            delete_row_ids_each_line(&mut conn, "normal", &mut row_ids)?;
            do_insert_each_line(&mut conn, "normal", 20000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "wal-insert" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode failed")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = full", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            delete_row_ids(&mut conn, "normal", &mut row_ids)?;
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "wal-insert-normal" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode failed")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            delete_row_ids(&mut conn, "normal", &mut row_ids)?;
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "wal-concurrent-readers" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            delete_row_ids(&mut conn, "normal", &mut row_ids)?;
            do_insert(&mut conn, "normal", 2000000, &mut row_ids)?;
            drop(conn);

            let row_ids_copy = Arc::new(row_ids.clone());
            let mut handles = vec![];
            for _ in 0..4 {
                let row_ids_ref = row_ids_copy.clone();
                let h = thread::spawn(move || -> Result<()> {
                    let mut conn = Connection::open_with_flags("db-with-wal", OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                    do_select(&mut conn, "normal", &row_ids_ref)?;
                    Ok(())
                });
                handles.push(h);
            }
            for h in handles {
                h.join().unwrap()?;
            }
        }
        "wal-insert-each-line" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = full", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert_each_line(&mut conn, "normal", 100000, &mut row_ids)?;
            delete_row_ids_each_line(&mut conn, "normal", &mut row_ids)?;
            do_insert_each_line(&mut conn, "normal", 100000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "wal-insert-each-line-normal" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert_each_line(&mut conn, "normal", 500000, &mut row_ids)?;
            delete_row_ids_each_line(&mut conn, "normal", &mut row_ids)?;
            do_insert_each_line(&mut conn, "normal", 500000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;
        }
        "wal-insert-each-line-concurrent-reader" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert_each_line(&mut conn, "normal", 500000, &mut row_ids)?;
            delete_row_ids_each_line(&mut conn, "normal", &mut row_ids)?;

            let row_ids_copy = Arc::new(row_ids.clone());
            let handle = thread::spawn(move || -> Result<()> {
                let mut conn = Connection::open_with_flags("db-with-wal", OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                do_select(&mut conn, "normal", &row_ids_copy)?;
                do_select(&mut conn, "normal", &row_ids_copy)?;
                do_select(&mut conn, "normal", &row_ids_copy)?;
                do_select(&mut conn, "normal", &row_ids_copy)?;

                Ok(())
            });
            do_insert_each_line(&mut conn, "normal", 500000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;

            handle.join().unwrap()?;
        }
        "wal-insert-each-line-concurrent-readers" => {
            let mut conn = Connection::open("db-with-wal")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;
            let mut row_ids = vec![];
            do_insert_each_line(&mut conn, "normal", 500000, &mut row_ids)?;
            delete_row_ids_each_line(&mut conn, "normal", &mut row_ids)?;

            let row_ids_copy = Arc::new(row_ids.clone());
            let mut handles = vec![];
            for _ in 0..4 {
                let row_ids_ref = row_ids_copy.clone();
                let h = thread::spawn(move || -> Result<()> {
                    let mut conn = Connection::open_with_flags("db-with-wal", OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                    do_select(&mut conn, "normal", &row_ids_ref)?;
                    Ok(())
                });
                handles.push(h);
            }
            do_insert_each_line(&mut conn, "normal", 500000, &mut row_ids)?;
            do_select(&mut conn, "normal", &row_ids)?;

            for h in handles {
                h.join().unwrap()?;
            }
        }
        "queue" => {
            let mut conn = Connection::open("db-queue")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;

            let rconn = Connection::open("db-queue")?;
            do_query_test(&mut conn, vec![rconn])?;
        }
        "queue-concurrent" => {
            let mut conn = Connection::open("db-queue")?;
            conn.query_row("PRAGMA journal_mode = wal", (), |_| Ok(()))
                .context("PRAGMA journal_mode")?;
            conn.execute("PRAGMA auto_vacuum = full", ())
                .context("PRAGMA auto_vacuum failed")?;
            conn.execute("PRAGMA synchronous = normal", ())
                .context("PRAGMA synchronous")?;

            let mut rconns = vec![];
            for _ in 0..4 {
                rconns.push(Connection::open("db-queue")?);
            }
            do_query_test(&mut conn, rconns)?;
        }
        _ => bail!("Unknown action {}", action),
    }

    Ok(())
}

fn do_query_test(conn: &mut Connection, rconns: Vec<Connection>) -> Result<()> {
    conn.execute("CREATE TABLE IF NOT EXISTS queue (col TEXT)", ())
        .context("CREATE TABLE do_query_test")?;

    const NUM_ELEMS: usize = 100000;
    let lines = generate_random_ascii_lines(NUM_ELEMS, 20, 200);

    let mut handles = vec![];
    let got = Arc::new(AtomicUsize::new(0));
    let writer_mutex = Arc::new(Mutex::new(()));
    for rconn in rconns {
        let got_ref = got.clone();
        let writer_mutex_ref = writer_mutex.clone();
        let h = thread::spawn(move || -> Result<()> {
            let rstart = Instant::now();
            let mut row_ids = vec![];
            let mut stmt = rconn
                .prepare("SELECT rowid, col FROM queue ORDER BY rowid LIMIT 1000")
                .context("PREPARE SELECT do_query_test")?;
            let mut delete_stmt = rconn
                .prepare("DELETE FROM queue WHERE rowid = ?1")
                .context("PREPARE DELETE FROM do_query_test")?;
            loop {
                let mut rows = stmt
                    .query(())
                    .context("EXEC SELECT do_query_test")?;
                while let Some(row) = rows
                    .next()
                    .context("EXEC SELECT next do_query_test")?
                {
                    row_ids.push(
                        row.get_ref(0)
                            .context("EXEC SELECT get_ref do_query_test")?
                            .as_i64()
                            .context("EXEC SELECT deref do_query_test")?,
                    );
                }
                let mut deleted = 0;
                for &row_id in &row_ids {
                    let _guard = writer_mutex_ref.lock();
                    deleted += delete_stmt
                        .execute(params![row_id])
                        .context("EXEC DELETE FROM do_query_test")?;
                }
                row_ids.clear();

                let prev = got_ref.fetch_add(deleted, Ordering::Relaxed);
                if prev + deleted >= NUM_ELEMS {
                    break
                }

                thread::sleep(Duration::from_millis(1));
            }

            let elapsed = rstart.elapsed();
            let per_elem = elapsed / NUM_ELEMS as u32;
            println!("Consumed {} elems in {:?} ({:?} per elem)", NUM_ELEMS, rstart.elapsed(), per_elem);
            Ok(())
        });
        handles.push(h);
    }


    let start = Instant::now();
    let mut stmt = conn
        .prepare("INSERT INTO queue VALUES (?1)")
        .context("PREPARE INSERT INTO do_query_test")?;
    for line in lines {
        let _guard = writer_mutex.lock();
        stmt.execute(params![line])
            .context("EXEC INSERT INTO do_query_test")?;
    }
    let elapsed = start.elapsed();
    let per_elem = elapsed / NUM_ELEMS as u32;
    println!("Produced {} elems in {:?} ({:?} per elem)", NUM_ELEMS, start.elapsed(), per_elem);

    for h in handles {
        h.join().unwrap()?;
    }

    Ok(())
}

// fn do_incremental_vacuum(conn: &mut Connection) -> Result<()> {
//     let start = Instant::now();
//     conn.execute("PRAGMA incremental_vacuum", ())
//         .context("PRAGMA incremental_vacuum")?;
//     println!("Incremental vacuum in {:?}", start.elapsed());
//     Ok(())
// }

fn do_insert(
    conn: &mut Connection,
    table: &str,
    num_lines: usize,
    row_ids: &mut Vec<i64>,
) -> Result<()> {
    let lines = generate_random_ascii_lines(num_lines, 20, 200);

    let start = Instant::now();
    conn.execute(&format!("CREATE TABLE IF NOT EXISTS {} (col TEXT)", table), ())
        .context("CREATE TABLE do_insert")?;
    let tx = conn.transaction()?;
    let mut stmt = tx
        .prepare(&format!("INSERT INTO {} (col) VALUES (?1)", table))
        .context("PREPARE INSERT INTO do_insert")?;
    for line in &lines {
        stmt.execute(params![line])
            .context("EXECUTE INSERT INTO do_insert")?;
        row_ids.push(tx.last_insert_rowid());
    }
    drop(stmt);
    tx.commit()
        .context("COMMIT do_insert")?;
    let elapsed = start.elapsed();
    let per_line = elapsed / num_lines as u32;
    print!("Inserted {} lines in {:?} ({:?} per line)\n", num_lines, elapsed, per_line);

    Ok(())
}

fn do_insert_each_line(
    conn: &mut Connection,
    table: &str,
    num_lines: usize,
    row_ids: &mut Vec<i64>,
) -> Result<()> {
    let lines = generate_random_ascii_lines(num_lines, 20, 200);

    let start = Instant::now();
    conn.execute(&format!("CREATE TABLE IF NOT EXISTS {} (col TEXT)", table), ())
        .context("CREATE TABLE do_insert_each_line")?;
    let mut stmt = conn
        .prepare(&format!("INSERT INTO {} (col) VALUES (?1)", table))
        .context("PREPARE INSERT INTO do_insert_each_line")?;
    for line in &lines {
        stmt.execute(params![line])
            .context("INSERT INTO do_insert_each_line")?;
        row_ids.push(conn.last_insert_rowid());
    }
    let elapsed = start.elapsed();
    let per_line = elapsed / num_lines as u32;
    print!("Inserted separately {} lines in {:?} ({:?} per line)\n", num_lines, elapsed, per_line);

    Ok(())
}

fn generate_random_ascii_lines(
    num_strings: usize,
    min_length: usize,
    max_length: usize,
) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let mut lines = vec![];

    for _ in 0..num_strings {
        let line_length = rng.gen_range(min_length..=max_length);
        let mut line = Vec::with_capacity(line_length);
        for _ in 0..line_length {
            line.push(rng.sample(rand::distributions::Alphanumeric));
        }
        lines.push(unsafe { String::from_utf8_unchecked(line) });
    }

    lines
}

fn delete_row_ids(conn: &mut Connection, table: &str, row_ids: &mut Vec<i64>) -> Result<()> {
    let mut rng = rand::thread_rng();
    let n = row_ids.len() / 2;

    let start = Instant::now();
    let tx = conn.transaction()?;
    let mut stmt = tx
        .prepare(&format!("DELETE FROM {} WHERE rowid = ?1", table))
        .context("PREPARE DELETE FROM delete_row_ids")?;
    for _ in 0..n {
        let idx = rng.gen_range(0..row_ids.len());
        let row_id = row_ids.swap_remove(idx);
        stmt.execute(params![row_id])
            .context("EXEC DELETE FROM delete_row_ids")?;
    }
    drop(stmt);
    tx.commit()
        .context("COMMIT delete_row_ids")?;
    let elapsed = start.elapsed();
    let per_line = elapsed / n as u32;
    print!("Removed {} lines in {:?} ({:?} per line)\n", n, elapsed, per_line);
    Ok(())
}

fn delete_row_ids_each_line(
    conn: &mut Connection,
    table: &str,
    row_ids: &mut Vec<i64>,
) -> Result<()> {
    let mut rng = rand::thread_rng();
    let n = row_ids.len() / 2;

    let start = Instant::now();
    let mut stmt = conn
        .prepare(&format!("DELETE FROM {} WHERE rowid = ?1", table))
        .context("PREPARE DELETE FROM delete_row_ids_each_line")?;
    for _ in 0..n {
        let idx = rng.gen_range(0..row_ids.len());
        let row_id = row_ids.swap_remove(idx);
        stmt.execute(params![row_id])
            .context("EXEC DELETE FROM delete_row_ids_each_line")?;
    }
    let elapsed = start.elapsed();
    let per_line = elapsed / n as u32;
    print!("Removed separately {} lines in {:?} ({:?} per line)\n", n, elapsed, per_line);
    Ok(())
}

fn do_select(conn: &mut Connection, table: &str, row_ids: &[i64]) -> Result<()> {
    let mut rng = rand::thread_rng();

    let start = Instant::now();
    let mut stmt = conn
        .prepare(&format!("SELECT * FROM {} WHERE rowid = ?1", table))
        .context("PREPARE SELECT FROM do_select")?;
    let n = row_ids.len();
    for _ in 0..n {
        let id = rng.gen_range(0..n);
        stmt.query_row(params![row_ids[id]], |_| Ok(()))
            .context("EXEC SELECT FROM do_select")?;
    }
    let elapsed = start.elapsed();
    let per_line = elapsed / n as u32;
    print!("Queried {} lines in {:?} ({:?} per line)\n", n, elapsed, per_line);
    Ok(())
}
