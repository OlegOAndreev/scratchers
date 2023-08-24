use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::{bail, Result};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        bail!("Usage: bits <data file> <algo>");
    }
    let filename = &args[1];
    let algo = &args[2];
    let file = File::open(filename)?;

    let start_load = Instant::now();
    let mut strings = vec![];
    let mut partitions = vec![];
    for line in BufReader::new(file).lines() {
        let line = line?;
        let parts: Vec<_> = line.split(" ").collect();
        if parts.len() != 2 {
            bail!("Data line must have two fields");
        }
        strings.push(parts[0].to_string());
        partitions.push(parts[1].parse::<u32>()?);
    }
    let load_time = Instant::now() - start_load;
    println!("Parsed data in {:?}", load_time);

    let start_compute = Instant::now();
    let mut our_partitions = vec![0; partitions.len()];
    match algo.as_str() {
        "array" => array_partition(&strings, &mut our_partitions),
        "bitmask" => bitmask_partition(&strings, &mut our_partitions),
        "array_noclear" => array_noclear_partition(&strings, &mut our_partitions),
        _ => bail!("Unknown algo {}", algo),
    }
    let compute_time = Instant::now() - start_compute;

    for i in 0..partitions.len() {
        if partitions[i] != our_partitions[i] {
            bail!(
                "Different partitions for {} (line {}): {} vs {}",
                strings[i],
                i,
                partitions[i],
                our_partitions[i]
            );
        }
    }

    println!("Computed using algo {} in {:?}", algo, compute_time);

    Ok(())
}

fn array_partition(strings: &[String], our_partitions: &mut [u32]) {
    let mut mask = [false; 256];
    for (s, out) in strings.iter().zip(our_partitions) {
        let mut ret = 1u32;
        mask.fill(false);
        for &b in s.as_bytes() {
            if mask[b as usize] {
                mask.fill(false);
                ret += 1;
            }
            mask[b as usize] = true
        }
        *out = ret
    }
}

fn bitmask_partition(strings: &[String], our_partitions: &mut [u32]) {
    for (s, out) in strings.iter().zip(our_partitions) {
        let mut ret = 1u32;
        let mut mask = 0u32;
        for &b in s.as_bytes() {
            let m = 1u32 << (b - b'A');
            if mask & m != 0 {
                mask = 0;
                ret += 1;
            }
            mask |= m;
        }
        *out = ret
    }
}

fn array_noclear_partition(strings: &[String], our_partitions: &mut [u32]) {
    let mut prev = [0u32; 256];
    for (s, out) in strings.iter().zip(our_partitions) {
        prev.fill(0);
        let mut ret = 1u32;
        for &b in s.as_bytes() {
            if prev[b as usize] == ret {
                ret += 1;
            }
            prev[b as usize] = ret
        }
        *out = ret
    }
}
