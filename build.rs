use std::env;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Result};

fn main() -> Result<()> {
    if cfg!(feature = "ispc") {
        compile_ispc()?;
    }

    Ok(())
}

fn compile_ispc() -> Result<()> {
    let out_dir = env::var_os("OUT_DIR").map(PathBuf::from).unwrap();
    let ispc_paths = glob::glob("./**/*.ispc")?;

    let mut build = cc::Build::new();
    for p in ispc_paths {
        let ispc_path = p?;
        let file_name = ispc_path.file_stem()
            .expect("Strange ispc filename")
            .to_str()
            .unwrap();
        let obj_path = out_dir.join(format!("{}_ispc", file_name))
            .with_extension(obj_extension());
        let header_path = out_dir.join(format!("{}_ispc", file_name))
            .with_extension("h");
        build.object(&obj_path);

        let args = ispc_args()?;
        // Windows users must place ispc.exe in PATH.
        let output = Command::new("ispc")
            .args(&args)
            .arg(&ispc_path)
            .arg("-o")
            .arg(&obj_path)
            .arg("-h")
            .arg(&header_path)
            .output()?;

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                println!("cargo:warning={}: {}", ispc_path.display(), line);
            }
        }
        if !output.status.success() {
            bail!("Failed to compile ISPC source file {}", ispc_path.display());
        }

        // TODO: This forces rebuild on each iteration, check out the ways to skip the rebuild
        // if nothing changed.
        // println!("cargo:rerun-if-changed={:?}", ispc_path);
    }
    build.compile("scratchers_ispc");

    Ok(())
}

fn ispc_args() -> Result<Vec<String>> {
    let mut ret = vec![];

    let opt_level = env::var("OPT_LEVEL").unwrap_or("0".to_string());
    ret.push(format!("-O{}", opt_level));

    let debug = env::var("DEBUG").map(|v| v == "true").unwrap_or(false);
    if debug {
        ret.push("-g".to_string());
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS not set");
    if target_os != "android" && target_os != "ios" && target_os != "linux"
        && target_os != "macos" && target_os != "windows" {
        bail!("Unsupported target OS {}", target_os);
    }
    ret.push(format!("--target-os={}", target_os));

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH not set");
    let target = target_for_arch(&target_arch)?;
    ret.push(format!("--arch={}", target_arch));
    ret.push(format!("--target={}", target));

    Ok(ret)
}

fn target_for_arch(arch: &str) -> Result<&'static str> {
    match arch {
        "x86" => Ok("sse2-i32x8"),
        "x86_64" => Ok("avx2-i32x8"),
        "aarch64" => Ok("neon-i32x8"),
        _ => bail!("Unsupported target arch {}", arch),
    }
}

fn obj_extension() -> &'static str {
    if cfg!(target_os = "windows") {
        "obj"
    } else {
        "o"
    }
}