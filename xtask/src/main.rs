use std::{env, fs};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::{App, Arg, ArgMatches, SubCommand};

fn main() {
    if let Err(e) = try_main() {
        eprintln!("{}", e);
        std::process::exit(-1);
    }
}

fn try_main() -> Result<()> {
    let app = App::new("xtask")
        .about("Must be run as cargo xtask")
        .subcommand(SubCommand::with_name("android")
            .about("Android-related commands")
            .arg(Arg::with_name("api-level")
                .long("--api-level")
                .value_name("VERSION")
                .help("Android API level")
                .default_value("26"))
            .arg(Arg::with_name("sdk-root")
                .long("--sdk-root")
                .value_name("PATH")
                .help("Path to Android SDK (if not set, search in standard locations")
                .required(false))
            .arg(Arg::with_name("ndk-root")
                .long("--ndk-root")
                .value_name("PATH")
                .help("Path to Android NDK (if not set, search in standard locations")
                .required(false))
            .subcommand(SubCommand::with_name("build")
                .about("Sets up cross-compiling environment (including cc & cmake) and runs cargo build.")
                .arg(Arg::with_name("args")
                    .help("Arguments to pass to cargo build")
                    .multiple(true)
                    .last(true))));
    let matches = app.get_matches();

    if let Some(subcommand) = matches.subcommand_matches("android") {
        return run_android(subcommand);
    }

    bail!("Incorrect subcommand")
}

const TARGET_TRIPLE: &str = "aarch64-linux-android";
const TARGET_ABI: &str = "arm64-v8a";

struct AndroidNdk {
    api_level: String,
    sdk_root: PathBuf,
    ndk_root: PathBuf,
}

fn run_android(matches: &ArgMatches) -> Result<()> {
    let ndk = find_ndk(matches.value_of("api-level"), matches.value_of("sdk-root"),
                       matches.value_of("ndk-root"))?;
    eprintln!("Using API level {}, SDK root {:?}, NDK root {:?}", ndk.api_level, ndk.sdk_root,
              ndk.ndk_root);

    if let Some(subcommand) = matches.subcommand_matches("build") {
        return run_android_build(subcommand, &ndk);
    }
    bail!("Unknown subcommand for android")
}

fn find_ndk(
    arg_api_level: Option<&str>,
    arg_sdk_root: Option<&str>,
    arg_ndk_root: Option<&str>,
) -> Result<AndroidNdk> {
    let sdk_root = find_sdk_root(arg_sdk_root)?;
    let ndk_root = find_ndk_root(&sdk_root, arg_ndk_root)?;
    if !sdk_root.exists() {
        bail!("Could not find Android SDK at {:?}", sdk_root);
    }
    if !ndk_root.exists() {
        bail!("Could not find Android NDK at {:?}", ndk_root);
    }

    Ok(AndroidNdk {
        api_level: arg_api_level.unwrap().to_string(),
        sdk_root,
        ndk_root,
    })
}

fn find_sdk_root(arg_sdk_root: Option<&str>) -> Result<PathBuf> {
    if let Some(root) = arg_sdk_root {
        return Ok(PathBuf::from(root));
    }
    // This looks like the "official" variable (e.g. see https://github.com/actions/virtual-environments/issues/2426)
    if let Some(env_root) = env::var("ANDROID_SDK_ROOT").ok() {
        return Ok(PathBuf::from(env_root));
    }
    let default_root = get_default_sdk_location()?;
    if default_root.exists() {
        return Ok(default_root);
    }
    bail!("Could not find Android SDK, install and set env var ANDROID_SDK_ROOT")
}

// See https://stackoverflow.com/questions/43330176/what-is-the-default-android-sdk-path-used-by-android-studio
fn get_default_sdk_location() -> Result<PathBuf> {
    if cfg!(target_os = "windows") {
        match dirs::data_local_dir() {
            None => bail!("Could not get AppData dir"),
            Some(local_dir) => Ok(local_dir.join("Android").join("Sdk"))
        }
    } else if cfg!(target_os = "macos") {
        match dirs::home_dir() {
            None => bail!("Could not get home dir"),
            Some(home_dir) => Ok(home_dir.join("Library").join("Android").join("sdk"))
        }
    } else if cfg!(target_os = "linux") {
        match dirs::home_dir() {
            None => bail!("Could not get home dir"),
            Some(home_dir) => Ok(home_dir.join("Android").join("Sdk"))
        }
    } else {
        bail!("Unsupported host OS")
    }
}

fn find_ndk_root(sdk_root: &Path, arg_ndk_root: Option<&str>) -> Result<PathBuf> {
    if let Some(root) = arg_ndk_root {
        return Ok(PathBuf::from(root));
    }
    // This looks like the "official" variable (e.g. see https://github.com/actions/virtual-environments/issues/2426)
    if let Some(env_root) = env::var("ANDROID_NDK_ROOT").ok() {
        return Ok(PathBuf::from(env_root));
    }
    let ndk_for_sdk = sdk_root.join("ndk");
    if ndk_for_sdk.is_dir() {
        return find_max_ndk(&ndk_for_sdk);
    }
    let ndk_bundle_for_sdk = sdk_root.join("ndk-bundle");
    if ndk_bundle_for_sdk.is_dir() {
        return Ok(ndk_bundle_for_sdk);
    }
    bail!("Could not find Android NDK, install and set env var ANDROID_NDK_ROOT")
}

fn find_max_ndk(base_path: &Path) -> Result<PathBuf> {
    let file_name = fs::read_dir(base_path)
        .context("Reading NDK dir")?
        .filter_map(|e| e.ok()
            .and_then(|e| e.file_name()
                .to_str()
                .map(|n| n.to_string()))
        )
        .filter_map(|file_name| semver::Version::parse(&file_name)
            .ok()
            .map(|version| (version, file_name))
        )
        .max_by_key(|(version, _)| version.clone());
    match file_name {
        None => bail!("Could not find any NDK in {:?}", base_path),
        Some((_, file_name)) => Ok(base_path.join(file_name))
    }
}

#[cfg(target_os = "windows")]
const HOST_TOOLCHAIN: &str = "windows-x86_64";
#[cfg(target_os = "macos")]
const HOST_TOOLCHAIN: &str = "darwin-x86_64";
#[cfg(target_os = "linux")]
const HOST_TOOLCHAIN: &str = "linux-x86_64";

#[cfg(target_os = "windows")]
const CLANG_EXT: &str = ".cmd";
#[cfg(not(target_os = "windows"))]
const CLANG_EXT: &str = "";

#[cfg(target_os = "windows")]
const EXE_EXT: &str = ".exe";
#[cfg(not(target_os = "windows"))]
const EXE_EXT: &str = "";

fn run_android_build(matches: &ArgMatches, ndk: &AndroidNdk) -> Result<()> {
    let build_args = matches.values_of_lossy("args").unwrap_or(vec![]);
    let verbose = build_args.iter().any(|v| v == "-v");

    let ndk_toolchain_dir = ndk.ndk_root
        .join("toolchains")
        .join("llvm")
        .join("prebuilt")
        .join(HOST_TOOLCHAIN)
        .join("bin");
    // Env vars for cc crate.
    let target_cc = ndk_toolchain_dir.join(format!("{}{}-clang{}", TARGET_TRIPLE, ndk.api_level, CLANG_EXT));
    let target_cxx = ndk_toolchain_dir.join(format!("{}{}-clang++{}", TARGET_TRIPLE, ndk.api_level, CLANG_EXT));
    let target_ar = ndk_toolchain_dir.join(format!("llvm-ar{}", EXE_EXT));
    // The c++abi library is required at least for some NDKs, see:
    // * https://github.com/Kitware/CMake/commit/4dca07882944ec5c1d87edf1b7df9f3c7294e0d0
    // * https://android.googlesource.com/platform/ndk/+/43b2de34ef9e3a70573fe51a9e069f985a4be5b9/build/cmake/android.toolchain.cmake#368
    // We use force-frame-pointers because the performance impact is absolutely minimal while the
    // profiling becomes a lot easier.
    let rustflags = format!("-C linker={} -C link-arg=-lc++abi -C force-frame-pointers=yes", target_cxx.display());
    // Set CMAKE_TOOLCHAIN_FILE env var for cmake crate.
    let original_cmake_toolchain_file = ndk.ndk_root
        .join("build")
        .join("cmake")
        .join("android.toolchain.cmake");
    let cmake_toolchain_contents = format!("set(ANDROID_ABI {})
set(ANDROID_PLATFORM {})
include({})", TARGET_ABI, ndk.api_level, original_cmake_toolchain_file.display());
    let cmake_toolchain_file = tempfile::Builder::new()
        .prefix("android.toolchain.")
        .suffix(".cmake")
        .tempfile().context("Creating temporary toolchain.cmake")?;
    if verbose {
        eprintln!("Setting TARGET_CC={}", target_cc.display());
        eprintln!("Setting TARGET_CXX={}", target_cxx.display());
        eprintln!("Setting TARGET_AR={}", target_ar.display());
        eprintln!("Setting RUSTFLAGS={}", rustflags);
        eprintln!("Setting CMAKE_TOOLCHAIN_FILE={}", cmake_toolchain_file.path().display());
        eprintln!("Writing\n\n{}\n\nto {}", cmake_toolchain_contents, cmake_toolchain_file.path().display())
    }
    fs::write(&cmake_toolchain_file, cmake_toolchain_contents)
        .context("Could not write toolchain.cmake")?;

    let cargo_bin = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cargo_cmd = Command::new(cargo_bin);
    cargo_cmd
        .env("TARGET_CC", target_cc)
        .env("TARGET_CXX", target_cxx)
        .env("TARGET_AR", target_ar)
        .env("RUSTFLAGS", rustflags)
        .env("CMAKE_TOOLCHAIN_FILE", cmake_toolchain_file.path())
        .arg("build")
        .arg("--target")
        .arg(TARGET_TRIPLE)
        .args(build_args);
    if verbose {
        eprintln!("Running {:?}", cargo_cmd);
    }
    let exit_code = cargo_cmd.status().context("Running cargo")?;
    if exit_code.success() {
        Ok(())
    } else {
        bail!("Cargo exited with code {:?}", exit_code.code())
    }
}
