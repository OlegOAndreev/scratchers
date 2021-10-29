use anyhow::{bail, Result};
use clap::{App, ArgMatches};

mod android;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("{}", e);
        std::process::exit(-1);
    }
}

const ANDROID_COMMAND: &str = "android";
const ANDROID_BUILD_COMMAND: &str = "build";

fn try_main() -> Result<()> {
    let app = App::new("xtask")
        .about("Must be run as cargo xtask")
        .subcommand(android::android_subcommand(ANDROID_COMMAND)
            .subcommand(android::android_build_subcommand(ANDROID_BUILD_COMMAND))
        );
    let matches = app.get_matches();

    if let Some(subcommand) = matches.subcommand_matches(ANDROID_COMMAND) {
        return run_android(subcommand);
    }

    bail!("Incorrect subcommand")
}

fn run_android(matches: &ArgMatches) -> Result<()> {
    let (ndk, arches) = android::parse_subcommand(matches)?;

    if let Some(subcommand_matches) = matches.subcommand_matches("build") {
        let args = android::parse_build_subcommand(subcommand_matches)?;
        for arch in arches {
            android::run_build(&ndk, arch, &args)?;
        }
        return Ok(());
    }
    bail!("Unknown subcommand for android")
}
