use anyhow::Result;

mod android;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("Error: {}", e);
        std::process::exit(-1);
    }
}

#[derive(argh::FromArgs)]
#[argh(description = "Various cargo-related tasks")]
struct CliArgs {
    #[argh(subcommand)]
    command: Commands,
}

#[derive(argh::FromArgs)]
#[argh(subcommand)]
enum Commands {
    Android(android::Args),
}

fn try_main() -> Result<()> {
    let cli: CliArgs = argh::from_env();
    match cli.command {
        Commands::Android(a) => android::run(&a),
    }
}
