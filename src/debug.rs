use std::arch::asm;
use std::time::Duration;
use std::{env, process, thread};

const WAIT_FOR_DEBUGGER: Duration = Duration::from_secs(1);

pub fn open_debugger() {
    let url = format!(
        "vscode://vadimcn.vscode-lldb/launch/config?{{'request':'attach','pid':{}}}",
        process::id()
    );
    let vscode_bin = match env::var("VSCODE_BIN") {
        Ok(b) => {
            if b.is_empty() {
                "code".to_string()
            } else {
                b
            }
        }
        Err(_) => "code".to_string(),
    };
    process::Command::new(vscode_bin).arg("--open-url").arg(url).output().unwrap();
    thread::sleep(WAIT_FOR_DEBUGGER);
    breakpoint();
}

#[cfg(target_arch = "x86_64")]
pub fn breakpoint() {
    unsafe {
        asm!("int3");
    }
}

#[cfg(target_arch = "aarch64")]
pub fn breakpoint() {
    unsafe {
        asm!("brk #0x1");
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn breakpoint() {}
