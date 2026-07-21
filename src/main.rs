use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::fs;
use std::net;

use serde::{Deserialize, Serialize};
use shakmaty::{Chess, Position, CastlingMode};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::san::San;

// ─── ANSI colour helpers ───────────────────────────────────────────────────────
const RESET: &str  = "\x1b[0m";
const BOLD:  &str  = "\x1b[1m";
const RED:   &str  = "\x1b[91m";
const GREEN: &str  = "\x1b[92m";
const BLUE:  &str  = "\x1b[94m";
const MAGENTA:&str = "\x1b[95m";
const CYAN:  &str  = "\x1b[96m";

// ─── Prefs ─────────────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct Prefs {
    engine:  String,
    depth:   u32,
    threads: Option<String>,
    #[serde(rename = "userId")]
    user_id: String,
    syzygy:  Option<String>,
}

// ─── FEN response ──────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct FenData {
    id: u64,
    fen: String,
}

// ─── Score submission (mirrors app/models/score.py::ScoreSubmission) ──────────
#[derive(Serialize)]
struct ScoreSubmission {
    #[serde(rename = "userId")]
    user_id: String,
    engine: String,
    #[serde(rename = "fenId")]
    fen_id: u64,
    ply: u32,
    score: Option<String>,
    nps: Option<String>,
    nodes: Option<String>,
    time: Option<String>,
    pv: Option<String>,
    mate: Option<String>,
}

/// Turn an empty UCI-parsed string into `None`, matching the optional-string
/// fields on the server's Pydantic model.
fn non_empty(s: &str) -> Option<String> {
    if s.is_empty() { None } else { Some(s.to_string()) }
}

/// Convert a space-separated UCI principal variation (e.g. "e2e4 e7e5 g1f3")
/// into SAN (e.g. "e4 e5 Nf3"), starting from the given FEN.
///
/// Returns `Err` with a description if the FEN or any move in the PV is
/// invalid/illegal — callers should fall back to the raw UCI string rather
/// than fail the whole submission over a cosmetic conversion.
fn uci_pv_to_san(fen: &str, pv: &str) -> Result<String, String> {
    let setup: Fen = fen
        .parse()
        .map_err(|e| format!("fen parse error: {:?}", e))?;
    let mut pos: Chess = setup
        .into_position(CastlingMode::Standard)
        .map_err(|e| format!("position error: {:?}", e))?;

    let mut sans = Vec::new();
    for uci_str in pv.split_whitespace() {
        let uci_move: UciMove = uci_str
            .parse()
            .map_err(|e| format!("uci parse error on {}: {:?}", uci_str, e))?;
        let m = uci_move
            .to_move(&pos)
            .map_err(|e| format!("illegal move {}: {:?}", uci_str, e))?;
        sans.push(San::from_move(&pos, m.clone()).to_string());
        pos = pos
            .play(m)
            .map_err(|e| format!("play error on {}: {:?}", uci_str, e))?;
    }

    Ok(sans.join(" "))
}

// ─── Host-info helper ──────────────────────────────────────────────────────────
fn ocdb_host_info(user_id: &str, session_id: &str) {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();

    let hostname  = System::host_name().unwrap_or_default();
    let os_name   = System::name().unwrap_or_default();
    let os_ver    = System::os_version().unwrap_or_default();
    let kernel    = System::kernel_version().unwrap_or_default();

    let total_mem = sys.total_memory();   // bytes
    let cpu_count = sys.cpus().len();

    // Best-effort local IP
    let ip = net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| { s.connect("8.8.8.8:80")?; s.local_addr() })
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "unknown".into());

    let mut params = std::collections::HashMap::new();
    params.insert("id",       user_id.to_string());
    params.insert("pid",      session_id.to_string());
    params.insert("ip",       ip);
    params.insert("hostname", hostname);
    params.insert("system",   os_name);
    params.insert("release",  os_ver);
    params.insert("version",  kernel);
    params.insert("log_cores",cpu_count.to_string());
    params.insert("memory",   format!("{:.2}MB", total_mem as f64 / 1024.0 / 1024.0));

    // NOTE: left disabled, same as before — not re-enabling this without
    // understanding why it posts to a bare LAN IP rather than openchessdb.org.
    //let client = reqwest::blocking::Client::new();
    //let _ = client
    //    .post("http://192.168.1.102/v2/updatenodes.php")
    //    .form(&params)
    //    .send();
    let _ = params; // silence unused-var warning while the above stays disabled
}

// ─── Rainbow spinner ───────────────────────────────────────────────────────────
struct RainbowSpinner {
    stop_flag: Arc<AtomicBool>,
    handle:    Option<thread::JoinHandle<()>>,
}

impl RainbowSpinner {
    fn new() -> Self {
        RainbowSpinner { stop_flag: Arc::new(AtomicBool::new(false)), handle: None }
    }

    fn start(&mut self) {
        let flag = Arc::clone(&self.stop_flag);
        self.handle = Some(thread::spawn(move || {
            let frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];
            let colors = ["\x1b[31m","\x1b[33m","\x1b[32m","\x1b[36m","\x1b[34m","\x1b[35m"];
            let mut i = 0usize;
            while !flag.load(Ordering::Relaxed) {
                let frame = frames[i % frames.len()];
                let color = colors[i % colors.len()];
                print!("\r{}{}{}  ", color, frame, RESET);
                let _ = std::io::stdout().flush();
                i += 1;
                thread::sleep(Duration::from_millis(100));
            }
            print!("\r   \r");
            let _ = std::io::stdout().flush();
        }));
    }

    fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.stop_flag.store(false, Ordering::Relaxed);
    }
}

// ─── Simple depth-progress bar (no external crate) ────────────────────────────
fn print_depth_bar(current: u32, total: u32, width: usize) {
    let pct  = current as f32 / total as f32;
    let done = (pct * width as f32) as usize;
    let bar  = format!("[{}{}]", "█".repeat(done), "░".repeat(width - done));
    print!("\r      Depth {}/{} {} ", current, total, bar);
    let _ = std::io::stdout().flush();
}

// ─── UCI line parser ───────────────────────────────────────────────────────────
#[derive(Default)]
struct InfoLine {
    depth:    u32,
    score:    String,
    nps:      String,
    nodes:    String,
    time_ms:  String,
    mate:     String,
    pv:       String,
}

fn parse_info(line: &str) -> InfoLine {
    let mut info = InfoLine::default();
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" if i + 1 < tokens.len() => { info.depth   = tokens[i+1].parse().unwrap_or(0); i += 1; }
            "cp"    if i + 1 < tokens.len() => { info.score   = tokens[i+1].into(); i += 1; }
            "nps"   if i + 1 < tokens.len() => { info.nps     = tokens[i+1].into(); i += 1; }
            "nodes" if i + 1 < tokens.len() => { info.nodes   = tokens[i+1].into(); i += 1; }
            "time"  if i + 1 < tokens.len() => { info.time_ms = tokens[i+1].into(); i += 1; }
            "mate"  if i + 1 < tokens.len() => { info.mate    = tokens[i+1].into(); i += 1; }
            "pv"    if i + 1 < tokens.len() => {
                // everything after "pv" is the principal variation
                info.pv = tokens[i+1..].join(" ");
                break;
            }
            _ => {}
        }
        i += 1;
    }
    info
}

// ─── Format elapsed seconds as H:MM:SS ────────────────────────────────────────
fn fmt_elapsed(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 { format!("{}:{:02}:{:02}", h, m, s) } else { format!("{}:{:02}", m, s) }
}

// ─── Main ──────────────────────────────────────────────────────────────────────
fn main() {
    let header_art = r#"
░█▀█░█▀▀░█▀▄░█▀▄░░░█▀▀░█░░░▀█▀░█▀▀░█▀█░▀█▀░░░█░█░▀▀▄░░░░▄▀▄░░
░█░█░█░░░█░█░█▀▄░░░█░░░█░░░░█░░█▀▀░█░█░░█░░░░▀▄▀░▄▀░░░░░█/█░░
░▀▀▀░▀▀▀░▀▀░░▀▀░░░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░░░░▀░░▀▀▀░▀░░░▀░░░
"#;
    println!("{}", header_art);
    println!("\nThank you for contributing your compute time and resources.\n");

    // ── Load prefs ──────────────────────────────────────────────────────────────
    let prefs_raw = fs::read_to_string("prefs.json").expect("Cannot open prefs.json");
    let prefs: Prefs = serde_json::from_str(&prefs_raw).expect("Invalid prefs.json");

    let session_id = uuid::Uuid::new_v4().to_string();
    ocdb_host_info(&prefs.user_id, &session_id);

    // ── Start chess engine ──────────────────────────────────────────────────────
    let mut proc = Command::new(&prefs.engine)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start engine");

    let stdin  = proc.stdin.as_mut().expect("No stdin");
    let stdout = BufReader::new(proc.stdout.take().expect("No stdout"));

    // UCI handshake
    write!(stdin, "uci\n").unwrap();
    stdin.flush().unwrap();

    let mut engine_name = String::from("unknown");
    let mut lines = stdout.lines();

    for line in lines.by_ref() {
        let line = line.unwrap_or_default();
        if line.contains("id name") {
            engine_name = line.replace("id name", "").trim().to_string();
        }
        if line.contains("uciok") { break; }
    }

    // Optional thread count
    if let Some(ref t) = prefs.threads {
        write!(stdin, "setoption name Threads value {}\n", t).unwrap();
        stdin.flush().unwrap();
        println!("🛠️  Setting threads to {}", t);
    }

    // Optional Syzygy path
    if let Some(ref path) = prefs.syzygy {
        write!(stdin, "setoption name SyzygyPath value {}\n", path).unwrap();
        stdin.flush().unwrap();
        println!("🛠️  Setting SyzygyPath to {}", path);
        // consume two info lines the engine emits
        lines.by_ref().take(2).for_each(|_| {});
    }

    let client = reqwest::blocking::Client::new();

    // ── Main loop ───────────────────────────────────────────────────────────────
    loop {
        // Fetch FEN
        let resp = match client.get("https://openchessdb.org/v2/getfen").send() {
            Ok(r)  => r,
            Err(e) => { eprintln!("{}Failed to fetch FEN: {}{}", RED, e, RESET); thread::sleep(Duration::from_secs(2)); continue; }
        };
        let body = match resp.text() {
            Ok(b) => b,
            Err(e) => { eprintln!("{}Failed to read FEN response body: {}{}", RED, e, RESET); thread::sleep(Duration::from_secs(2)); continue; }
        };
        let fen_data: FenData = match serde_json::from_str(&body) {
            Ok(d)  => d,
            Err(e) => {
                eprintln!("{}Bad FEN JSON: {} — body was: {}{}", RED, e, body, RESET);
                thread::sleep(Duration::from_secs(2));
                continue;
            }
        };

        println!("{}", "▔".repeat(80));
        println!("\n{}FEN:{} {}", CYAN, RESET, fen_data.fen);

        // Position + go
        write!(stdin, "ucinewgame\n").unwrap();
        stdin.flush().unwrap();
        write!(stdin, "position fen {}\n", fen_data.fen).unwrap();
        stdin.flush().unwrap();
        write!(stdin, "go depth {}\n", prefs.depth).unwrap();
        stdin.flush().unwrap();

        println!("{}Start Analysis...{}", GREEN, RESET);

        let run_start  = Instant::now();
        let mut spinner = RainbowSpinner::new();
        spinner.start();

        let mut last_info = InfoLine::default();
        let mut last_ply  = 0u32;

        for line in lines.by_ref() {
            let line = match line { Ok(l) => l, Err(_) => break };

            if line.contains("info depth") {
                let info = parse_info(&line);

                // Progress bar on new ply
                if info.depth > last_ply {
                    print_depth_bar(info.depth, prefs.depth, 30);
                    last_ply = info.depth;
                }

                last_info = info;
            }

            if line.contains("bestmove") {
                spinner.stop();
                print_depth_bar(prefs.depth, prefs.depth, 30);
                println!("\n");

                // Convert PV from UCI to SAN for display/submission. If
                // conversion fails for any reason (unexpected FEN shape,
                // illegal move reported by the engine, etc.) fall back to
                // the raw UCI string rather than losing the data.
                let pv_display = if last_info.pv.is_empty() {
                    String::new()
                } else {
                    match uci_pv_to_san(&fen_data.fen, &last_info.pv) {
                        Ok(san) => san,
                        Err(e) => {
                            eprintln!("{}PV UCI->SAN conversion failed: {}{}", RED, e, RESET);
                            last_info.pv.clone()
                        }
                    }
                };

                let pv_short = pv_display.chars().take(20).collect::<String>();
                let elapsed  = run_start.elapsed().as_secs();

                println!("{}{}Engine Analysis Complete{}", BOLD, MAGENTA, RESET);
                println!("┌───────────────┬─────────────┐");
                println!("│ Ply           │ {:<11} │", last_info.depth);
                println!("│ Score         │ {:<11} │", last_info.score);
                println!("│ NPS           │ {:<11} │", last_info.nps);
                println!("│ Mate          │ {:<11} │", last_info.mate);
                println!("│ PV            │ {:<11} │", pv_short);
                println!("│ Nodes         │ {:<11} │", last_info.nodes);
                println!("│ Time(ms)      │ {:<11} │", last_info.time_ms);
                println!("└───────────────┴─────────────┘");

                // Send results — JSON body matching ScoreSubmission on the server.
                let payload = ScoreSubmission {
                    user_id: prefs.user_id.clone(),
                    engine: engine_name.clone(),
                    fen_id: fen_data.id,
                    ply: last_info.depth,
                    score: non_empty(&last_info.score),
                    nps: non_empty(&last_info.nps),
                    nodes: non_empty(&last_info.nodes),
                    time: non_empty(&last_info.time_ms),
                    pv: non_empty(&pv_display),
                    mate: non_empty(&last_info.mate),
                };

                match client
                    .post("https://openchessdb.org/v2/sendresults")
                    .json(&payload)
                    .send()
                {
                    Ok(r) => {
                        let status = r.status();
                        let resp_body = r.text().unwrap_or_default();
                        if status.is_success() {
                            println!("{}Response code:{} {}", BLUE, RESET, status);
                        } else {
                            println!(
                                "{}Response code:{} {} — body: {}",
                                RED, RESET, status, resp_body
                            );
                        }
                    }
                    Err(e) => println!("{}Failed to send results:{} {}", RED, RESET, e),
                }

                println!("{}Elapsed Time:{} {}\n", BLUE, RESET, fmt_elapsed(elapsed));
                break;
            }
        }
    }
}