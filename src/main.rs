use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::fs;

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

// ─── Session telemetry ──────────────────────────────────────────────────────
// Reports anonymous hardware info to openchessdb.org so the dashboard can
// show current worker fleet composition. Every call here is best-effort:
// telemetry is incidental to ocdb-client's real job (scoring FENs), so a
// failure here must never interrupt or abort analysis.

#[derive(Serialize, Debug, Clone)]
struct Snapshot {
    os: String,
    arch: String,
    cpu_tier: String,
    logical_cores: usize,
    physical_cores: Option<usize>,
    ram_total_mb: u64,
}

#[derive(Deserialize, Debug)]
struct CreateSessionResponse {
    session_id: String,
    #[allow(dead_code)]
    created_at: String,
}

/// Returns true if this is an AMD CPU whose bmi2 implementation is known to
/// be slow (pre-Zen3 microcode emulation of pdep/pext) rather than a real
/// fast path. Intel bmi2 is fast everywhere it's supported, so this only
/// matters for AMD.
fn is_amd_slow_bmi2() -> bool {
    use raw_cpuid::CpuId;
    let cpuid = CpuId::new();

    let is_amd = cpuid
        .get_vendor_info()
        .map(|v| v.as_str() == "AuthenticAMD")
        .unwrap_or(false);

    if !is_amd {
        return false;
    }

    // Zen 3 corresponds to effective family 0x19+. Zen/Zen+/Zen2 (family
    // 0x17) have the slow microcoded pdep/pext. If we can't read family
    // info at all, assume the slow path since that's the safer default.
    match cpuid.get_feature_info() {
        Some(finfo) => finfo.extended_family_id() + finfo.family_id() < 0x19,
        None => true,
    }
}

/// Picks the best instruction-set tier this CPU can actually use well.
/// Falls back gracefully on non-x86_64 targets (e.g. a Raspberry Pi's
/// aarch64, or Apple Silicon).
fn cpu_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        let has_bmi2 = is_x86_feature_detected!("bmi2") && !is_amd_slow_bmi2();

        if is_x86_feature_detected!("avx512f") {
            "avx512"
        } else if has_bmi2 {
            "bmi2"
        } else if is_x86_feature_detected!("avx2") {
            "avx2"
        } else if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
            "sse41-popcnt"
        } else {
            "generic"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "generic"
    }
}

fn collect_snapshot() -> Snapshot {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();

    Snapshot {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_tier: cpu_tier().to_string(),
        logical_cores: sys.cpus().len(),
        physical_cores: sys.physical_core_count(),
        ram_total_mb: sys.total_memory() / 1024 / 1024,
    }
}

/// Calls /v2/create-session. Returns None (rather than erroring out the
/// whole program) if the API is unreachable — the worker can still fetch
/// FENs and submit scores without a session, it just won't show up on the
/// dashboard until the next successful attempt.
fn create_session(agent: &ureq::Agent) -> Option<String> {
    match agent.get("https://openchessdb.org/v2/create-session").call() {
        Ok(resp) => match resp.into_json::<CreateSessionResponse>() {
            Ok(r) => Some(r.session_id),
            Err(e) => {
                eprintln!("{}create-session: bad response: {}{}", RED, e, RESET);
                None
            }
        },
        Err(e) => {
            eprintln!("{}create-session failed: {}{}", RED, e, RESET);
            None
        }
    }
}

/// Result of a heartbeat attempt, so the caller can react to a purged
/// session (404) without polling or retrying blindly.
enum HeartbeatResult {
    Ok,
    SessionGone,
    OtherError,
}

/// Calls /v2/update-session with the current hardware snapshot. This both
/// reports hardware info and (via the server's ON UPDATE CURRENT_TIMESTAMP
/// on last_seen) acts as the heartbeat that keeps the session alive and out
/// of the 1-hour purge — no separate timer needed since this is called once
/// per FEN cycle, well under an hour even on modest hardware.
fn update_session(agent: &ureq::Agent, session_id: &str, snapshot: &Snapshot) -> HeartbeatResult {
    let mut req = agent
        .get("https://openchessdb.org/v2/update-session")
        .query("session_id", session_id)
        .query("os", &snapshot.os)
        .query("arch", &snapshot.arch)
        .query("cpu_tier", &snapshot.cpu_tier)
        .query("logical_cores", &snapshot.logical_cores.to_string())
        .query("ram_total_mb", &snapshot.ram_total_mb.to_string());

    if let Some(p) = snapshot.physical_cores {
        req = req.query("physical_cores", &p.to_string());
    }

    match req.call() {
        Ok(_) => HeartbeatResult::Ok,
        Err(ureq::Error::Status(404, _)) => HeartbeatResult::SessionGone,
        Err(e) => {
            eprintln!("{}update-session (heartbeat) failed: {}{}", RED, e, RESET);
            HeartbeatResult::OtherError
        }
    }
}

/// Sends the heartbeat and self-heals if the session was purged (e.g. the
/// worker was asleep/disconnected long enough that last_seen aged out).
/// Transient errors are just logged and retried on the next cycle — no
/// extra requests beyond the one heartbeat unless the session is genuinely
/// gone, in which case exactly one extra create-session call re-establishes
/// it.
fn heartbeat(agent: &ureq::Agent, session_id: &mut Option<String>, snapshot: &Snapshot) {
    let Some(sid) = session_id.clone() else {
        // No session yet (startup create-session failed) — try again now,
        // piggybacking on this workload cycle instead of a separate retry loop.
        *session_id = create_session(agent);
        return;
    };

    match update_session(agent, &sid, snapshot) {
        HeartbeatResult::Ok => {}
        HeartbeatResult::SessionGone => {
            eprintln!("{}Session expired — creating a new one{}", RED, RESET);
            if let Some(new_sid) = create_session(agent) {
                // Best-effort initial report on the new session; ignore the
                // result since the next cycle's heartbeat will retry anyway.
                let _ = update_session(agent, &new_sid, snapshot);
                *session_id = Some(new_sid);
            } else {
                *session_id = None;
            }
        }
        HeartbeatResult::OtherError => {} // already logged; next cycle retries
    }
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

// ─── HTTP helpers (ureq) ───────────────────────────────────────────────────────
// ureq's default agent has NO built-in timeout unless the OS-level TCP
// connect hangs. We build an agent with explicit connect/read/write timeouts
// so a stalled connection surfaces as a clear error within a bounded time
// instead of hanging the whole program forever.
fn build_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(20))
        .timeout_write(Duration::from_secs(10))
        .timeout(Duration::from_secs(30)) // overall request timeout
        .build()
}

fn fetch_fen(agent: &ureq::Agent) -> Result<FenData, String> {
    let resp = agent
        .get("https://openchessdb.org/v2/getfen")
        .call()
        .map_err(|e| format!("request failed: {}", e))?;

    let body = resp
        .into_string()
        .map_err(|e| format!("failed to read body: {}", e))?;

    serde_json::from_str::<FenData>(&body)
        .map_err(|e| format!("bad FEN JSON: {} — body was: {}", e, body))
}

fn submit_score(agent: &ureq::Agent, payload: &ScoreSubmission) -> Result<(u16, String), String> {
    match agent
        .post("https://openchessdb.org/v2/sendresults")
        .send_json(payload)
    {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.into_string().unwrap_or_default();
            Ok((status, body))
        }
        // ureq treats non-2xx as an Err(Status), so we still need to recover
        // the status code and body from that case rather than treating it
        // as a hard failure.
        Err(ureq::Error::Status(code, resp)) => {
            let body = resp.into_string().unwrap_or_default();
            Ok((code, body))
        }
        Err(e) => Err(format!("{}", e)),
    }
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

    let agent = build_agent();

    // ── Session telemetry (best-effort — never blocks or aborts the worker) ────
    let snapshot = collect_snapshot();
    let mut session_id: Option<String> = create_session(&agent);
    match &session_id {
        Some(sid) => {
            println!("🔑 Session: {}", sid);
            let _ = update_session(&agent, sid, &snapshot);
        }
        None => println!("⚠️  Session telemetry unavailable — continuing without it"),
    }

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

    // Optional Syzygy path — skip entirely if empty, since "setoption ... value "
    // with a blank value is meaningless and some engines respond to it oddly.
    if let Some(ref path) = prefs.syzygy {
        if !path.is_empty() {
            write!(stdin, "setoption name SyzygyPath value {}\n", path).unwrap();
            stdin.flush().unwrap();
            println!("🛠️  Setting SyzygyPath to {}", path);
        }
    }

    // Sync with the engine using the UCI protocol's actual handshake
    // (isready/readyok) instead of guessing how many lines setoption calls
    // will print. Engines emit a variable number of info/string lines here —
    // sometimes zero — so blindly consuming a fixed count (as the old code
    // did) can block forever waiting for output that never arrives.
    write!(stdin, "isready\n").unwrap();
    stdin.flush().unwrap();
    for line in lines.by_ref() {
        let line = line.unwrap_or_default();
        if line.trim() == "readyok" { break; }
    }

    // ── Main loop ───────────────────────────────────────────────────────────────
    loop {
        // Fetch FEN
        eprintln!("{}Requesting FEN...{}", CYAN, RESET);
        let fen_data = match fetch_fen(&agent) {
            Ok(d)  => d,
            Err(e) => {
                eprintln!("{}Failed to fetch FEN: {}{}", RED, e, RESET);
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

                // Cap how wide the value column is allowed to grow (long PVs
                // would otherwise stretch the box arbitrarily wide); anything
                // longer gets truncated with an ellipsis.
                const MAX_VALUE_WIDTH: usize = 40;
                let pv_short = if pv_display.chars().count() > MAX_VALUE_WIDTH {
                    format!("{}…", pv_display.chars().take(MAX_VALUE_WIDTH - 1).collect::<String>())
                } else {
                    pv_display.clone()
                };
                let elapsed  = run_start.elapsed().as_secs();

                let depth_str = last_info.depth.to_string();
                let rows: [(&str, &str); 7] = [
                    ("Ply",      &depth_str),
                    ("Score",    &last_info.score),
                    ("NPS",      &last_info.nps),
                    ("Mate",     &last_info.mate),
                    ("PV",       &pv_short),
                    ("Nodes",    &last_info.nodes),
                    ("Time(ms)", &last_info.time_ms),
                ];

                // Size the columns to the widest label/value actually being
                // printed (with a floor and cap on the value column) so the
                // border always lines up, no matter how long the PV is.
                let label_w = rows.iter().map(|(l, _)| l.chars().count()).max().unwrap_or(0);
                let value_w = rows.iter()
                    .map(|(_, v)| v.chars().count())
                    .max()
                    .unwrap_or(0)
                    .clamp(11, MAX_VALUE_WIDTH);

                let top = format!("┌{}┬{}┐", "─".repeat(label_w + 2), "─".repeat(value_w + 2));
                let sep = format!("├{}┼{}┤", "─".repeat(label_w + 2), "─".repeat(value_w + 2));
                let bot = format!("└{}┴{}┘", "─".repeat(label_w + 2), "─".repeat(value_w + 2));

                println!("{}{}Engine Analysis Complete{}", BOLD, MAGENTA, RESET);
                println!("{}", top);
                for (i, (label, value)) in rows.iter().enumerate() {
                    println!("│ {:<lw$} │ {:<vw$} │", label, value, lw = label_w, vw = value_w);
                    if i == 3 {
                        // visually separate the PV row from the stats above it
                        println!("{}", sep);
                    }
                }
                println!("{}", bot);

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

                match submit_score(&agent, &payload) {
                    Ok((status, resp_body)) => {
                        if (200..300).contains(&status) {
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

                // Heartbeat the session right alongside this workload send —
                // self-heals if the session was purged (404), otherwise this
                // is the sole extra request per FEN cycle.
                heartbeat(&agent, &mut session_id, &snapshot);

                println!("{}Elapsed Time:{} {}\n", BLUE, RESET, fmt_elapsed(elapsed));
                break;
            }
        }
    }
}