use clap::Parser;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use qwen3_5_rs::inference::InferenceEngine;
use qwen3_5_rs::sampling::SamplingConfig;

/// Educational Qwen3-0.6B inference engine
#[derive(Parser, Debug)]
#[command(name = "qwen3-5-rs", version, about = "Educational Qwen3-0.6B inference engine")]
struct Args {
    /// Path to model directory containing config.json, tokenizer.json, model.safetensors
    #[arg(short, long)]
    model_dir: String,

    /// Input prompt (single-turn mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Interactive chat mode
    #[arg(short, long, default_value_t = false)]
    interactive: bool,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(short, long, default_value_t = 1.0)]
    temperature: f32,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();

    // Build sampling config from CLI args
    let sampling_config = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };

    // Load model
    println!("Loading model from {}...", args.model_dir);
    let mut engine = InferenceEngine::load(
        Path::new(&args.model_dir),
        sampling_config,
    ).unwrap_or_else(|e| {
        eprintln!("Error loading model: {}", e);
        std::process::exit(1);
    });
    println!("Model loaded successfully.");

    if args.interactive {
        interactive_mode(&mut engine, &args);
    } else if let Some(prompt) = &args.prompt {
        single_prompt_mode(&mut engine, prompt, args.max_tokens);
    } else {
        eprintln!("Error: either --prompt or --interactive is required");
        std::process::exit(1);
    }
}

/// Single-prompt mode: generate a response and exit.
fn single_prompt_mode(engine: &mut InferenceEngine, prompt: &str, max_tokens: usize) {
    let start = Instant::now();
    let mut token_count = 0usize;

    let _output = engine.generate_with_callback(prompt, max_tokens, |token_text| {
        token_count += 1;
        print!("{}", token_text);
        std::io::stdout().flush().unwrap();
    });

    let elapsed = start.elapsed();
    println!();

    if elapsed.as_secs_f64() > 0.0 {
        let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();
        eprintln!(
            "\nGenerated {} tokens in {:.2}s ({:.1} tokens/sec)",
            token_count,
            elapsed.as_secs_f64(),
            tokens_per_sec,
        );
    }
}

/// Interactive chat mode: loop reading user input and generating responses.
fn interactive_mode(engine: &mut InferenceEngine, args: &Args) {
    println!();
    println!("========================================");
    println!("  qwen3.5-rs Interactive Chat");
    println!("========================================");
    println!();
    println!("Type your message and press Enter to generate a response.");
    println!("Commands:");
    println!("  /reset  - Clear conversation history (KV cache)");
    println!("  quit    - Exit the program");
    println!("  exit    - Exit the program");
    println!();

    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => {
                // Ctrl+D (EOF)
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            break;
        }

        if input == "/reset" {
            engine.reset();
            println!("Conversation history cleared.");
            continue;
        }

        // Generate response with streaming output
        let start = Instant::now();
        let mut token_count = 0usize;

        let _output = engine.generate_with_callback(input, args.max_tokens, |token_text| {
            token_count += 1;
            print!("{}", token_text);
            std::io::stdout().flush().unwrap();
        });

        let elapsed = start.elapsed();
        println!();

        if elapsed.as_secs_f64() > 0.0 {
            let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();
            eprintln!(
                "\n[{} tokens in {:.2}s ({:.1} tokens/sec)]",
                token_count,
                elapsed.as_secs_f64(),
                tokens_per_sec,
            );
        }

        println!();
    }
}
