use anyhow::{bail, Context, Result};
use binary_greedy_mesher_demo_rs as demo;
use demo::{get_xyz_key, get_zxy_index, CS, CS_P, CS_P3};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    output: PathBuf,
    chunks_per_side: u8,
    seed: u64,
    noise_scale: f32,
    height_scale: f32,
    octaves: u8,
    gain: f32,
    lacunarity: f32,
}

fn parse_args() -> Result<Args> {
    let mut args = env::args().skip(1);

    let mut out = PathBuf::from("levels/generated_level");
    let mut chunks_per_side: u8 = 1;
    let mut seed: u64 = 0;
    let mut noise_scale: f32 = 0.035;
    let mut height_scale: f32 = 1.0;
    let mut octaves: u8 = 5;
    let mut gain: f32 = 0.5;
    let mut lacunarity: f32 = 2.0;

    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--output" => {
                out = PathBuf::from(args.next().context("--output requires a value")?);
            }
            "-c" | "--chunks-per-side" => {
                chunks_per_side = args
                    .next()
                    .context("--chunks-per-side requires a value")?
                    .parse::<u16>()
                    .context("--chunks-per-side must be an integer")?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("--chunks-per-side must fit in 1..=255"))?;
                if chunks_per_side == 0 {
                    bail!("--chunks-per-side must be >= 1");
                }
            }
            "-s" | "--seed" => {
                seed = args
                    .next()
                    .context("--seed requires a value")?
                    .parse::<u64>()
                    .context("--seed must be an integer")?;
            }
            "--noise-scale" => {
                noise_scale = args
                    .next()
                    .context("--noise-scale requires a value")?
                    .parse::<f32>()
                    .context("--noise-scale must be a float")?;
            }
            "--height-scale" => {
                height_scale = args
                    .next()
                    .context("--height-scale requires a value")?
                    .parse::<f32>()
                    .context("--height-scale must be a float")?;
            }
            "--octaves" => {
                octaves = args
                    .next()
                    .context("--octaves requires a value")?
                    .parse::<u8>()
                    .context("--octaves must be an integer")?;
                if octaves == 0 {
                    bail!("--octaves must be >= 1");
                }
            }
            "--gain" => {
                gain = args
                    .next()
                    .context("--gain requires a value")?
                    .parse::<f32>()
                    .context("--gain must be a float")?;
            }
            "--lacunarity" => {
                lacunarity = args
                    .next()
                    .context("--lacunarity requires a value")?
                    .parse::<f32>()
                    .context("--lacunarity must be a float")?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => bail!("Unknown arg: {a}. Use --help."),
        }
    }

    Ok(Args {
        output: out,
        chunks_per_side,
        seed,
        noise_scale,
        height_scale,
        octaves,
        gain,
        lacunarity,
    })
}

fn print_usage() {
    eprintln!(
        "\
Generates a level file in this repo's binary format.

USAGE:
  cargo run --bin gen_level -- [options]

OPTIONS:
  -o, --output <path>           Output file path (default: levels/generated_level)
  -c, --chunks-per-side <n>     Number of chunks along X and Z (1..=255) (default: 1)
  -s, --seed <u64>              Seed (default: 0)
      --noise-scale <f32>       World noise scale (default: 0.035)
      --height-scale <f32>      Height multiplier (default: 1.0)
      --octaves <u8>            fBM octaves (default: 5)
      --gain <f32>              fBM gain per octave (default: 0.5)
      --lacunarity <f32>        fBM frequency multiplier (default: 2.0)
  -h, --help                    Print help

NOTES:
  - This generator outputs a flat chunk stack (y=0) because the current level format stores size^2 chunks.
  - Chunk voxel dimensions are fixed by the demo constants (CS=62, CS_P=64).
"
    );
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn hash2(seed: u64, x: i32, y: i32) -> u64 {
    let mut v = seed;
    v ^= (x as u32 as u64).wrapping_mul(0xA24BAED4963EE407);
    v ^= (y as u32 as u64).wrapping_mul(0x9FB21C651E98DF25);
    splitmix64(v)
}

#[inline]
fn u64_to_unit_f32(v: u64) -> f32 {
    let bits = (v >> 40) as u32; // top 24 bits
    (bits as f32) * (1.0 / ((1u32 << 24) as f32))
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn value_noise_2d(seed: u64, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let tx = fade(x - (x0 as f32));
    let ty = fade(y - (y0 as f32));

    let v00 = u64_to_unit_f32(hash2(seed, x0, y0));
    let v10 = u64_to_unit_f32(hash2(seed, x1, y0));
    let v01 = u64_to_unit_f32(hash2(seed, x0, y1));
    let v11 = u64_to_unit_f32(hash2(seed, x1, y1));

    let a = lerp(v00, v10, tx);
    let b = lerp(v01, v11, tx);
    lerp(a, b, ty)
}

fn fbm_2d(seed: u64, x: f32, y: f32, octaves: u8, gain: f32, lacunarity: f32) -> f32 {
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;

    for i in 0..octaves {
        let octave_seed = splitmix64(seed ^ (i as u64).wrapping_mul(0xD6E8FEB86659FD93));
        let n = value_noise_2d(octave_seed, x * frequency, y * frequency); // [0,1)
        let n = n * 2.0 - 1.0; // [-1,1)

        sum += n * amplitude;
        norm += amplitude;

        frequency *= lacunarity;
        amplitude *= gain;
    }

    if norm > 0.0 { sum / norm } else { 0.0 }
}

fn rle_encode_sparse_trailing_zeros(voxels: &[u8]) -> Vec<u8> {
    debug_assert_eq!(voxels.len(), CS_P3);

    let end = voxels
        .iter()
        .rposition(|&v| v != 0)
        .map(|i| i + 1)
        .unwrap_or(0);

    let mut out = Vec::<u8>::new();
    let mut i = 0usize;

    while i < end {
        let ty = voxels[i];
        let mut run = 1usize;
        while i + run < end && voxels[i + run] == ty && run < (u8::MAX as usize) {
            run += 1;
        }

        out.push(ty);
        out.push(run as u8);
        i += run;
    }

    out
}

fn write_level_file(path: &Path, args: &Args) -> Result<()> {
    let size = args.chunks_per_side as usize;
    let table_len = size * size;

    let mut chunk_rle: Vec<Vec<u8>> = Vec::with_capacity(table_len);

    for cz in 0..size {
        for cx in 0..size {
            let mut voxels = vec![0u8; CS_P3];

            for z in 1..=CS {
                for x in 1..=CS {
                    let wx = (cx * CS + (x - 1)) as f32;
                    let wz = (cz * CS + (z - 1)) as f32;

                    let n = fbm_2d(
                        args.seed,
                        wx * args.noise_scale,
                        wz * args.noise_scale,
                        args.octaves,
                        args.gain,
                        args.lacunarity,
                    );

                    let t = (n * 0.5 + 0.5).clamp(0.0, 1.0).powf(1.35);
                    let height = ((t * (CS as f32) * args.height_scale) as i32).clamp(0, CS as i32) as usize;

                    for y in 1..=height {
                        let idx = get_zxy_index(x, y, z);
                        voxels[idx] = 1;
                    }
                }
            }

            chunk_rle.push(rle_encode_sparse_trailing_zeros(&voxels));
        }
    }

    let mut bytes: Vec<u8> = Vec::new();
    bytes.push(args.chunks_per_side);
    bytes.resize(1 + table_len * 12, 0u8);

    for rle in &chunk_rle {
        bytes.extend_from_slice(rle);
    }

    let mut table_offset = 1usize;
    let mut data_offset = 1 + table_len * 12;

    for cz in 0..size {
        for cx in 0..size {
            let i = cz * size + cx;
            let rle = &chunk_rle[i];

            let key = get_xyz_key(cx as u8, 0, cz as u8);
            let begin = data_offset as u32;
            let sz = rle.len() as u32;

            bytes[table_offset..table_offset + 4].copy_from_slice(&key.to_le_bytes());
            bytes[table_offset + 4..table_offset + 8].copy_from_slice(&begin.to_le_bytes());
            bytes[table_offset + 8..table_offset + 12].copy_from_slice(&sz.to_le_bytes());

            table_offset += 12;
            data_offset += rle.len();
        }
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create dir: {}", parent.display()))?;
        }
    }

    fs::write(path, bytes).with_context(|| format!("write level file: {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    if !(0.0..=10.0).contains(&args.height_scale) {
        bail!("--height-scale out of supported range (0..=10)");
    }
    if !(0.0001..=10.0).contains(&args.noise_scale) {
        bail!("--noise-scale out of supported range (0.0001..=10)");
    }
    if !(0.0..=1.0).contains(&args.gain) {
        bail!("--gain out of supported range (0..=1)");
    }
    if args.lacunarity < 1.0 {
        bail!("--lacunarity must be >= 1");
    }

    write_level_file(&args.output, &args)?;
    eprintln!(
        "Wrote level: {} (chunks_per_side={}, seed={}, chunk_dims={}^3 incl padding)",
        args.output.display(),
        args.chunks_per_side,
        args.seed,
        CS_P
    );
    Ok(())
}
