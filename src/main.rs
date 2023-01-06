use clap::{Parser, Subcommand};
use log::{error, info, warn, LevelFilter};
use rogue::UserModuleType;
use simple_logger::SimpleLogger;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "rogue-cli", about)]
struct Cli {
    /// Enable debug output
    #[arg(short = 'd', long, global = true)]
    debug: bool,

    /// Module type (modfx, delfx, revfx, osc)
    #[arg(short = 'm', long, global = true)]
    module_type: Option<rogue::UserModuleType>,

    /// Slot (see `probe` for available slots per module)
    #[arg(short = 's', long, global = true)]
    slot: Option<u8>,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Print information about available modules and slots
    Probe,
    /// Load a module unit to the device
    Load {
        /// Module unit filename
        #[arg(short = 'u', long)]
        filename: PathBuf,
    },
    /// Validate module unit
    Check {
        /// Module unit filename
        #[arg(short = 'u', long)]
        filename: PathBuf,
    },
    /// Clear a module from the device
    Clear {
        /// Clear all slots
        #[arg(short = 'a')]
        all: bool,
    },
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn do_probe(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: Option<rogue::UserModuleType>,
) -> Result<()> {
    let modules = match module {
        Some(x) => vec![x],
        None => UserModuleType::variants(),
    };
    for module in modules {
        let info = dev.user_module_info(did, module)?;
        let info = match info {
            None => {
                println!("No information for module {:?}", module);
                continue;
            }
            Some(info) => {
                println!(">>> {:?} [ {} ]", module, info);
                info
            }
        };

        for slot in 0..info.available_slot_count {
            let slot_info = dev.user_slot_status(did, module, slot)?;
            let slot_info = match slot_info {
                None => {
                    println!("[{}]: empty", slot);
                    continue;
                }
                Some(slot_info) => slot_info,
            };
            println!("[{}] {}", slot, slot_info);
        }
    }
    Ok(())
}

fn find_first_free_slot(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: rogue::UserModuleType,
) -> Result<u8> {
    let info = dev.user_module_info(did, module)?;
    let info = match info {
        Some(i) => i,
        None => return Err(format!("no information for module {}", module).into()),
    };

    for slot in 0..info.available_slot_count {
        let slot_info = dev.user_slot_status(did, module, slot)?;
        if slot_info.is_none() {
            return Ok(slot);
        }
    }

    Err(format!("No free slot for module {} found", module).into())
}

fn do_load(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: Option<rogue::UserModuleType>,
    slot: Option<u8>,
    filename: &Path,
) -> Result<()> {
    let pkg = rogue::ModulePackage::from_file(filename)?;
    pkg.validate()?;

    if let Some(platform) = did.platform() {
        if platform != pkg.manifest.api.platform {
            warn!(
                "Device {} does not match target platform {}",
                platform, pkg.manifest.api.platform
            );
        }
    }

    if let Ok(api) = dev.user_api_version(did) {
        info!("Device API: {}", api);
        if api != pkg.manifest.api {
            warn!(
                "Requested API {} does not match device API {}",
                pkg.manifest.api, api
            );
        }
    }

    let mm = pkg.manifest.module;
    let module = match module {
        Some(m) => m,
        None => {
            info!("No -m option set, inferred module type {}", mm);
            mm
        }
    };

    if module != mm {
        warn!(
            "Requested module type {} does not match unit type {}",
            module, mm
        )
    }

    let slot = match slot {
        Some(s) => s,
        None => {
            info!("Finding a free slot...");
            let s = find_first_free_slot(dev, did, module)?;
            info!("No -s option set, using first free slot {}", s);
            s
        }
    };

    dev.upload_module(did, module, slot, filename)?;
    info!("Upload successful");
    Ok(())
}

fn do_check(filename: &Path) -> Result<()> {
    let pkg = rogue::ModulePackage::from_file(filename)?;
    pkg.validate()?;
    info!("Looks good, parsed manifest:\n{:#x?}", pkg.manifest);
    Ok(())
}

fn do_clear(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: Option<rogue::UserModuleType>,
    slot: Option<u8>,
    all: bool,
) -> Result<()> {
    let module = match module {
        None => return Err("--module must be specified".into()),
        Some(m) => m,
    };
    let slots = if let Some(s) = slot {
        s..s + 1
    } else {
        if !all {
            return Err("Either a slot or -a/--all must be specified".into());
        }

        // Could also use Clear User Module (0x1d)..
        match dev.user_module_info(did, module)? {
            None => return Err(format!("no information for module {}", module).into()),
            Some(info) => 0..info.available_slot_count,
        }
    };

    for slot in slots {
        info!("Clearing {} slot {}...", module, slot);
        dev.clear_user_slot(did, module, slot)?;
    }
    info!("Done.");
    Ok(())
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Determining the local UTC offset apparently is not thread-safe, so set a fixed one
    // for the logger at start up. Will be wrong on DST change, but meh.
    let utc_offset = time::UtcOffset::current_local_offset()?;
    let log_level = if args.debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    SimpleLogger::new()
        .with_level(log_level)
        .with_local_timestamps()
        .with_utc_offset(utc_offset)
        .env()
        .init()?;

    let mut dev = rogue::get_logue_device()?;
    let did = dev.identify()?;
    info!("Device identifies as: {:x?}", did);

    let ret = match args.commands {
        Commands::Probe => do_probe(&mut dev, &did, args.module_type),
        Commands::Load { filename } => {
            do_load(&mut dev, &did, args.module_type, args.slot, &filename)
        }
        Commands::Check { filename } => do_check(&filename),
        Commands::Clear { all } => do_clear(&mut dev, &did, args.module_type, args.slot, all),
    };

    if let Err(ref e) = ret {
        error!("Error during execution: {}", e)
    }
    ret
}
