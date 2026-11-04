use clap::{Parser, Subcommand};
use log::{debug, error, info, warn, LevelFilter};
use rogue::UserModuleType;
use simple_logger::SimpleLogger;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "rogue-cli", about)]
struct Cli {
    /// Enable debug output
    #[arg(short = 'd', long, global = true)]
    debug: bool,

    /// Input device name or number
    #[arg(short = 'i', long, global = true)]
    input_device: Option<String>,

    /// Output device name or number
    #[arg(short = 'o', long, global = true)]
    output_device: Option<String>,

    /// Module type (modfx, delfx, revfx, osc, genericfx)
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

fn uses_elf_units(did: &rogue::DeviceIdentity) -> bool {
    did.platform().is_some_and(|p| p.uses_elf_units())
}

/// Returns the slot count of a module, with a description of the module if the device has one.
fn module_slots(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: rogue::UserModuleType,
) -> Result<Option<(u8, String)>> {
    if let Some(n) = did.platform().and_then(|p| p.fixed_slot_count(module)) {
        return Ok(Some((n, format!("{}: {} slots", module, n))));
    }
    Ok(dev
        .user_module_info(did, module)?
        .map(|info| (info.available_slot_count, info.to_string())))
}

/// Returns a description of the unit in a slot, or None if the slot is empty.
fn slot_status(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: rogue::UserModuleType,
    slot: u8,
) -> Result<Option<String>> {
    if uses_elf_units(did) {
        Ok(dev.read_elf_unit(did, module, slot)?.map(|u| u.header.to_string()))
    } else {
        Ok(dev.user_slot_status(did, module, slot)?.map(|s| s.to_string()))
    }
}

fn do_probe(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: Option<rogue::UserModuleType>,
) -> Result<()> {
    let modules = match module {
        Some(x) => vec![x],
        None => did
            .platform()
            .map_or_else(UserModuleType::variants, |p| p.modules()),
    };
    for module in modules {
        let slots = match module_slots(dev, did, module)? {
            None => {
                println!("No information for module {:?}", module);
                continue;
            }
            Some((slots, info)) => {
                println!("\n>>> {:?} [ {} ]", module, info);
                slots
            }
        };

        for slot in 0..slots {
            match slot_status(dev, did, module, slot)? {
                None => println!("[{}]: empty", slot),
                Some(status) => println!("[{}] {}", slot, status),
            }
        }
    }
    Ok(())
}

fn find_first_free_slot(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: rogue::UserModuleType,
) -> Result<u8> {
    let slots = match module_slots(dev, did, module)? {
        Some((slots, _)) => slots,
        None => return Err(format!("no information for module {}", module).into()),
    };

    for slot in 0..slots {
        if slot_status(dev, did, module, slot)?.is_none() {
            return Ok(slot);
        }
    }

    Err(format!("No free slot for module {} found", module).into())
}

fn do_load_elf(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: Option<rogue::UserModuleType>,
    slot: Option<u8>,
    filename: &Path,
) -> Result<()> {
    let unit = rogue::ElfUnit::from_file(filename)?;
    unit.validate()?;
    info!("Unit: {}", unit.header);

    if let Some(platform) = did.platform() {
        if platform != unit.header.platform {
            warn!(
                "Device {} does not match target platform {}",
                platform, unit.header.platform
            );
        }
    }

    let um = unit.header.module;
    let module = match module {
        Some(m) => m,
        None => {
            info!("No -m option set, inferred module type {}", um);
            um
        }
    };

    if module != um {
        warn!(
            "Requested module type {} does not match unit type {}",
            module, um
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

    dev.upload_elf_unit(did, module, slot, unit)?;
    info!("Upload successful");
    Ok(())
}

fn do_load(
    dev: &mut rogue::Device,
    did: &rogue::DeviceIdentity,
    module: Option<rogue::UserModuleType>,
    slot: Option<u8>,
    filename: &Path,
) -> Result<()> {
    if rogue::ElfUnit::is_elf_file(filename)? {
        return do_load_elf(dev, did, module, slot, filename);
    }

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
        if api.version != pkg.manifest.api.version {
            warn!(
                "Requested API {} does not match device API {}",
                pkg.manifest.api.version, api.version
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
    if rogue::ElfUnit::is_elf_file(filename)? {
        let unit = rogue::ElfUnit::from_file(filename)?;
        unit.validate()?;
        info!("Looks good, parsed unit header:\n{:#x?}", unit.header);
        return Ok(());
    }

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
    if let Some(s) = slot {
        dev.clear_user_slot(did, module, s)?;
    } else {
        if !all {
            return Err("Either a slot or -a/--all must be specified".into());
        }

        dev.clear_user_module(did, module)?;
    }
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

    let mut dev = rogue::get_logue_device(&args.input_device, &args.output_device)?;
    let did = dev.identify()?;
    debug!("Device identifies as: {:x?}", did);

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
